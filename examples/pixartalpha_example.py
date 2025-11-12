import time
import os
import torch
import torch.distributed
from transformers import T5EncoderModel
from xfuser import xFuserPixArtAlphaPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    is_dp_last_group,
    get_data_parallel_world_size,
    get_runtime_state,
    get_data_parallel_rank,
)

# Global timing storage
_phase_timings = {}


class TimingWrapper:
    """Wrapper to add timing to pipeline methods"""
    def __init__(self, pipe):
        self.pipe = pipe
        self.timings = {}
        
    def __getattr__(self, name):
        return getattr(self.pipe, name)
    
    def __call__(self, *args, **kwargs):
        torch.cuda.synchronize()
        
        # Store original methods
        original_encode = self.pipe.encode_prompt
        original_prepare_latents = self.pipe.prepare_latents
        original_sync_pipeline = self.pipe._sync_pipeline
        original_async_pipeline = self.pipe._async_pipeline
        
        # Wrap encode_prompt
        def timed_encode(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.time()
            result = original_encode(*args, **kwargs)
            torch.cuda.synchronize()
            self.timings['prompt_encoding'] = time.time() - start
            return result
        
        # Wrap prepare_latents
        def timed_prepare_latents(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.time()
            result = original_prepare_latents(*args, **kwargs)
            torch.cuda.synchronize()
            self.timings['latent_preparation'] = time.time() - start
            return result
        
        # Wrap _sync_pipeline
        def timed_sync_pipeline(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.time()
            result = original_sync_pipeline(*args, **kwargs)
            torch.cuda.synchronize()
            self.timings['warmup_denoising'] = time.time() - start
            return result
        
        # Wrap _async_pipeline
        def timed_async_pipeline(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.time()
            result = original_async_pipeline(*args, **kwargs)
            torch.cuda.synchronize()
            self.timings['pipefusion_denoising'] = time.time() - start
            return result
        
        # Apply wrappers
        self.pipe.encode_prompt = timed_encode
        self.pipe.prepare_latents = timed_prepare_latents
        self.pipe._sync_pipeline = timed_sync_pipeline
        self.pipe._async_pipeline = timed_async_pipeline
        
        # Time VAE decode (will be captured in the main call)
        torch.cuda.synchronize()
        vae_start = time.time()
        
        # Call the pipeline
        result = self.pipe(*args, **kwargs)
        
        torch.cuda.synchronize()
        total_time = time.time() - vae_start
        
        # Calculate VAE decode time as remainder
        accounted_time = sum(self.timings.values())
        self.timings['vae_decode_and_postprocess'] = max(0, total_time - accounted_time)
        
        # Restore original methods
        self.pipe.encode_prompt = original_encode
        self.pipe.prepare_latents = original_prepare_latents
        self.pipe._sync_pipeline = original_sync_pipeline
        self.pipe._async_pipeline = original_async_pipeline
        
        return result


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank
    
    # Phase timing dictionary
    phase_times = {}
    
    # Phase 1: Model Loading
    phase_start = time.time()
    text_encoder = T5EncoderModel.from_pretrained(engine_config.model_config.model, subfolder="text_encoder", torch_dtype=torch.float16)
    if args.use_fp8_t5_encoder:
        from optimum.quanto import freeze, qfloat8, quantize
        print(f"rank {local_rank} quantizing text encoder")
        quantize(text_encoder, weights=qfloat8)
        freeze(text_encoder)

    pipe = xFuserPixArtAlphaPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.float16,
        text_encoder=text_encoder,
    ).to(f"cuda:{local_rank}")
    model_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    pipe.prepare_run(input_config)
    phase_times['model_loading'] = time.time() - phase_start

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Wrap the pipeline to capture internal timings
    timed_pipe = TimingWrapper(pipe)
    output = timed_pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        use_resolution_binning=input_config.use_resolution_binning,
        guidance_scale=input_config.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    
    # Store timings for reporting
    phase_times.update(timed_pipe.timings)

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}_tc_{engine_args.use_torch_compile}"
    )
    if input_config.output_type == "pil":
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if pipe.is_dp_last_group():
            if not os.path.exists("results"):
                os.mkdir("results")
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                img_file = (
                    f"./results/pixart_alpha_result_{parallel_info}_{image_rank}.png"
                )
                image.save(img_file)
                print(img_file)

    if get_world_group().rank == get_world_group().world_size - 1:
        total_epoch = phase_times.get('model_loading', 0) + elapsed_time
        
        print("\n" + "="*80)
        print("TIMING BREAKDOWN")
        print("="*80)
        
        # Phase 1: Model Loading
        model_load_time = phase_times.get('model_loading', 0)
        print(f"1. Model Loading:              {model_load_time:7.2f} sec  ({model_load_time/total_epoch*100:5.1f}%)")
        
        # Phase 2: Prompt Encoding
        prompt_time = phase_times.get('prompt_encoding', 0)
        print(f"2. Prompt Encoding:            {prompt_time:7.2f} sec  ({prompt_time/total_epoch*100:5.1f}%)")
        
        # Phase 3: Latent Preparation
        latent_time = phase_times.get('latent_preparation', 0)
        print(f"3. Latent Preparation:         {latent_time:7.2f} sec  ({latent_time/total_epoch*100:5.1f}%)")
        
        # Phase 4: Warmup Denoising
        warmup_time = phase_times.get('warmup_denoising', 0)
        print(f"4. Warmup Denoising:           {warmup_time:7.2f} sec  ({warmup_time/total_epoch*100:5.1f}%)")
        
        # Phase 5: PipeFusion Denoising
        pipefusion_time = phase_times.get('pipefusion_denoising', 0)
        print(f"5. PipeFusion Denoising:       {pipefusion_time:7.2f} sec  ({pipefusion_time/total_epoch*100:5.1f}%)")
        
        # Phase 6: VAE Decode & Post-processing
        vae_time = phase_times.get('vae_decode_and_postprocess', 0)
        print(f"6. VAE Decode & Post-process:  {vae_time:7.2f} sec  ({vae_time/total_epoch*100:5.1f}%)")
        
        print("-"*80)
        print(f"TOTAL INFERENCE TIME:          {elapsed_time:7.2f} sec")
        print(f"TOTAL EPOCH TIME:              {total_epoch:7.2f} sec")
        print("="*80)
        print(f"Model Memory:                  {model_memory/1e9:7.2f} GB")
        print(f"Peak Memory:                   {peak_memory/1e9:7.2f} GB")
        print("="*80 + "\n")
    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
