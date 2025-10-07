import os
import sys
import logging
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import format_prompt_for_filename, validate_and_process_user_prompt
from ovi.utils.utils import get_arguments
from ovi.distributed_comms.util import get_world_size, get_local_rank, get_global_rank
from ovi.distributed_comms.parallel_states import (
    initialize_sequence_parallel_state,
    get_sequence_parallel_state,
    nccl_info,
)
from ovi.ovi_fusion_engine import OviFusionEngine

# ---- Apple Silicon runtime: force MPS + SDPA, no FlashAttention/xFormers ----
import warnings

# Kill CUDA/FA/xformers at the root (safe no-ops on Mac)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("USE_FLASH_ATTENTION", "0")
os.environ.setdefault("FLASH_ATTENTION", "0")
os.environ.setdefault("FLASH_ATTN", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Choose device once, globally
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("⚡ Using Metal/MPS backend (no CUDA, no FlashAttention).")
else:
    DEVICE = torch.device("cpu")
    print("💤 MPS unavailable; CPU fallback.")

# Prefer PyTorch SDPA/math paths; safe on MPS
try:
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
except Exception as e:
    warnings.warn(f"SDPA backend guard skipped: {e}")

# MPS stability hint
try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass
# ---------------------------------------------------------------------------


def _init_logging(rank: int) -> None:
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def main(config, args):
    world_size = get_world_size()
    global_rank = get_global_rank()
    local_rank = get_local_rank()

    # --- DO NOT map device to rank on Mac; keep the global DEVICE chosen above ---
    device = DEVICE

    # Only set CUDA device if CUDA exists (Linux/NVIDIA case). On Mac this is skipped.
    if torch.cuda.is_available():
        try:
            # torch.cuda.set_device(local_rank)  # disabled on Mac
            pass
        except Exception:
            pass

    sp_size = config.get("sp_size", 1)
    assert sp_size <= world_size and world_size % sp_size == 0, (
        "sp_size must be <= world_size and world_size must be divisible by sp_size."
    )

    _init_logging(global_rank)

    # Distributed init: NCCL for CUDA; Gloo otherwise (Mac/CPU)
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            rank=global_rank,
            world_size=world_size,
        )
    else:
        assert sp_size == 1, f"When world_size is 1, sp_size must also be 1, but got {sp_size}."
        # TODO: assert not sharding t5 etc...

    initialize_sequence_parallel_state(sp_size)
    logging.info(f"Using SP: {get_sequence_parallel_state()}, SP_SIZE: {sp_size}")

    args.local_rank = local_rank
    args.device = device

    # DType choice: fp16 on MPS (best support/perf), bf16 elsewhere
    if device.type == "mps":
        target_dtype = torch.float16
    else:
        target_dtype = torch.float16

    # ---- validate inputs before loading model ----
    text_prompt = config.get("text_prompt")
    image_path = config.get("image_path", None)
    assert config.get("mode") in ["t2v", "i2v", "t2i2v"], (
        f"Invalid mode {config.get('mode')}, must be one of ['t2v', 'i2v', 't2i2v']"
    )

    text_prompts, image_paths = validate_and_process_user_prompt(
        text_prompt, image_path, mode=config.get("mode")
    )

    if config.get("mode") != "i2v":
        logging.info(f"mode: {config.get('mode')}, setting all image_paths to None")
        image_paths = [None] * len(text_prompts)
    else:
        assert all(p is not None and os.path.isfile(p) for p in image_paths), (
            f"In i2v mode, all image paths must be provided.{image_paths}"
        )

    logging.info("Loading OVI Fusion Engine...")
    ovi_engine = OviFusionEngine(config=config, device=device, target_dtype=target_dtype)
    logging.info("OVI Fusion Engine loaded!")

    output_dir = config.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV data
    all_eval_data = list(zip(text_prompts, image_paths))

    # Get SP configuration
    use_sp = get_sequence_parallel_state()
    if use_sp:
        sp_size = nccl_info.sp_size
        sp_rank = nccl_info.rank_within_group
        sp_group_id = global_rank // sp_size
        num_sp_groups = world_size // sp_size
    else:
        # No SP: treat each (logical) process as its own group
        sp_size = 1
        sp_rank = 0
        sp_group_id = global_rank
        num_sp_groups = world_size

    # Data distribution - by SP groups
    total_files = len(all_eval_data)
    require_sample_padding = False

    if total_files == 0:
        logging.error("ERROR: No evaluation files found")
        this_rank_eval_data = []
    else:
        # Pad to match number of SP groups
        remainder = total_files % num_sp_groups
        if require_sample_padding and remainder != 0:
            pad_count = num_sp_groups - remainder
            all_eval_data += [all_eval_data[0]] * pad_count

        # Distribute across SP groups
        this_rank_eval_data = all_eval_data[sp_group_id::num_sp_groups]

    for _, (text_prompt, image_path) in tqdm(enumerate(this_rank_eval_data)):
        video_frame_height_width = config.get("video_frame_height_width", None)
        seed = config.get("seed", 100)
        solver_name = config.get("solver_name", "unipc")
        sample_steps = config.get("sample_steps", 50)
        shift = config.get("shift", 5.0)
        video_guidance_scale = config.get("video_guidance_scale", 4.0)
        audio_guidance_scale = config.get("audio_guidance_scale", 3.0)
        slg_layer = config.get("slg_layer", 11)
        video_negative_prompt = config.get("video_negative_prompt", "")
        audio_negative_prompt = config.get("audio_negative_prompt", "")

        for idx in range(config.get("each_example_n_times", 1)):
            generated_video, generated_audio, generated_image = ovi_engine.generate(
                text_prompt=text_prompt,
                image_path=image_path,
                video_frame_height_width=video_frame_height_width,
                seed=seed + idx,
                solver_name=solver_name,
                sample_steps=sample_steps,
                shift=shift,
                video_guidance_scale=video_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                slg_layer=slg_layer,
                video_negative_prompt=video_negative_prompt,
                audio_negative_prompt=audio_negative_prompt,
            )

            if sp_rank == 0:
                formatted_prompt = format_prompt_for_filename(text_prompt)
                dims = "x".join(map(str, video_frame_height_width)) if video_frame_height_width else "NA"
                output_path = os.path.join(
                    output_dir, f"{formatted_prompt}_{dims}_{seed+idx}_{global_rank}.mp4"
                )
                save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
                if generated_image is not None:
                    generated_image.save(output_path.replace(".mp4", ".png"))


if __name__ == "__main__":
    args = get_arguments()
    config = OmegaConf.load(args.config_file)
    # For non-CUDA environments, ensure harmless rank defaults so SP math works
    if not torch.cuda.is_available():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
    main(config=config, args=args)