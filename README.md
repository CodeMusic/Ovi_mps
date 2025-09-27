# Ovi
---

## 🚀 Features
- Feature 1
- Feature 2
- Feature 3

---

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/character-ai/Ovi.git

# Navigate into the project directory
cd Ovi

# Install dependencies
virtualenv ovi-env
source ovi-env/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -r requirements.txt
```

## Download Weights
We use open-sourced checkpoints from Wan and MMAudio, and thus we will need to download them from huggingface
```
# optional can specific --output-dir to download to a specific directory, but if a custom directory is used, the inference yaml has to be updated with the custom directory as shown below
python3 download_weights.py

OR

python3 download_weights.py --output-dir custom_dir
```

## Run Examples
Inference parameters are controlled via a yaml file, for example `ovi/configs/inference/inference_fusion.yaml`
```
output_dir: "<path to save generated outputs>"  # default: ./outputs
ckpt_dir: "<path to model checkpoints>"                # default: ./ckpts

num_steps: "<number of sampling steps>"                # default: 50
solver_name: "<sampler/solver algorithm>"              # default: unipc
shift: "<temporal shift factor>"                       # default: 5.0
sp_size: "<spatial size multiplier>"                   # default: 1

audio_guidance_scale: "<strength of audio conditioning>"  # default: 3.0
video_guidance_scale: "<strength of video conditioning>"  # default: 4.0

shard_text_model: "<whether to shard text model across devices>"    # default: False
shard_fusion_model: "<whether to shard fusion model across devices>" # default: False

video_negative_prompt: "<undesired artifacts to avoid in video>"    # default: common artifacts (jitter, bad hands, blur, etc.)
audio_negative_prompt: "<undesired artifacts to avoid in audio>"    # default: common artifacts (robotic, muffled, echo, etc.)

seed: "<random seed for reproducibility>"             # default: 100
aspect_ratio: "<video aspect ratio>"                  # default: 9:16  (choices: [9:16, 16:9, 1:1])

text_prompt: "<either raw text prompt or path to TSV/JSON with prompts>"  
# default: "A short description of the scene..." or /path/to/prompts.tsv
t2v_only: "<generate only text-to-video (ignore audio)>"             # default: True
slg_layer: "<which layer to apply SLG guidance>"                     # default: 9
```
fill in description of each parameter...

### a. single process (for a single GPU, text_prompt can be a single string input or a csv file containing prompts and images)
```
python3 inference.py --config-file ovi/configs/inference/inference_fusion.yaml
```

### b. multiprocess (for multiple GPUs, where we can run samples in parallel)
```
torchrun --nnodes 1 --nproc_per_node 8 inference.py --config-file ovi/configs/inference/inference_fusion.yaml
```




