# Start from an official NVIDIA PyTorch image to ensure CUDA/GPU support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Clone the repository
RUN git clone https://github.com/semtex1987/diffusion-audio-restoration .

# 3. Install Python dependencies
RUN pip install --no-cache-dir \
    moviepy==1.0.3 \
    "jsonargparse[signatures]>=4.27.7" \
    scikit-image \
    torchlibrosa \
    pyyaml \
    numpy \
    scipy \
    matplotlib \
    librosa \
    soundfile \
    torchaudio \
    einops \
    pytorch_lightning \
    lightning \
    rotary_embedding_torch \
    tqdm \
    gradio

RUN pip install --no-cache-dir --no-deps ssr_eval

# 4. Create checkpoints directory
RUN mkdir -p ckpts

# 5. Download Checkpoints
RUN wget -O ckpts/A2SB_twosplit_0.5_1.0_release.ckpt https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_twosplit_0.5_1.0_release.ckpt
RUN wget -O ckpts/A2SB_onesplit_0.0_1.0_release.ckpt https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_onesplit_0.0_1.0_release.ckpt
RUN wget -O ckpts/A2SB_twosplit_0.0_0.5_release.ckpt https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge/resolve/main/ckpt/A2SB_twosplit_0.0_0.5_release.ckpt

# 6. Automate the Config Update
# FIX: Replaced 'if' statement with 'setdefault' to fix SyntaxError in one-liner
RUN python3 -c "import yaml; \
    path = 'configs/ensemble_2split_sampling.yaml'; \
    data = yaml.safe_load(open(path)); \
    data['model']['pretrained_checkpoints'] = [ \
        '/app/ckpts/A2SB_onesplit_0.0_1.0_release.ckpt', \
        '/app/ckpts/A2SB_twosplit_0.0_0.5_release.ckpt' \
    ]; \
    trainer = data.setdefault('trainer', {}); \
    trainer['strategy'] = 'auto'; \
    trainer['devices'] = 1; \
    trainer['accelerator'] = 'gpu'; \
    yaml.dump(data, open(path, 'w'), default_flow_style=False, sort_keys=False)"

# 7. Set Environment Variables
ENV CUDA_VISIBLE_DEVICES=0 \
    MKL_THREADING_LAYER=GNU \
    SLURM_NODEID=0 \
    SLURM_PROCID=0 \
    SLURM_LOCALID=0 \
    SLURM_JOB_ID=1 \
    SLURM_NTASKS=1 \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# 8. Setup Entrypoint
COPY app.py /app/app.py
CMD ["python3", "app.py"]
