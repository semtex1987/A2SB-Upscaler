# A2SB Audio Restoration Wrapper

This repository provides a simple, Dockerized interface for [NVIDIA's Audio-to-Audio Schrödinger Bridges (A2SB)](https://github.com/NVIDIA/diffusion-audio-restoration), a diffusion-based model for audio restoration and upsampling.

It wraps the original inference code in a [Gradio](https://gradio.app/) web UI, making it easy to upload audio files and restore them using a graphical interface.

## Features

-   **Web Interface**: Easy-to-use Gradio UI for uploading and processing audio.
-   **Automatic Setup**: Docker handles all dependencies, including system libraries and Python packages.
-   **Pre-trained Models**: Automatically downloads the required checkpoints (`A2SB_twosplit_0.5_1.0_release.ckpt`, etc.).
-   **GPU Acceleration**: Leverages NVIDIA GPUs for fast inference.
-   **Persisted Output**: Restored audio files are saved to a local folder on your host machine.

## Prerequisites

-   **Docker** and **Docker Compose** installed on your machine.
-   **NVIDIA GPU** with appropriate drivers and `nvidia-container-toolkit` installed (required for CUDA support).

## Installation & Usage

1.  **Clone this repository**:
    ```bash
    git clone https://github.com/semtex1987/A2SB-Upscaler/tree/main
    cd A2SB-Upscaler
    ```

2.  **Start the container**:
    Run the following command to build the image and start the service:
    ```bash
    docker compose up --build
    ```
    *Note: The first run may take 5-10+ minutes as it builds the Docker image and downloads the model checkpoints.*

3.  **Access the Interface**:
    Open your web browser and go to:
    ```
    http://localhost:7860
    ```

4.  **Restore Audio**:
    -   Upload an audio file in the "Upload Audio" box.
    -   Adjust the "Steps" slider (higher steps = potentially better quality but slower processing).
    -   Click the "Submit" button.
    -   The restored audio will appear in the "Restored Result" box, and the file will be saved locally.

## Output

All restored audio files are automatically saved to the `restored_audio/` directory in the project root. This directory is mounted from the container, so your files persist even after the container stops.

## Troubleshooting

-   **"No such file or directory"**: Ensure you are running `docker compose` from the root of this repository.
-   **GPU Issues**: Verify that your NVIDIA drivers are up to date and that `nvidia-smi` works on your host. Ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.
-   **Port Conflicts**: If port 7860 is in use, modify the `ports` section in `docker-compose.yml` (e.g., `"8080:7860"`).

## Credits

This project wraps the work done by NVIDIA.
-   **Original Repository**: [NVIDIA/diffusion-audio-restoration](https://github.com/NVIDIA/diffusion-audio-restoration)
-   **Paper**: [Audio-to-Audio Schrödinger Bridges](https://arxiv.org/abs/2305.15083)
