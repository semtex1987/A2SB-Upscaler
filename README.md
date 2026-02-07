# **A2SB Audio Restoration Wrapper**

This repository provides a simple, Dockerized interface for [NVIDIA's Audio-to-Audio Schrödinger Bridges (A2SB)](https://github.com/NVIDIA/diffusion-audio-restoration), a diffusion-based model for audio restoration and upsampling.

It wraps the original inference code in a [Gradio](https://gradio.app/) web UI, adding robust support for stereo audio and simulated bandwidth degradation to ensure the best possible restoration results.

## **Features**

- **Web Interface**: Easy-to-use Gradio UI for uploading and processing audio.  
- **Stereo Support**: Automatically detects stereo audio, splits the Left/Right channels, processes them individually using the A2SB model, and recombines them into a high-fidelity stereo output.  
- **Bandwidth Control**: Includes a selectable Low-Pass Filter (4kHz, 14kHz, 16kHz) to simulate specific bandwidth limitations, giving the model a clear starting point for restoration.  
- **Automatic Setup**: Docker handles all dependencies, including system libraries, Python packages, and NVIDIA drivers.  
- **Pre-trained Models**: Automatically downloads the required checkpoints (A2SB\_twosplit\_0.5\_1.0\_release.ckpt, etc.) during the first build.  
- **GPU Acceleration**: Leverages NVIDIA GPUs for fast inference using Passthrough or vGPU modes.  
- **Persisted Output**: Restored audio files are saved to a local folder on your host machine.

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
    -   **Web Interface**: Easy-to-use Gradio UI for uploading and processing audio.
    -   **Automatic Setup**: Docker handles all dependencies, including system libraries and Python packages.
    -   **Pre-trained Models**: Automatically downloads the required checkpoints (`A2SB_twosplit_0.5_1.0_release.ckpt`, etc.).
    -   **GPU Acceleration**: Leverages NVIDIA GPUs for fast inference.
    -   **Persisted Output**: Restored audio files are saved to a local folder on your host machine.

## **Output**

All restored audio files are automatically saved to the restored\_audio/ directory in the project root. This directory is mounted from the container, so your files persist even after the container stops.

## **Troubleshooting**

* **"Operation not supported" (vGPU)**: If you are using an NVIDIA vGPU (GRID driver), you may encounter CUDA errors. Switch your VM to PCIe Passthrough mode for best compatibility, or ensure your Docker environment variables are set to disable P2P/Unified Memory features.  
* **"No such file or directory"**: Ensure you are running docker compose from the root of this repository.  
* **GPU Issues**: Verify that your NVIDIA drivers are up to date and that nvidia-smi works on your host. Ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.  
* **Port Conflicts**: If port 7860 is in use, modify the ports section in docker-compose.yml (e.g., "8080:7860").

## **Credits**

This project wraps the work done by NVIDIA.

* **Original Repository**: [NVIDIA/diffusion-audio-restoration](https://github.com/NVIDIA/diffusion-audio-restoration)  
* **Paper**: [Audio-to-Audio Schrödinger Bridges](https://arxiv.org/abs/2305.15083)
