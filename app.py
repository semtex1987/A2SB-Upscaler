import gradio as gr
import subprocess
import os
import numpy as np
from scipy.signal import butter, sosfilt
from pydub import AudioSegment

# Directories
INPUT_DIR = "/app/inputs"
OUTPUT_DIR = "/app/outputs"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def butter_lowpass_filter(data, cutoff, fs, order=10):
    """
    Applies a steep Butterworth low-pass filter to a numpy array.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Check for valid cutoff relative to Nyquist
    if normal_cutoff >= 1:
        return data # No filtering needed if cutoff is above Nyquist
    
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    y = sosfilt(sos, data)
    return y

def apply_lowpass_to_segment(segment, cutoff_freq_hz):
    """
    Converts PyDub segment to numpy, filters it, and converts back.
    """
    # Convert to numpy array
    channel_data = np.array(segment.get_array_of_samples())
    
    # Handle stereo/mono if needed (though we split before this usually)
    if segment.channels == 2:
        channel_data = channel_data.reshape((-1, 2))
    
    fs = segment.frame_rate
    
    # Apply filter
    # If stereo, we'd need to filter cols separately, but we are working on mono chunks here
    filtered_data = butter_lowpass_filter(channel_data, cutoff_freq_hz, fs)
    
    # Cast back to appropriate integer type (usually int16)
    filtered_data = filtered_data.astype(channel_data.dtype)
    
    # Create new segment
    return segment._spawn(filtered_data.tobytes())

def run_a2sb_inference(input_path, output_path, steps):
    """
    Runs the A2SB inference script on a single file.
    """
    script_name = "A2SB_upsample_api.py"
    
    # We REMOVED the --cutoff flag here because we are applying the filter manually
    command = [
        "python3", script_name,
        "-f", input_path,
        "-o", output_path,
        "-n", str(int(steps))
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"

    print(f"Running inference on {input_path}...")
    
    result = subprocess.run(
        command, 
        check=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True,
        cwd="/app/inference",
        env=env
    )
    return result

def restore_audio(input_file, steps, cutoff_choice):
    if not input_file:
        return None

    # Parse cutoff choice "4kHz" -> 4000
    cutoff_hz = int(cutoff_choice.lower().replace("khz", "")) * 1000
    print(f"Selected Cutoff: {cutoff_hz} Hz")

    # 1. Load Audio
    try:
        audio = AudioSegment.from_file(input_file)
        print(f"Loaded audio. Channels: {audio.channels}, Rate: {audio.frame_rate}Hz")
    except Exception as e:
        raise gr.Error(f"Failed to load audio: {e}")

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    final_output_path = os.path.join(OUTPUT_DIR, f"{base_name}_restored.wav")

    # Helper to process a single channel
    def process_channel(segment, channel_name):
        # A. Apply Lowpass Filter
        print(f"[{channel_name}] Applying {cutoff_hz}Hz Lowpass Filter...")
        filtered_segment = apply_lowpass_to_segment(segment, cutoff_hz)
        
        # B. Export temp file
        temp_in = os.path.join(INPUT_DIR, f"temp_{channel_name}.wav")
        temp_out = os.path.join(OUTPUT_DIR, f"temp_{channel_name}_restored.wav")
        filtered_segment.export(temp_in, format="wav")
        
        # C. Run AI Restoration
        print(f"[{channel_name}] Running A2SB Restoration...")
        run_a2sb_inference(temp_in, temp_out, steps)
        
        return temp_out

    try:
        if audio.channels == 1:
            # --- MONO ---
            restored_path = process_channel(audio, "mono")
            return restored_path

        elif audio.channels == 2:
            # --- STEREO ---
            print("Stereo detected. Splitting channels.")
            channels = audio.split_to_mono()
            
            # Process L and R
            out_l_path = process_channel(channels[0], "L")
            out_r_path = process_channel(channels[1], "R")

            # Recombine
            print("Recombining channels...")
            restored_l = AudioSegment.from_file(out_l_path)
            restored_r = AudioSegment.from_file(out_r_path)
            
            restored_stereo = AudioSegment.from_mono_audiosegments(restored_l, restored_r)
            restored_stereo.export(final_output_path, format="wav")
            print("Success!")
            return final_output_path
        else:
            raise gr.Error(f"Unsupported channels: {audio.channels}")

    except subprocess.CalledProcessError as e:
        print("STDERR:", e.stderr)
        raise gr.Error(f"Restoration failed: {e.stderr}")
    except Exception as e:
        print("Error:", str(e))
        raise gr.Error(f"Processing error: {str(e)}")

# Custom CSS
custom_css = "body { background-color: #121212; color: white; }"

# Build Interface
iface = gr.Interface(
    fn=restore_audio,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio"),
        gr.Slider(minimum=10, maximum=200, value=50, step=10, label="Steps (Quality)"),
        # New Dropdown Selector
        gr.Dropdown(
            choices=["4kHz", "14kHz", "16kHz"], 
            value="14kHz", 
            label="Input Lowpass Filter (Cutoff)"
        )
    ],
    outputs=gr.Audio(label="Restored Result"),
    title="NVIDIA A2SB Stereo Restorer",
    description="Splits stereo audio, applies a strict low-pass filter to simulate bandwidth loss, restores it using A2SB, and recombines.",
    css=custom_css
)

iface.launch(server_name="0.0.0.0", server_port=7860)
