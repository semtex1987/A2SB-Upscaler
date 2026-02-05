import gradio as gr
import subprocess
import os

# Ensure output directory exists
OUTPUT_DIR = "/app/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def restore_audio(input_file, steps):
    if not input_file:
        return None
    
    # Create output path
    base_name = os.path.basename(input_file)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_restored.wav"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Clean up previous run
    if os.path.exists(output_path):
        os.remove(output_path)

    print(f"Processing: {input_file} -> {output_path} with {steps} steps...")

    # FIX: The script is inside the 'inference' folder
    script_name = "A2SB_upsample_api.py"
    
    # We run the command from inside the 'inference' directory
    # We assume the input/output paths are absolute (which they are)
    command = [
        "python3", script_name,
        "-f", input_file,
        "-o", output_path,
        "-n", str(int(steps))
    ]

    # FIX: We need to tell Python where to look for the main modules (in /app)
    # since we are running from /app/inference
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"

    try:
        result = subprocess.run(
            command, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            cwd="/app/inference",  # Run from the inference subdirectory
            env=env                # Pass the modified environment
        )
        print("STDOUT:", result.stdout)
        
        if not os.path.exists(output_path):
            raise gr.Error("Script finished but output file is missing.")
            
        return output_path
    
    except subprocess.CalledProcessError as e:
        print("STDERR:", e.stderr)
        # Debugging: If the file is still not found, list the directory
        if "No such file" in e.stderr or e.returncode == 2:
             print(f"Listing /app/inference to debug: {os.listdir('/app/inference')}")
        raise gr.Error(f"Restoration failed: {e.stderr}")

custom_css = "body { background-color: #121212; color: white; }"

iface = gr.Interface(
    fn=restore_audio,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio"),
        gr.Slider(minimum=10, maximum=200, value=50, step=10, label="Steps")
    ],
    outputs=gr.Audio(label="Restored Result"),
    title="NVIDIA A2SB Restorer",
    description="Upload audio to restore. Higher steps = better quality.",
    css=custom_css
)

iface.launch(server_name="0.0.0.0", server_port=7860)
