import subprocess, os
from time import sleep
from sys import executable as python_exe

def print_gpu_status(tag=""):
    print(f"\n--- NVIDIA-SMI {tag} ---")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("nvidia-smi not found. Ensure NVIDIA drivers are installed.")

# List of argument sets for text_completition_finetune.py
# Each dictionary corresponds to a run
runs = [
    {
        "--model_path": os.path.join(os.getenv('FAST', ''), 'models', "Meta-Llama-31-8B"), 
        "--save_model_path": os.path.join(os.getenv('WORK'), "VojoLe-LM", "finetune_outputs"), 
        # "--load_in_4bit": '', 
        # "--load_in_8bit": '', 
    },

]
"""
    {
        "--model_path": os.path.join(os.getenv('FAST', ''), 'models', "Minerva-7B-base"), 
        "--save_model_path": os.path.join(os.getenv('WORK'), "VojoLe-LM", "finetune_outputs"), 
        # "--load_in_4bit": '', 
        # "--load_in_8bit": '', 
    },
    {
        "--model_path": os.path.join(os.getenv('FAST', ''), 'models', "Mistral-7B"), 
        "--save_model_path": os.path.join(os.getenv('WORK'), "VojoLe-LM", "finetune_outputs"), 
        # "--load_in_4bit": '', 
        # "--load_in_8bit": '', 
    },
"""

if __name__ == '__main__':
    script_path = "finetune.text_completion_finetune"
    processes = []
    files = []
    for i, arg_set in enumerate(runs):
        MODEL_NAME = runs[i]["--model_path"].split("/")[-1] 
        cmd = [python_exe, "-m", script_path]

        # Convert dictionary into command line arguments
        for k, v in arg_set.items():
            cmd.append(k)
            if v != "":
                cmd.append(str(v))
    
        print(f"\n=== Running batch {i+1}/{len(runs)}: {' '.join(cmd)} ===")
        print(f'Command: `{" ".join(cmd)}`')

        # Run the command and wait until it finishes
        stdout_file = f"logs/{MODEL_NAME}.out.log"
        stderr_file = f"logs/{MODEL_NAME}.err.log"
        files.append((open(stdout_file, "w+"), open(stderr_file, "w+")))
        processes.append(subprocess.Popen(cmd, stdout=files[-1][0], stderr=files[-1][1]))

    sleep(60 * 20) # sleep 20 minutes so the models are linkely going to be finetuned

    print_gpu_status()

    for i, process in enumerate(processes):
        process.wait()  # Wait for completion

        if process.returncode != 0:
            print(f"Batch {i+1} exited with errors!\n`{runs[i]}`")
        else:
            print(f"Batch {i+1} finished successfully.\n`{runs[i]}`")
        for f in files[i]:
            f.close()

    print("\nAll batches completed.")
