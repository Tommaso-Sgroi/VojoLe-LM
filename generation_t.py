from transformers import AutoModelForCausalLM, AutoTokenizer
from time import time
from accelerate import Accelerator
from accelerate.utils import gather_object
import torch 
# Setup accelerator
accelerator = Accelerator()

# each GPU creates a string
device = accelerator.device

# Get memory info
if torch.cuda.is_available():
    torch.cuda.set_device(device)
    memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # in GB
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # in GB
    mem_info = f"Memory allocated: {memory_allocated:.2f} GB / {total_memory:.2f} GB"
else:
    mem_info = "CUDA not available."

# Collect info
message = [(
    f"GPU {accelerator.process_index} | "
    f"Device: {device} | "
    f"{mem_info}"
)]


# collect the messages from all GPUs
messages=gather_object(message)

# output the messages only on the main process with accelerator.print() 
accelerator.print(messages)


accelerator.print(f"Hello from GPU {accelerator.process_index}")

# Load model and tokenizer
accelerator.print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "CohereLabs/c4ai-command-a-03-2025",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/leonardo_scratch/fast/try25_sgroi/command_a-tokenizer")
accelerator.print("Model loaded.")

# Tokenize on CPU, then move to model's device using accelerator
inputs = tokenizer(
    ["The secret to baking a good cake is ", "Se il signor ocane ha un cane, allora di chi è il cane?"],
    return_tensors="pt",
    padding=True
)
inputs = accelerator.prepare(inputs)

# Generate
accelerator.print("Running inference...")
start = time()
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_length=30)

# Decode results
decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
accelerator.print(f"Finished in {time() - start:.2f} seconds")
for i, text in enumerate(decoded):
    accelerator.print(f"Result {i+1}: {text}")
