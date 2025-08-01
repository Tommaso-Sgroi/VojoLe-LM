# from unsloth import FastLanguageModel
# from unsloth.chat_templates import get_chat_template
from time import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, infer_auto_device_map


from vllm import LLM, SamplingParams

model_path = os.path.join(os.getenv('FAST'), 'models', 'c4ai')
llm = LLM(model_path, tensor_parallel_size=4)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

"""
model_path = os.path.join(os.getenv('FAST'), 'models', 'c4ai')
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", local_files_only=True, token='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')
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, token='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')


messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

start = time()
outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
print('Elapsed time', time() - start)

# --------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.3-70B-Instruct",
    load_in_4bit = True,
    device_map = "balanced",
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
                               # EDIT HERE!
    {"from": "human", "value": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")


text_streamer = TextStreamer(tokenizer)

start = time()
generated_ids = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 64)

# Decode results
decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
accelerator.print(f"Finished in {time() - start:.2f} seconds")
for i, text in enumerate(decoded):
    accelerator.print(f"Result {i+1}: {text}")

"""