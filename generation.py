from unsloth import FastLanguageModel

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length = 8192,
    load_in_4bit = True,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    # chat_template = "llama-3.1",
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"from": "human", "value": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
    {"from": "human", "value": "Hi how are you?"},
    {"from": "human", "value": "Pls, tell me that this test will be good"},
]

for i in range(len(messages)):
    inputs = tokenizer.apply_chat_template(messages[i], tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 1024, use_cache = True)

