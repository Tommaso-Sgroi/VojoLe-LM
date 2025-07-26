from transformers import AutoModelForCausalLM, AutoTokenizer

print('Loading models')
model = AutoModelForCausalLM.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("/leonardo/home/userexternal/tsgroi00/Vojollm/VojoLe-LM/tokenizer")
print('Loaded models')

model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to("cuda")

print('Starting inference:\n')
generated_ids = model.generate(**model_inputs, max_length=30)
print(tokenizer.batch_decode(generated_ids)[0])
# '<s> The secret to baking a good cake is 100% in the preparation. There are so many recipes out there,'



