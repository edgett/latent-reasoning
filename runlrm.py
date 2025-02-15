import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if CUDA is available and use GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Device: {device}')

# Load model and tokenizer
model_name = "tomg-group-umd/huginn-0125"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Encode input and move to device
input_text = "The capital of Westphalia is"
input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=True).to(device)

# Set model to evaluation mode
model.eval()

# Generate text
with torch.no_grad():
    outputs = model.generate(input_ids, max_length=32)

# Decode and print output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)
