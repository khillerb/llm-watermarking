import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

model_path = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")

import hashlib

# Define the unique identifiers for each use case
use_cases = ["use_case_1", "use_case_2", "use_case_3", "default"]

# Generate a unique signature for each use case
signatures = {
    use_case: hashlib.sha256(use_case.encode()).digest() for use_case in use_cases
}


def generate_watermarked(
    prompt, num_tokens=100, gamma=0.5, delta=2.0, use_case="default"
):
    # Get the signature for the specified use case
    signature = signatures.get(use_case, signatures["default"])

    encoded = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    output = model.generate(encoded, max_length=num_tokens + len(encoded[0]))
    text = tokenizer.decode(output[0])

    tokens = tokenizer.tokenize(text)
    watermarked = []

    for i, token in enumerate(tokens):
        if i < len(encoded[0]):
            watermarked.append(token)
            continue

        # Use the signature to influence the pattern of red and green tokens
        seed = hash(token) ^ int.from_bytes(signature, "big")
        green_list = bool(seed % 2)

        if green_list:
            watermarked.append(f"<green>{token}</green>")
        else:
            logits = model(encoded, return_dict=True).logits
            green_mask = (logits > logits[0, -1] - delta).float()
            logits = logits + (green_mask * delta)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            watermarked.append(f"<red>{token}</red>")

    return tokenizer.convert_tokens_to_string(watermarked)


def detect_use_case(watermarked_text, prompt):
    # Tokenize the watermarked text and the original prompt
    tokens = tokenizer.tokenize(watermarked_text)
    prompt_tokens = tokenizer.tokenize(prompt)

    # Determine the start index of the generated portion of the text
    start_index = len(prompt_tokens)

    # Extract the pattern of red and green tokens in the generated portion of the text
    input_pattern = [bool(hash(token) % 2) for token in tokens[start_index:]]

    # Compare the input pattern against the expected pattern for each use case
    for use_case, signature in signatures.items():
        expected_pattern = [
            bool(hash(token) ^ int.from_bytes(signature, "big") % 2)
            for token in tokens[start_index:]
        ]
        if input_pattern == expected_pattern:
            return use_case

    # Return None if no matching use case is found
    return None


id = "test"
prompt = """System: You are an AI assistant, here to provide helpful responses to the user.
User: Write a story about a pissed off elephant starting with Once upon a time.
Assistant:"""

# Generate watermarked text for a specific use case
watermarked = generate_watermarked(
    prompt, num_tokens=150, gamma=0.5, delta=2.0, use_case="default"
)
print(watermarked)

# Detect the use case from the watermarked text
use_case = detect_use_case(watermarked, prompt)
print(use_case)

# Remove the tags from the watermarked text
decoded_text = re.sub(r"<red>|</red>|<green>|</green>", "", watermarked)

# Convert the decoded text to a readable output
tokens = tokenizer.tokenize(decoded_text)
readable_output = tokenizer.convert_tokens_to_string(tokens)
print(readable_output)
