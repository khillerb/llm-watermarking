import torch
import transformers
import watermarking2


name = "mosaicml/mpt-7b"

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.attn_config["attn_impl"] = "torch"
config.init_device = "cuda:0"  # For fast initialization directly on GPU!

model = transformers.AutoModelForCausalLM.from_pretrained(
    name,
    config=config,
    torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
    trust_remote_code=True,
)
tokenizer = transformers.AutoTokenizer.from_pretrained(name)

prompt = "Hi My Name is Kevin and I am a"

import hashlib


def hash_key(key):
    hash_obj = hashlib.md5(key.encode())
    return int(hash_obj.hexdigest(), 16) / (2**64 - 1)


key = torch.tensor([hash_key("storygeneration")])
response = watermarking2.generate_watermarked(
    model, tokenizer, prompt, key  # , max_length=100
)
print(response)
