import torch
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda")


def decoder(key, logits):
    # Get token probabilities
    probs = torch.softmax(logits, dim=-1)

    # Inverse CDF
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Inverse transform sampling
    idx = (cum_probs - key).abs().argmin()
    return sorted_ids[idx]


def get_logits(outputs):
    return outputs[0][-1]


# def update_input(input_ids, next_token_id):
#     return torch.cat([next_token_id, input_ids[:-1]], dim=-1)
def update_input(input_ids, next_token_id):
    next_token_id = next_token_id.unsqueeze(1)
    return torch.cat([next_token_id, input_ids[:, :-1]], dim=-1)


def generate_watermarked(model, tokenizer, prompt, key, max_length=100):
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)

    output = []

    # Generate watermarked text
    for i in range(max_length):
        # Get next token logits
        outputs = model(input_ids)
        next_token_logits = get_logits(outputs)

        # Sample next token
        next_token_id = decoder(key[i], next_token_logits)

        # Append to output
        output.append(next_token_id)

        # Update input
        input_ids = update_input(input_ids, next_token_id)

    return tokenizer.decode(output.tolist()[0])


def detect(text, key, tokenizer, num_samples=1000):
    # Tokenize text
    input_ids = tokenizer.encode(text)

    # Compute test statistic
    stat = alignment_score(input_ids, key)

    # Generate null distribution
    stats = []
    for _ in range(num_samples):
        key_perm = np.random.permutation(key)
        stats.append(alignment_score(input_ids, key_perm))

    # Compute p-value
    p_val = (1 + np.sum(np.array(stats) <= stat)) / (num_samples + 1)

    return p_val


def alignment_score(ids, key):
    # Permute text token indices
    perm = np.random.permutation(len(ids))
    ids_perm = [ids[i] for i in perm]

    # Compute score
    score = np.sum(np.abs(np.array(ids_perm) - np.array(key)))
    return score


__all__ = [
    "alignment_score",
    "detect",
    "generate_watermarked",
    "update_input",
    "get_logits",
    "decoder",
]
