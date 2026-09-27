# LLM Watermarking

This project is a prototype for **watermarking text written by an LLM**. The
idea is to hide a secret, key-based signal in the way the model picks its
words. The text still reads normally, but anyone with the key can later test
whether that model and key produced it.

It's a research prototype from 2023. It explores two approaches, and neither
one is ready for production yet (see Status).

## Approach 1: keyed sampling (`watermarking/`)

The first approach follows the "distortion-free" idea from Kuditipudi et al.,
[*Robust Distortion-free Watermarks for Language Models*](https://arxiv.org/abs/2307.15593).

**Generating text.** Normally, an LLM picks each next word at random, based
on its probabilities. `generate_watermarked(model, tokenizer, prompt, key)`
replaces that random pick with a keyed one. At step *i*, it sorts the
possible next tokens by probability and chooses the one where the running
total comes closest to `key[i]` (see `decoder`). If the key values are
random, the output follows the model's own probabilities, so the text quality
doesn't change. The difference is that the chosen tokens now line up with the
secret key.

**Detecting the watermark.** `detect(text, key, tokenizer)` measures how well
the text's tokens line up with the key. Then it shuffles the key 1,000 times
and runs the same check each time, which shows what a normal score looks
like. The result is a p-value. A low p-value means the text matches the key
much better than chance.

`llm_watermarking_example.py` runs this on
[MPT-7B](https://huggingface.co/mosaicml/mpt-7b). It builds the key from an
MD5 hash of a use-case name.

## Approach 2: red and green token lists (`tests/example.py`)

The second approach sketches the "green list" method from Kirchenbauer et
al., [*A Watermark for Large Language Models*](https://arxiv.org/abs/2301.10226).
It also adds a twist: **each use case gets its own signature**. Every use
case has a SHA-256 signature, and each token is marked "green" or "red" by
hashing it together with that signature. `detect_use_case` then checks which
signature's pattern a text matches. This one runs on
[Falcon-7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct).

## Layout

```
llm-watermarking/
├── watermarking/__init__.py        decoder, generate_watermarked, detect, alignment_score
├── llm_watermarking_example.py     End-to-end generation demo on MPT-7B
└── tests/example.py                Red/green list prototype with use-case signatures
```

## Running

You'll need a CUDA GPU with enough memory for a 7B model in bf16 (about
15 GB).

```bash
pip install torch transformers numpy einops
```

```bash
python llm_watermarking_example.py
```

## Status

This is a proof of concept, and the current code has some known issues:

- `llm_watermarking_example.py` imports `watermarking2`, but the package here
  is called `watermarking`.
- `generate_watermarked` collects tokens in a list and then calls `.tolist()`
  on it. Also, `update_input` puts each new token at the front of the input
  and drops the last one, instead of adding it to the end. So after the first
  step, the model's context gets mixed up.
- `alignment_score` compares raw token IDs with key values as plain numbers.
  The paper uses a per-token score based on where each token sits in the
  sorted probabilities. That's what makes the test meaningful.
- In `tests/example.py`, tokens get their green or red tags *after* the text
  is generated, so the tags don't actually steer the model. Detection also
  uses Python's `hash()`, which changes every time Python starts. Because of
  that, detection can't work across separate runs.
