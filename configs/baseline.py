# nanoGPT's model and optimizer, reading about 500M tokens in updates of 524,288 tokens,
# as autoresearch reads. nanoGPT's schedule for its 600,000 updates is scaled to these; its warmup
# keeps a small share of the run, and the rate decays to a tenth of its peak over all of it.
# The Japanese and English of language-model's v6 corpora, which prepare_bilingual.py rewrites in a tokenizer
# trained on them. ClimbMix, read until 2026-11-04, is non-commercial at its source.
dataset = 'jaen/stable'
out_dir = 'out/baseline'
batch_size = 64
gradient_accumulation_steps = 8
block_size = 1024
max_iters = 954
lr_decay_iters = 954
warmup_iters = 48
# Validation is only a monitor, so once at the end is enough.
eval_interval = 954
eval_iters = 20
log_interval = 10
