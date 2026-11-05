# nanoGPT's model and optimizer, reading about 500M tokens in updates of 524,288 tokens,
# as autoresearch reads. nanoGPT's schedule for its 600,000 updates is scaled to these; its warmup
# keeps a small share of the run, and the rate decays to a tenth of its peak over all of it.
# FineWeb's sample in a tokenizer trained on it (prepare.py). ClimbMix, read until 2026-11-04, is
# non-commercial at its source.
dataset = 'fineweb'
out_dir = 'out/baseline'
# train.py's defaults are research's current best; this run, and those built on it, set back nanoGPT's.
# They run their updates without frame.py's time budget, as they did before the frame.
time_budget = 0
n_layer = 12
n_head = 12
n_embd = 768
value_embeddings = False
rotary = False
rms_norm = False
qk_norm = False
relu_squared = False
zero_output = False
separate_embeddings = False
residual_weights = False
logit_softcap = 0.0
stream_bfloat16 = False
rows_from_document_heads = False
count_table = ''
dispersion = 0.0
bigram_table = False
attention_gate = False
key_offset = False
dense_connections = False
backout = False
optimizer_name = 'adamw'
grad_clip = 1.0
muon_other_lr = 0.004
muon_embedding_lr = 0.6
attention = 'sdpa'
bigram_rows_per_token = 5
trigram_table = False
stem = False
xielu = False
document_attention = False
muon_cautious_adamw = False
muon_warmdown_share = 0.5
data_mix = ''
data_mix_late = ''
attention_window_start = 0
experts = 0
shared_experts = 0
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
