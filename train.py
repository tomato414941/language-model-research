"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import json
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from averaging import WeightAverage, averaged_updates
from count_table import CountTable
from model import GPTConfig, GPT, choose_attention
from frame import TIME_BUDGET, UNCOUNTED_UPDATES, CoreTokenizer, evaluate_bpb, evaluate_core, save_core_results
from muon import configure_muon, set_schedule_at
from rows import bos_token, document_heads, whole_row_heads

started = time.time()

# -----------------------------------------------------------------------------
# The defaults are research's current best (docs/research/README.md), the run that a candidate must beat. A candidate
# changes them here, directly. The frame that every run follows, its data, training time and evaluation, is frame.py's.
# configs/ holds earlier runs, which set back what they need.
# I/O
out_dir = 'out/run'
eval_interval = 1000000 # with a time budget, the loss is estimated at the start and at the end
log_interval = 10
eval_iters = 20
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
average_last = 0 # above zero, how many of the last weights, average_every updates apart, to average for out_dir/mixed
average_every = 32
average_share = 0.55 # the average's share of the mixed weights; the very last weights take the rest
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'fineweb' # frame.py's allowed data
decay_dataset = '' # with Muon's schedule, the dataset that training reads once the learning rate begins to fall
gradient_accumulation_steps = 4 # used to simulate larger batch sizes
batch_size = 128 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
time_budget = TIME_BUDGET # seconds of training, frame.py's; at or below zero, max_iters updates instead
evaluate = True # does the run end with frame.py's evaluation, CORE and the reference validation BPB?
# model
n_layer = 10
n_head = 5
n_embd = 640
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
value_embeddings = True # do we add a table per token to the values of some layers?
rotary = True # do positions come from rotary embeddings instead of a table?
rms_norm = True # do we use RMSNorm with no weights instead of LayerNorm?
qk_norm = True # do we normalize queries and keys before they meet?
relu_squared = True # do we use ReLU squared in the MLP instead of GELU?
zero_output = True # do the last weights of attention and the MLP start at zero?
separate_embeddings = True # does one table read the tokens and another predict them?
residual_weights = True # does each block take the stream and the embedded tokens by weights it learns?
logit_softcap = 15.0 # above zero, the logits are squashed smoothly into plus or minus this
stream_bfloat16 = True # do the values through the layers stay in bfloat16, when training in it on a GPU?
rows_from_document_heads = True # do training rows begin at the head of a document, and read on from there?
count_table = 'data/count-fineweb' # the directory of a COUNT table that count_table.py built, which the input then takes; '' for none
dispersion = 0.1 # above zero, the weight of Dispersion Loss on the directions of each row's token vectors
bigram_table = True # does a table looked up by each token and the one before it add to every block's input?
attention_gate = True # is each head's output scaled by a gate read from the first channels of the input?
key_offset = True # do half of each head's channel pairs stay unturned, the keys taking them from the token before?
dense_connections = True # do the last two blocks add mixtures of earlier outputs, weighted per position (MUDD)?
backout = True # is a share of the stream after two thirds of the layers taken out before the prediction?
bigram_rows_per_token = 5 # the rows of the tables of pairs and of triples, per token of the vocabulary
trigram_table = False # does a table looked up by each token and the two before it add to every block's input?
stem = False # do half of the MLPs scale their widened values by a table looked up by each token (STEM)?
xielu = False # is xIELU the MLP's activation, in the place of ReLU squared or GELU?
document_attention = False # does attention stay within each document, which begins with the tokenizer's BOS?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 1000000 # total number of training iterations; with a time budget, only a bound
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# 'adamw' is the recipe above; 'muon' uses the settings below, and its schedule comes from muon.py
optimizer_name = 'muon'
muon_lr = 0.04 # learning rate of the matrices of the blocks
muon_other_lr = 0.0044 # learning rate of every other weight, except for the value embeddings
muon_value_embedding_lr = 0.1
muon_embedding_lr = 0.66 # learning rate of a token table that only reads
muon_scalar_lr = 0.5 # learning rate of the x0 weights; the resid weights take a hundredth of it
muon_weight_decay = 0.2 # at the start of training, falling to zero at the end
muon_polar_express = True
muon_neuron_normalization = True
muon_cautious_decay = True
muon_cautious_adamw = False # does AdamW skip each step that points against the gradient (Cautious AdamW)?
muon_warmdown_share = 0.5 # the last share of training over which the learning rate falls to zero
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
seed = 1337
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
attention = 'fa2' # how attention runs on a GPU: 'sdpa', 'fa2', 'fa3' or 'fa4'; see choose_attention in model.py
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
# the updates after which training keeps the weights that the mixed weights average, checked before any work
if average_last and time_budget > 0:
    raise ValueError("averaging the last weights needs a run of max_iters updates; set time_budget=0")
averaged = averaged_updates(max_iters, average_last, average_every) if average_last else set()

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)

def row_starts(directory):
    """Where a dataset's training rows may begin, or None if they may begin anywhere."""
    if not rows_from_document_heads:
        return None
    train_tokens = os.path.join(directory, 'train.bin')
    heads = document_heads(train_tokens, os.path.join(directory, 'tokenizer.json'))
    return whole_row_heads(heads, os.path.getsize(train_tokens) // 2, block_size)

train_dir = data_dir
row_heads = row_starts(train_dir)

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(train_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    if split == 'train' and row_heads is not None:
        ix = row_heads[torch.randint(len(row_heads), (batch_size,)).numpy()]
    else:
        ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias,
                  vocab_size=None, dropout=dropout, value_embeddings=value_embeddings, rotary=rotary,
                  rms_norm=rms_norm, qk_norm=qk_norm, relu_squared=relu_squared,
                  zero_output=zero_output, separate_embeddings=separate_embeddings,
                  residual_weights=residual_weights, logit_softcap=logit_softcap, count=bool(count_table),
                  dispersion=dispersion, bigram_table=bigram_table, attention_gate=attention_gate,
                  key_offset=key_offset, dense_connections=dense_connections,
                  backout=backout, bigram_rows_per_token=bigram_rows_per_token, trigram_table=trigram_table,
                  stem=stem, xielu=xielu,
                  document_bos=bos_token(os.path.join(data_dir, 'tokenizer.json')) if document_attention else -1,
                  ) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    for k in ['value_embeddings', 'rotary', 'rms_norm', 'qk_norm', 'relu_squared', 'zero_output', 'separate_embeddings',
              'residual_weights', 'logit_softcap', 'count', 'dispersion', 'bigram_table', 'attention_gate',
              'key_offset', 'dense_connections', 'backout', 'trigram_table', 'stem', 'xielu']:
        model_args[k] = checkpoint_model_args.get(k, False)
    for k in ['bigram_rows_per_token', 'document_bos']:
        model_args[k] = checkpoint_model_args.get(k, getattr(GPTConfig, k))
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)
if model.config.count:
    model.use_count_table(CountTable(count_table, device))

# The other ways of running attention have no dropout and need bf16 on a GPU; anywhere else PyTorch chooses
can_choose_attention = device_type == 'cuda' and dtype == 'bfloat16' and dropout == 0.0
print(f"attention: {choose_attention(model, attention) if can_choose_attention else 'sdpa'}")
# Only autocast lets the linear layers, whose weights stay in float32, take values in bfloat16.
if stream_bfloat16 and device_type == 'cuda' and dtype == 'bfloat16':
    model.keep_stream_in(torch.bfloat16)
print(f"stream: {'as computed' if model.stream_dtype is None else model.stream_dtype}")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
if optimizer_name == 'muon':
    optimizer = configure_muon(model, muon_lr, muon_other_lr, muon_value_embedding_lr, muon_polar_express,
                               muon_neuron_normalization, muon_cautious_decay, embedding_lr=muon_embedding_lr,
                               scalar_lr=muon_scalar_lr, cautious_adamw=muon_cautious_adamw)
else:
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
# the seconds of training that count toward the time budget, which leave out the first updates
training_seconds = 0.0
# all the names of the parameters, as the state dictionary holds them, tied ones too
average = WeightAverage(name for name, _ in raw_model.named_parameters(remove_duplicate=False))
while True:
    if time_budget > 0:
        progress = training_seconds / time_budget
        last = local_iter_num > UNCOUNTED_UPDATES and training_seconds >= time_budget
    else:
        progress = iter_num / max_iters
        last = iter_num == max_iters

    # determine and set the learning rate for this iteration
    if optimizer_name == 'muon':
        lr = muon_lr * set_schedule_at(optimizer, iter_num, progress, muon_weight_decay, muon_warmdown_share)
        if decay_dataset and lr < muon_lr and train_dir == data_dir:
            train_dir = os.path.join('data', decay_dataset)
            row_heads = row_starts(train_dir)
            print(f"iter {iter_num}: reading {decay_dataset}")
    else:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if (iter_num % eval_interval == 0 or last) and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # keep the weights to average, and once the last are in, write them out mixed with the average
    if iter_num in averaged and master_process:
        average.add(raw_model.state_dict())
        if iter_num == max_iters and average.count < len(averaged):
            print(f"not mixing: this run kept {average.count} of the {len(averaged)} weights to average")
        elif iter_num == max_iters:
            mixed_dir = os.path.join(out_dir, 'mixed')
            os.makedirs(mixed_dir, exist_ok=True)
            mixed = {'model': average.mixed(raw_model.state_dict(), average_share), 'model_args': model_args,
                     'iter_num': iter_num, 'config': config}
            print(f"saving the last weights mixed with the average of {average.count} to {mixed_dir}")
            torch.save(mixed, os.path.join(mixed_dir, 'ckpt.pt'))
            mixed = None
    if (iter_num == 0 and eval_only) or (last and time_budget > 0):
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    if time_budget > 0 and device_type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if local_iter_num >= UNCOUNTED_UPDATES:
        training_seconds += dt
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

# frame.py's evaluation of the last weights, and a summary of the run beside them
if master_process:
    summary = {'updates': iter_num, 'training_seconds': training_seconds, 'total_seconds': time.time() - started,
               'tokens': iter_num * tokens_per_iter}
    if evaluate:
        tokenizer_path = os.path.join(data_dir, 'tokenizer.json')
        with ctx:
            summary['val_bpb'] = evaluate_bpb(raw_model, os.path.join(data_dir, 'val.bin'), tokenizer_path, device)
            core = evaluate_core(raw_model, CoreTokenizer(tokenizer_path), device)
        save_core_results(core, out_dir)
        summary['core_metric'] = core['core_metric']
        summary['total_seconds'] = time.time() - started
    if device_type == 'cuda':
        summary['peak_vram_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("---")
    for key, value in summary.items():
        print(f"{key + ':':<18}{value:.17g}" if key == 'core_metric' else f"{key + ':':<18}{value}")

if ddp:
    destroy_process_group()
