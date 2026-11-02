# The second bundle's shape on the English of language-model's v6 corpora, rewritten in this project's
# tokenizer by prepare_packed.py: the mix of its stable phase until the learning rate begins to fall, and
# then the mix of its decay phase, which is where that recipe changes mix. The ways of running and the
# updates are the ones speed.py chose for the second bundle, so that it reads as many tokens at the same cost.
exec(open('configs/bundle2.py').read())
out_dir = 'out/bundle2-v6'
dataset = 'v6-en-stable'
decay_dataset = 'v6-en-decay'
attention = 'fa2'
stream_bfloat16 = True
max_iters = 1281
lr_decay_iters = max_iters
eval_interval = max_iters
