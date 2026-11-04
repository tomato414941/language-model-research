# The second bundle, the BPE model that H-Net is compared with, on the same documents of language-model's v6
# corpora in prepare_bytes.py's tokenizer: the stable phase's mix until the learning rate begins to fall, then the
# decay phase's. calibrate.py sets the updates that the baseline's cost buys.
exec(open('configs/bundle2.py').read())
out_dir = 'out/hnet-bpe'
dataset = 'hnet-bpe-stable'
decay_dataset = 'hnet-bpe-decay'
attention = 'fa2'
stream_bfloat16 = True
