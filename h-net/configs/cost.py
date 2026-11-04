# The baseline, whose time on the machine is the cost that every model of the H-Net comparison gets. It reads the
# BPE model's data, of the same vocabulary as ClimbMix's tokenizer, so its speed is the baseline's without ClimbMix.
exec(open('configs/baseline.py').read())
out_dir = 'out/hnet-cost'
dataset = 'hnet-bpe-stable'
