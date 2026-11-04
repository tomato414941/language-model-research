# H-Net with Gated DeltaNet encoder and decoder layers in place of the paper's Mamba-2.
outer = 'gated_deltanet'
exec(open('h-net/configs/hnet.py').read())
out_dir = 'out/hnet-gated-deltanet'
