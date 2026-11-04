# H-Net with Mamba-2 encoder and decoder layers, as the paper has them.
outer = 'mamba2'
exec(open('h-net/configs/hnet.py').read())
out_dir = 'out/hnet-mamba2'
