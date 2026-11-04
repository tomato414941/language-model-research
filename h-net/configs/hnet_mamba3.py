# H-Net with Mamba-3 encoder and decoder layers in place of the paper's Mamba-2.
outer = 'mamba3'
exec(open('h-net/configs/hnet.py').read())
out_dir = 'out/hnet-mamba3'
