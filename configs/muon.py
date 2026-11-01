# The baseline with Muon for the matrices of the blocks and AdamW for every other weight. These
# values start from autoresearch's, which were found for 8 layers of width 512 and separate input and
# output embeddings. Its output rate is used for the embeddings here, which are shared. None of this
# has been measured on this model yet.
exec(open('configs/baseline.py').read())
out_dir = 'out/muon'
optimizer_name = 'muon'
grad_clip = 0.0
