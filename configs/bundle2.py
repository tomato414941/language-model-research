# The second bundle: the first with the parts of the model's shape that autoresearch and nanochat use.
# To turn one part off, for example: --rotary=False, --rms_norm=False
exec(open('configs/bundle.py').read())
out_dir = 'out/bundle2'
rotary = True
rms_norm = True
qk_norm = True
relu_squared = True
zero_output = True
separate_embeddings = True
# autoresearch's rates for separate tables, with its (width/768)^-0.5. For one shared table, --muon_other_lr=0.004.
muon_embedding_lr = 0.66
muon_other_lr = 0.0044
residual_weights = True
