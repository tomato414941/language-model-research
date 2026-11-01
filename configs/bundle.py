# The first bundle: Muon, its learning rates and schedule, value embeddings, heads of 128 dimensions,
# and 10 layers of width 640. The updates assume the compute of the baseline and should be set from
# the measured speed before a full run. To turn one part off, for example:
#   --value_embeddings=False, --n_head=10, --n_layer=12 --n_embd=768 --n_head=12,
#   --muon_polar_express=False, --muon_neuron_normalization=False, --muon_cautious_decay=False
exec(open('configs/muon.py').read())
out_dir = 'out/bundle'
n_layer = 10
n_embd = 640
n_head = 5
value_embeddings = True
max_iters = 1602
lr_decay_iters = 1602
eval_interval = 1602
