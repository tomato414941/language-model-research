# The first bundle: Muon, its learning rates and schedule, value embeddings, heads of 128 dimensions,
# and 10 layers of width 640. The passes and updates are the ones speed.py chose for the first run, for
# the cost of the baseline; bundle.sh replaces them with what the speed of its machine buys.
# To turn one part off, for example:
#   --value_embeddings=False, --n_head=10, --n_layer=12 --n_embd=768 --n_head=12,
#   --muon_polar_express=False, --muon_neuron_normalization=False, --muon_cautious_decay=False
exec(open('configs/muon.py').read())
out_dir = 'out/bundle'
n_layer = 10
n_embd = 640
n_head = 5
value_embeddings = True
batch_size = 128
gradient_accumulation_steps = 4
max_iters = 1441
lr_decay_iters = max_iters
eval_interval = max_iters
