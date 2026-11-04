# H-Net around the second bundle's main network, on research's dataset as bytes_from_tokens.py writes it. A row holds
# the text of 1,024 of the BPE model's tokens. The files beside this one name the encoder and decoder layers and then
# read it; calibrate.py sets the updates that the baseline's cost buys.
import json
import sys

exec(open('configs/bundle2.py').read())
sys.path[:0] = ['.', 'h-net']
from hnet_model import HNet, HNetConfig  # noqa: E402

out_dir = 'out/hnet'
dataset = 'hnet-bytes'
block_size = json.load(open('data/hnet-bytes/rows.json'))['block_size']
value_embeddings = False  # its tables are by token, and the main network reads chunks
attention = 'fa2'
stream_bfloat16 = True
compile = False  # the model compiles its main network's blocks, around the attention kernel, itself
# The routing begins a chunk at about every other byte early on, which half as many rows per pass keeps in memory.
batch_size = 64
gradient_accumulation_steps = 8

outer_width = 512
encoder_layers = 4
decoder_layers = 4
target_ratio = 6.0
ratio_weight = 0.03
fixed_chunks = 0
HNET_FIELDS = ('outer', 'outer_width', 'encoder_layers', 'decoder_layers', 'target_ratio', 'ratio_weight',
               'fixed_chunks')
GPT = HNet


def GPTConfig(**arguments):
    # Read when train.py builds the model, after the command line has set its values, and added to the model_args
    # that train.py saves with the weights, so that the saved model can be built again.
    fields = {name: globals()[name] for name in HNET_FIELDS}
    model_args.update(fields)
    return HNetConfig(**{**arguments, **fields})
