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
# The model compiles its parts itself (compile_main), since train.py's compile cannot run through Mamba-2's kernels.
compile = False
# The routing begins a chunk at about every other byte early on, which passes of 32 rows leave room for.
batch_size = 32
gradient_accumulation_steps = 16

outer_width = 512
encoder_layers = 4
decoder_layers = 4
target_ratio = 6.0
ratio_weight = 0.03
ratio_weight_rate = 0.03
compile_main = True
fixed_chunks = 0
HNET_FIELDS = ('outer', 'outer_width', 'encoder_layers', 'decoder_layers', 'target_ratio', 'ratio_weight',
               'ratio_weight_rate', 'compile_main', 'fixed_chunks')
GPT = HNet


def GPTConfig(**arguments):
    # Read when train.py builds the model, after the command line has set its values, and added to the model_args
    # that train.py saves with the weights, so that the saved model can be built again.
    fields = {name: globals()[name] for name in HNET_FIELDS}
    model_args.update(fields)
    return HNetConfig(**{**arguments, **fields})
