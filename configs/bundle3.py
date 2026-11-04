# The third bundle: the second with a COUNT table counted from FineWeb's text, Dispersion Loss, its last weights
# mixed with the average of the last few, which rented.sh scores beside the last weights, and five parts of
# modded-nanogpt's records: a table of token pairs, attention gates, the partial key offset, MUDD in the last two
# blocks and the backout.
# count_data.py and count_table.py make the table. To turn one part off: --count_table=, --dispersion=0.0,
# --average_last=0, --bigram_table=False, --attention_gate=False, --key_offset=False, --dense_connections=False
# or --backout=False
exec(open('configs/bundle2.py').read())
out_dir = 'out/bundle3'
count_table = 'data/count-fineweb'
dispersion = 0.1
# Terminal Shrinkage Averaging's values, which its authors chose before their runs: 8 weights 32 updates apart,
# the last weights among them, and the average taking 55% of the mix.
average_last = 8
average_every = 32
average_share = 0.55
bigram_table = True
attention_gate = True
key_offset = True
dense_connections = True
backout = True
