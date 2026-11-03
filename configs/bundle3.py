# The third bundle: the second with a COUNT table counted from FineWeb's text, Dispersion Loss, and its last weights
# mixed with the average of the last few, which rented.sh scores beside the last weights.
# count_data.py and count_table.py make the table. To turn one part off: --count_table=, --dispersion=0.0 or
# --average_last=0
exec(open('configs/bundle2.py').read())
out_dir = 'out/bundle3'
count_table = 'data/count'
dispersion = 0.1
# Terminal Shrinkage Averaging's values, which its authors chose before their runs: 8 weights 32 updates apart,
# the last weights among them, and the average taking 55% of the mix.
average_last = 8
average_every = 32
average_share = 0.55
