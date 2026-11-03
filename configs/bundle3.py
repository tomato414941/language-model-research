# The third bundle: the second with a COUNT table counted from FineWeb's text, and Dispersion Loss.
# count_data.py and count_table.py make the table. To turn one part off: --count_table= or --dispersion=0.0
exec(open('configs/bundle2.py').read())
out_dir = 'out/bundle3'
count_table = 'data/count'
dispersion = 0.1
