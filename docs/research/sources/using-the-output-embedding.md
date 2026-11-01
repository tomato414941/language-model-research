# Using the Output Embedding

Press、Wolf、Using the Output Embedding to Improve Language Models（arXiv 1608.05859、EACL 2017）。LSTM の言語モデルで、埋め込みの共有を勧めた論文の1つ。

- 共有した表の更新は、分けたモデルの入力より出力の表に近い。
- 小さいモデル（LSTM 200、ドロップアウトなし）のテストの perplexity は、PTB で 114.5 から 112.4、text8 で 195.3 から 187.1、BBC で 110.8 から 106.8、IMDB で 94.3 から 94.8。
- 学習データでは良くならない場合が多く（text8 で 90.4 から 95.6）、著者は過学習が減ったためとしている。

読んだのは要旨、導入、実験の節と表5〜7。
