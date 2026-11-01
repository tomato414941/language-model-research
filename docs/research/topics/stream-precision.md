# 層を流れる値の精度

各層を流れる値（活性）を何ビットで持つか。モデルの形も重みも変わらず、速さの話。重みの精度については `language-model` に[メモ](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/training-precision.md)がある。

今のコードは autocast の下で、行列の積と注意を bf16 で計算し、層を流れる値は fp32 のまま。[PyTorch のコード](../sources/pytorch-autocast.md)では GPU の autocast で `rms_norm` が fp32 を返すので、流れる値を bf16 にしても、RMSNorm のたびに fp32 に戻るはず（GPU では確かめていない）。値が大きいほどメモリの読み書きが増える。

## 分かっていること

- **[modded-nanogpt の記録10](../sources/modded-nanogpt-records.md)：** 12層、幅768、語彙50,304で、活性を bf16 にして autocast をやめた。1回の更新が6.5%短くなり、3.28 に届く更新は1.3%増え（val 3.2791 と 3.2781 で揃ってはいない）、全体で5.3%短い。換算で約 −0.31%。
- **[nanochat](../sources/nanochat-log.md)（03-04）：** autocast をやめて型を明示した。速さの数字はない。
- **GPU の使用率：** nanochat の LOG は約46%で、autocast のままの autoresearch と最初の束は約39%。計算の式が同じでも形は違い、原因は調べていない。この差の一部かもしれない（自分たちの読み）。
- **品質：** 丸めが増える。nanochat は「たぶん問題ない」としている。記録10の更新の増えは、丸めのせいか種のばらつきかは分からない。
- **勾配：** 同じ重みで fp32 と比べると、幅640、4層で行列の勾配のコサイン類似度は0.9999以上、x0 の重みは0.98。幅32の小さいモデルでは x0 と resid の重みの勾配が丸めに埋もれる（0.82、0.56。[確認の記録](../experiments/second-bundle-code-check.md)）。CPU の autocast での比較。
- **確かめやすさ：** 速さは `speed.py` の短い計測で分かり、出力と勾配が合うかも `attention_check.py` と同じ方法で確かめられる。読むだけで決めなくてよい。

## 今の見方

2026-11時点の見方では、2番目の束に入れる。流れる値を bf16 に揃え、RMSNorm は入力と同じ型で返す。重みと保存の形は変えず、autocast も残す。束を借りるときに `speed.py` でありなしを測り、速くて出力が合うときだけ使う。同じコストでの効果は0〜0.4%と見ている。

autocast をやめて型を明示する形（記録10、nanochat）は、変更が大きいので後に回す。小さい形で速くならなければ、次に試す。

## 見方が変わりそうな条件

- 測って速くならない場合。今の H100、compile、QK-norm の組み合わせで効くかは分からない。
- 学習が不安定になる場合。丸めを疑う。x0 と resid の重みの勾配が一番影響を受けやすい。
- autocast をやめた形を測り、もっと速い場合。
