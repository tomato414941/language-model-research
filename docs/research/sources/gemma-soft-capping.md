# Gemma の logit の soft-capping

Gemma 2（arXiv 2408.00118）と Gemma 3（arXiv 2503.19786）の技術報告。

- Gemma 2 は、注意の各層と最後の層のロジットを `soft_cap * tanh(logits / soft_cap)` で抑える。値は注意が50、最後の層が30。手法の出典は Bello ら（2016）としている。効果の比較はない。
- Gemma 3 は「Gemma 2 の soft-capping を QK-norm に置き換えた」とある。理由と比較は書かれていない。

読んだのは、どちらも soft-capping と QK-norm の出てくる段落だけ。
