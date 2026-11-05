# OLMoE

Muennighoff et al., OLMoE: Open Mixture-of-Experts Language Models (arXiv 2409.02060). An MoE with 1 billion active and 7 billion total parameters.

- OLMo (arXiv 2402.00838) used LayerNorm without learned weights, on the grounds that it is the safest and fastest.
- In OLMoE, replacing it with RMSNorm with learned weights made results better (Figure 14). The form without weights has more sudden gradient spikes (Figure 16), and the authors think the reason is that gradient clipping then makes the gradient differ from the true gradient. RMSNorm lowers throughput by 15%, but it was adopted for the final model.
- Whether weight decay is applied to the RMSNorm weights makes little difference.
- QK-norm slightly improved stability and performance (Figure 18). In the RMSNorm form too, it slightly lowered the training loss and prevented large gradient spikes. Throughput drops by about 10%, but it was adopted.

The text of 4.2.3 and of the QK-norm section was read. The numbers in the figures were not read. The optimizer is AdamW.
