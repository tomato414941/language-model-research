# Improving our LLM Pretraining Efficiency

Larry Dial's post (Open Athena, 2026-06-03, https://www.openathena.ai/blog/pretraining-speedup/). Marin's move from dense models to MoE, and the improvements after it.

The multipliers are in compute, with the wall-clock values in parentheses.

| Change | Multiplier |
|---|---|
| Dense to Marin MoE V1 (1e23) | 6.7 (3.6) |
| Experts from 64 to 256 | 1.4 (1.3) |
| AdamH to MuonH | 1.3 (1.25) |
| Partial Key Offset | 1.2 (1.2) |
| Expert normalization and scale | 1.04 (1.04) |

- The gap from dense to MoE is about 4× in compute at 1e18–3e20. The last four were measured one at a time; stacked, they gave 2.1× at 3e19.
- The MoE law is loss = 1.6 + 95.18·C^−0.0941, and the dense law is loss = 1.65 + 117.8·C^−0.0966.
- Adam's learning rate fit lr = 1.680·tokens^−0.282·dim^−0.372·bs^0.5.
- It says half of Partial Key Offset's effect comes from the form that applies RoPE to only half of the heads.
- Attention Residuals, per-layer embeddings, MUDD, mHC and Engram also look able to speed things up by 15% or more, but they are passed over to keep complexity down.

The summary, the MoE section, and the openings of the MuonH and Partial Key Offset sections were read. The values in the figures were not read.
