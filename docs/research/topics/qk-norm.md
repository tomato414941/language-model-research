# QK-norm

Normalizing Q and K before attention's dot product. The main purpose is training stability. [The `language-model` note](https://github.com/tomato414941/language-model/blob/main/docs/research/topics/qk-norm.md) also has a short summary.

## What is known

- **Quality:** −0.63% in training loss in [IMU-1](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/imu-1.md) (70 million, AdamW). In [Marin](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/marin-nano-arch-ablations.md), from 1.3160 to 1.3156, almost 0. [OLMoE](../sources/olmoe.md) says it got slightly better, and [QuacK](../sources/quack.md) says there is almost no difference when the learning rate is small.
- **Stability:** [the comparison of divergence](../sources/training-stability-methods.md), OLMoE and QuacK all say it helps. The 70B [Apertus](../sources/apertus.md) also includes it for stability, but its tables have no comparison of it alone. However, the first bundle trained stably to the end without QK-norm, at a learning rate of 0.04 throughout.
- **Cost:** on the H100 of [the second bundle](../experiments/second-bundle.md), each update is 4.2% slower with QK-norm (0.591 and 0.615 seconds; +0.23% when converted). In what was read, throughput falls about 10% in OLMoE, and in QuacK too it is about 10% faster without QK-norm. At the current conversion that is a loss of about 0.6%, about the same as the quality benefit. By the formula for the computation it is tiny, so with an efficient implementation it may be small. autoresearch, with QK-norm, has a GPU utilization of about 39.8%, no different from the about 39% of the first bundle without it. However, the formula does not include normalization, and differences in activation functions and normalization are also mixed in.
- **Alternatives:** QuacK keeps it in check through the learning rate, and with ordinary attention it loses slightly to QK-norm. Muon equalizes the size of updates, so that cannot be applied as is (our reading).

## Current view

The view as of 2026-11 is that the effect at the same cost is near 0 (−0.6% to +0.5%). The measured cost is smaller than the break-even point given below, so it stays, as in the second bundle.

## Conditions likely to change the view

- When changing the shape changes how much slower it is. If it is more than 5% slower, consider removing it. 5% is the break-even point when the quality benefit is taken as a middling 0.3% (about 0.3% when converted).
- When training becomes unstable. The reason to include it becomes stronger.
- When the effect is measured item by item.
