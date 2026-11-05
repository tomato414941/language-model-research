# Marin Belay

Issue #9451 of `marin-community/marin` (from 2026-09-25). An agent improves MoE training a little at a time and reports the results of changes tried in batches.

## Conditions

- The MoE at the smallest rung (d512) is trained for 8 minutes on 8 H100s and compared by the mean loss on Paloma. The vocabulary is 16,000 and the context 4,096.
- Differences smaller than about 0.005 are treated as noise.

## GradPower

- In the 64th batch, as one of the Parameter Golf ideas, p=0.9 was applied to the gradients of the matrices that MuonH updates, before the momentum.
- At the same 2,817 updates, it gave 3.0200 against the baseline's 3.0197 (+0.0003), which was judged to be within the noise.
- A note written before the trial says that since Newton-Schulz discards each matrix's magnitude, only the shape of the distribution of values within a row remains.

Only the opening of the issue's description and the report of the 64th batch were read. The other trials were not read.
