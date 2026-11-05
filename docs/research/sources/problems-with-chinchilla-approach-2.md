# Problems with Chinchilla Approach 2

A post by Eric Czech (Open Athena, 2026-03, https://www.openathena.ai/blog/problems-with-chinchilla-approach-2/).

- Chinchilla's Approach 2 fits a parabola to points of the same compute to find the optimum. The actual curves are not parabolas, so the choice of points introduces a bias.
- Some of the roughly 15 published examples also have the optimum away from the middle.
- On Llama 3's data, Approach 2 and Approach 3 differed by about 6.5% in the predicted allocation of compute.
- The author recommends Approach 3 as a less biased method, but does not present it as fully correct.

Read: the whole post. The original paper was not read.
