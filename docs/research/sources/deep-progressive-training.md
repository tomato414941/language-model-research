# Deep Progressive Training

Zhiqi Bu (arXiv 2511.04981; the ICML 2026 title is Scaling Depth Capacity via Zero/One-Layer Model Expansion).

It trains a 0-layer or 1-layer model and expands it to the target depth all at once partway through training.

## How it works

- New layers are randomly initialized by default. From one layer, copying (w → [w, w, w]) also works. Zero initialization is not used because training does not progress. Adding the new layers below gives a smaller loss spike than inserting them above (Appendix A.3).
- For the optimizer state, carrying it over with the new layers' state at zero, copying it, or not carrying it over all give nearly the same final loss (copying is slightly unstable; Appendix C.2).
- The learning-rate schedule is WSD, and the expansion happens within the constant-learning-rate phase. The hyperparameters are the same before and after the expansion.
- The optimizer is Muon-NSGD (Muon for matrices, normalized SGD for the rest, with a single learning rate). The learning rate is 0.01 with WSD (0.05 with cosine), weight decay 0.01, no gradient clipping. It says this works better than AdamW.
- The expansion time is set by subtracting the time needed to catch up (the mixing time) from the end of the constant phase. That time is measured with two short runs: one that is deep throughout and one that expands right after the warmup (Section 7).

## Time to catch up

- A one-layer GPT expanded at 10% and at 60% of training both took about 16 billion tokens (30,000 updates) to catch up with the 12-layer model (Section 5.2, Figure 7). Within WSD's constant phase, this does not depend on when the expansion happens. Expanding after the learning rate starts to fall never catches up. With cosine, it varies a lot with the expansion time, and expanding after 50% never catches up (Section 4.2).
- This time is set by the number of tokens (with each update made 4× larger, the number of updates fell sharply; Appendix C.4), and it barely depends on model size (124 million, 400 million, 1 billion; Appendix C.1).
- In Figure 1, about 40,000 updates were subtracted from a constant phase of 528,000 updates (2% warmup, 10% decay), and the expansion came at update 480,000 (80%).

## Results

- Figure 1 (GPT-2, OpenWebText, context 1,024): the gap in final validation loss is under 0.5% at 124 million and under 0.2% at 7 billion (60 layers). Compute falls by about 80% at 7 billion.
- Each update is 512 sequences by default (about 520,000 tokens). With about 600,000 updates, that is about 310 billion tokens at 124 million, or about 2,500 tokens per parameter (our own calculation). The 0-layer model has 39 million parameters and the 1-layer model 46 million, mostly embeddings. With the paper's formula that counts by parameters, the saving at this size is about 50% (our own calculation).
- Scaling laws (Figure 2): LLaMA3 at 250 million to 2 billion with 50 tokens per parameter, and DeepSeekV3 at 200 million to 500 million active with 100 tokens per parameter. Both expand at 80%. Compute efficiency is 3–5×, and the gap widens with size.
- 150 runs (Figure 10, expanding to 12, 24 and 36 layers): expanding from 0 or 1 layers is close to the best in loss against compute. Expanding from 2 or more layers costs more compute.
- It points out that comparisons that do not count the small model's compute overstate the speedup.
- There are no downstream scores.

## Applied to the current training (our own view)

The current training reads about 670 million tokens in all (1,281 updates), about 1/24 of the roughly 16 billion tokens needed to catch up. If the catch-up time is the same at our size, an expanded model does not catch up with a model that is deep from the start before training ends. A one-layer model's compute is about 18% of the current 10 layers', so expanding at 2% saves only about 1.6% of the compute.

The abstract was read, and the HTML main text through a summary (Sections 1, 4.2, 5.2, 6 and 7, Appendices A.3, B, C.1, C.2 and C.4, and the figure captions). The numbers in the figures could not be read.
