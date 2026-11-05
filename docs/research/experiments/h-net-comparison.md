# Comparing H-Net with the second bundle at the same cost

Decided on 2026-11-04, before measuring. The comparison asks whether H-Net does better than a BPE model at the same cost. Three versions are measured, with Mamba-2 (the paper's form), Mamba-3 or Gated DeltaNet as the outer layers. The code and settings are in `h-net/`.

## Conditions

- Comparison: the second bundle (as in second-bundle.md), retrained on research's current training data (FineWeb from 2026-11-04, training-data.md). The H-Net side does not train it.
- Data: the documents of the same train.bin are turned into bytes in the same order and read (bytes_from_tokens.py). As in the second bundle, rows start at the head of a document and are chosen from the same set of documents. A row holds the number of bytes that 1,024 BPE tokens hold on average, which on FineWeb is 3,840 bytes (3.81 bytes per token on average in val.bin). The data is English only, so this comparison cannot measure the effect on Japanese.
- Machine: one H100 SXM 80GB on Vast.ai. The image is the same as the baseline's.
- Cost: the baseline (baseline.md) is run briefly on the same machine to find the time that 954 updates take. Each model gets as many updates as it can do in that time (the same calculation as speed.py). H-Net is timed with its chunk boundaries fixed every 6 bytes. Early in training it places a chunk boundary about every 2 bytes, so a short measurement would make it look slower than the real run. The real run's training time is recorded, and noted if it differs from the time decided.
- H-Net: the main network is the same 10 layers of width 640 as the second bundle and runs on the chunks. Value embeddings were removed, since they are a per-token table. The outer layers have width 512, with 4 layers each in the encoder and the decoder. The target is a chunk boundary every 6 bytes, and the ratio loss weight is 0.03 (as in the paper). 512 rows per update (16 passes of 32 rows). The outer layers' learning rate is untuned: 0.0044, the same as AdamW's other weights. Mamba-2 runs on the path that computes the convolution in PyTorch, because causal-conv1d, which the fast path uses, has no prebuilt version for this PyTorch. This path keeps many intermediate values, so the rows read per pass were reduced.
- Scoring: the same measures as the second bundle, on the same text. Validation BPB reads the bytes of the same rows as scoring.py (the first 2²⁰ tokens of val.bin, 1,024 tokens per row) and divides by the number of bytes scoring.py counts. Accuracy and correct-answer BPB on harness.py's 6 English and 3 Japanese tasks. The Japanese tasks are measured the same way but not used in reading the results.

## How to read the results, decided before seeing them

- Validation BPB and the mean correct-answer BPB over the 6 English tasks are compared with the second bundle's. The noise from the seed is taken as 0.3% for validation BPB (scoring.md) and 1.3% for the mean correct-answer BPB (second-bundle.md). Lower by more than that is a win for H-Net, higher by more than that is a loss, and in between the difference cannot be measured.
- The prediction before measuring: all three lose. For the outer layers, Gated DeltaNet and Mamba-3 do better than Mamba-2.

## Result: Mamba-2 (2026-11-04)

One run on a Vast.ai H100 SXM 80GB (Japan, $2.41/hour). By the reading decided beforehand, H-Net (Mamba-2) lost. Its validation BPB was 2.431, 2.0× the second bundle's 1.2146 on FineWeb (third-bundle.md, the same rows as scoring.py). This is far beyond the 0.3% taken as noise. Correct-answer BPB was not measured. Since validation BPB differed by 2×, measuring it was judged not to change the reading of a loss (unverified).

| Item | Value |
|---|---|
| Baseline: one update, startup | 0.830 s, 37 s |
| H-Net: one update, startup (measured with chunk boundaries fixed every 6 bytes) | 3.93 s, 129 s |
| Updates run at the same cost | 178 (the baseline 954; the second bundle 1,275 on FineWeb, in a separate rental, third-bundle.md) |
| Training time (from startup to saving) | About 13 minutes. The cost decided was 829 s, and it came close to that |
| Spacing of chunk boundaries | Fell short of the 6-byte target, ending at about 2.7 bytes. Partway, it briefly swung up to 63 bytes |
| Final training loss (nats per byte, including the ratio loss) | Training 1.7273, validation 1.7245 |
| Validation BPB (the same rows as scoring.py, 4.01 million bytes) | 2.431 |
| Correct-answer BPB and accuracy (harness.py) | Not measured. lm-eval was not installed on this machine |

- One update reads about 1.97 million bytes for H-Net (512 rows × 3,840 bytes) and about 2 million bytes for the second bundle (524,288 tokens × 3.81 bytes), nearly the same. At the same cost, H-Net read about one seventh of the amount the second bundle read.
- This was measured under the rules from before research's frame changed to CORE and 30 minutes of training (the baseline's cost, BPB). It has not been measured under the new frame.
- One H-Net update was 4.7× slower than the baseline's, so at the same cost it got less than a fifth of the baseline's updates. Where the slowness comes from was not measured. We guess that the causes are running Mamba-2 on the unfused path, reading only 32 rows at a time over 16 passes, and calling the main network layer by layer from Python.
- The chunk boundaries were finer than the target, so the main network read about 2.2× as many chunks as in the timing measurement. One update during training therefore took about 4.4 s, slower than the measured 3.93 s.
- The machine was held about 31 minutes from renting to deleting, about $1.25. Two failed setups (mamba-ssm's setup.py stopped without nvcc; pyarrow was missing) lost about 10 minutes.
- Validation BPB was measured with score_hnet.py's validation_bits (the same calculation as `--validation-only`). The weights are in out/h-net-rental/models/hnet-mamba2.


## Where the time goes, and what was fixed (2026-11-05)

In the measurement above, one update took 4.7× the baseline's, so the causes of the slowness were investigated and fixed, and one Vast.ai H100 SXM 80GB (Japan, $2.14/hour) was rented to measure before and after the fixes on the same machine. The data was random bytes, chunk boundaries were fixed every 6 bytes, and one update was 512 rows. The code before the fixes was that of the previous commit (2132b32), sent to the same machine and measured. The rental lasted about 30 minutes, about $1.1.

| Form | One update | Peak memory |
|---|---|---|
| Before the fixes (Mamba-2 on the unfused path, 32 rows × 16 passes) | 3.917 s | 25.7 GB |
| Fixing only the GPU waits and compilation (same path and passes) | 3.647 s | 24.6 GB |
| Also Mamba-2 on the fused path (32 rows × 16 passes) | 2.651 s | 19.5 GB |
| Same (64 rows × 8 passes) | 2.616 s | 38.0 GB |
| Same (128 rows × 4 passes) | 2.583 s | 75.0 GB |
| Same, with a chunk boundary every 2 bytes (64 rows × 8 passes) | 3.412 s | 58.0 GB |
| The second bundle (random tokens, 128 rows × 4 passes) | 0.616 s | 45.7 GB |

- Fused path: mamba-ssm's fused path computes the convolution with causal-conv1d's CUDA function, which has no build for this PyTorch. A convolution that runs with the same arguments was written in Triton and slotted in (h-net/causal_conv.py). In Triton's interpreter on the CPU it matched PyTorch's convolution down to the gradients, and on the H100 its difference from the unfused path was at most 2.0% (dt_bias's gradient, bf16 rounding). Updates became 27% faster.
- GPU waits: every 32 rows, the code waited on the GPU 3 times (for the chunk positions, the length of the longest row and the share of chunk boundaries). The last two were removed, leaving a single wait, to learn the number of chunks. In the profile, the share of time the GPU was busy went from 92.3% (before, 32 rows) to 97.1% (after, 128 rows). The idea of capping the number of chunks to make the sizes uniform was not taken: trimming the chunk boundaries of rows over the cap would make which bytes become chunk boundaries depend on later bytes (breaking causality), and the computation on the padding up to the cap would also be wasted.
- Compilation: the main network, which had been split into 20 graphs by layer, became one, and the normalizations and additions before and after the outer layers, the chunk boundary probabilities, and the prediction and loss were compiled too. Together with the change to the waits, this made it 7% faster. The first update (startup), compared cold against cold, was unchanged at 84 s and 85 s. Under the new frame, startup does not count toward the 30 minutes.
- Rows read at a time: even 128 rows was only 2.6% faster than 32. A training try on FineWeb with 64 rows × 8 passes ran out of memory at the 9th update, while the chunk boundaries were still fine (67.5 GB allocated, 9.9 GB reserved but unused). It was left at 32 rows.
- The finer the chunk boundaries, the slower: a boundary every 2 bytes was 30% slower than every 6 bytes (64 rows). The 11-04 training ended at about 2.7 bytes.

In the profile after the fixes (128 rows, 2 updates), causal_conv.py's convolution backward took 28.5% of the GPU time, Mamba-2's SSD 29.0%, matrix multiplications 15.4%, the compiled parts 9.2%, Mamba-2's normalization 6.9% and attention 2.3%. Before the fixes, the unfused path's copies of values took 28.8% and cuDNN's depthwise convolution 15.9%.

- The convolution backward took 22.8 ms per call, slower than the about 1.5 ms estimated from the amount read and written. With ptxas for sm_90, the tile of 64 positions × 128 channels spilled about 24 KB per thread to memory. Tiles were made 16–64 positions, with one of 5 variants chosen on the GPU, and the spill became 0–52 bytes (98b78c1). The speed has not been measured yet. If it runs at the bandwidth limit, one update is estimated at about 1.9–2.0 s, about 3× the second bundle's.
- Even then, much of the time goes to the outer Mamba-2, so the outer width was cut from 512 to 256. At width 256, the outer layers' matrix multiplications become about a quarter, and the SSD, convolution and normalization about half.
- While the chunk boundaries are finer than the target, the ratio loss weight is now raised (every 512 rows, it moves on a log scale by 0.1 × (target ratio × share of chunk boundaries − 1), with 0.03 as the floor and 1 as the ceiling). The paper keeps it at 0.03. The reason: in the 11-04 training the chunk boundaries started getting finer from the 68th update, and at the very start of training they were if anything coarse (63 bytes at the 25th update), so fixing the chunk boundaries only at the start would not reach the problem.
- On the CPU, a small H-Net (outer layers convolutions of width 64, main network 2 layers of width 128, 16 rows of 512 bytes per update) was trained for 400 updates on text from FineWeb's training data (the weight's movement per update was made the same as in the real run). With the weight fixed at 0.03, the chunk boundaries grew finer, from 4.5 bytes at update 60 to 3.25 bytes at update 400. With a rate of 0.03 they stopped at 4.7–5.3 bytes while the weight kept rising; with 0.1 they held at 5.4–6.2 bytes from update 80, and the weight reached the ceiling of 1 at update 360. There was no oscillation. Cross-entropy was about 3% higher at update 400, the cost of the coarser chunk boundaries. The behavior at the real size has not been seen yet.
- The loss in the training log includes weight × ratio loss (1 at the target ratio). Once the weight moves, the loss can no longer be compared, so cross-entropy is now reported separately.

## One run under the frame (2026-11-05)

Following the frame in README.md, H-Net was trained for 30 minutes on a Vast.ai H100 SXM 80GB (Japan, $2.41/hour) and measured once on the three OLMES suites and validation BPB. The code is 235f8ed (h-net/run.sh; evaluation in h-net/evaluate_hnet.py). The rental lasted 58 minutes, about $2.33.

- Form: the main network is the same 10 layers of width 640 as the second bundle. The outer layers are Mamba-2 of width 256, with 4 layers each in the encoder and the decoder. The target is a chunk boundary every 6 bytes, and while the chunk boundaries are finer, the ratio loss weight is raised (rate 0.1, from 0.03 up to 1). One update is 512 rows (32 rows × 16 passes), at 3,840 bytes per row.
- Data: FineWeb alone, turned into bytes and read (bytes_from_tokens.py). FineMath and code were not read.
- Evaluation: the same requests and the same olmes as frame.py's evaluation were connected to H-Net and a byte tokenizer. The length limit for rows passed to olmes was set to 16,384 bytes, because the longest answer is 5,314 bytes and frame.py's 4,096 would cut answers off. Rows longer than the context (3,840 bytes) are read in windows by frame.py's OlmesModel. Validation BPB read the same rows as scoring.py and divided by the number of bytes scoring.py counts.

| Item | H-Net | research-v4 (e77674d, runs.md) |
|---|---|---|
| QA BPB | 1.9240 | 1.5155 |
| Math BPB | 2.8731 | 0.9386 |
| Code BPB | 2.8107 | 1.1122 |
| Validation BPB | 1.3558 | 1.1255 |
| Updates in 30 minutes | 1,138 (median 1.55 s) | 3,375 |
| Amount read | 2.24 billion bytes | 885 million tokens |
| Evaluation time | 17.4 min | 20 min |

- By the frame's reading, a loss. The mean improvement rate over the three suites was −129% (QA −27%, Math −206%, Code −153%), and validation BPB was also 20% higher. This matches the prediction made before measuring (a loss).
- We think most of the gap in Math and Code comes from the training data. Comparing H-Net's BPB with research-v4's task by task, it was 1.0–1.6× on most QA tasks (median 1.27× over MMLU's 58 subjects, 2.05× on the two tasks answered yes or no), 2.6–4.3× on math (Minerva's 7 tasks) and 1.8–4.1× on code (MBPP's 18 languages; median 3.1×). There is no measurement comparing it with a BPE model on the same data.
- It is 12% higher than the validation BPB of 1.2146 of the second bundle (11-04, FineWeb, 1,275 updates at the same cost as the baseline). It fell 44% from the 11-04 H-Net (2.431).
- The chunk boundaries swung widely over the first 100 or so updates, between 1 and 150 bytes, and after that were 5.1–5.7 bytes (median 5.5). The ratio loss weight reached the ceiling of 1 around update 190 and stayed at 1 to the end. We think the ceiling is why it fell short of 6 bytes. Cross-entropy at the end of training was about 0.9 nats per byte.
- One update took 1.55 s (median), faster than the width-512 measurement above (random bytes, 2.651 s). Per amount read, it is about 1.5× slower than research-v4 (0.53 s per update, about 1 million bytes). The first update (compilation) took 166 s, and peak memory was 38.6 GB.
- The evaluation read 650,000 requests in 3 processes. The first 7 tasks, including the 4 large ones (SQuAD, DROP, CoQA, HellaSwag), took about 10 minutes, and the remaining 102 tasks about 8 minutes.
- Breakdown of the time: 7 minutes from renting to the start of training (3 minutes to send the data, 2 minutes to turn it into bytes), 33 minutes of training (including startup and compilation), 17.5 minutes of evaluation (including starting the processes).
- The rented machine's Python had no ensurepip, so run.sh could not create the olmes environment (venv). During training it was recreated as a venv without pip, and the evaluation then went ahead after training with the same code. run.sh has been fixed.
- The weights are in out/h-net-frame-2026-11-05/model.
