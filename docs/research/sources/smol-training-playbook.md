# The Smol Training Playbook

Hugging Face's record of training SmolLM3 (huggingfacetb-smol-training-playbook.hf.space).

The section on embeddings:

- The baseline is shared. Separating them gives 2 × vocabulary × width parameters. The share of the separate tables is 13% in Llama 3.2 8B and 3% in Llama 3.1 70B.
- Three are compared: a shared 1.2 billion (16 layers), a separate 1.46 billion (16 layers) and a separate 1.2 billion (12 layers).
- The shared 1.2 billion, with about 18% fewer parameters, is about the same as the separate 1.46 billion on the benchmarks other than WinoGrande. The separate 1.2 billion is worse than both.
- The authors say that at the same parameter count, going deeper works better than separating, and made SmolLM3-3B shared.
- It cites MobileLLM's result at 125 million: sharing cut the parameters by 11.8% with only a slight drop in accuracy.

The numbers come only from the descriptions in the text; the values in the tables were not read. Read: the part on embeddings extracted with the fetching tool; the rest was not read. `language-model`'s [note](https://github.com/tomato414941/language-model/blob/main/docs/research/sources/smol-training-playbook.md) was written from a separate reading note.
