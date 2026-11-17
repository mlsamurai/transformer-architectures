# Transformer architectures from BERT to diffusion

A free course in fifteen lessons on the models that grew out of the Transformer. Each lesson writes the model's key
part in PyTorch, loads the official weights into that code and checks it against the reference implementation, then
puts the model to work on a task it does well, with a different dataset every time.

The lessons go from encoders that read text to models that see, listen and draw, and each one starts from a limit of
an earlier model. Every lesson is a notebook that runs on a free GPU on Kaggle or Colab, and a post on
[blog.korolevalab.com](https://blog.korolevalab.com/teaching/transformer-architectures/) with all of its outputs and figures. It helps to
have taken [the first course](https://github.com/mlsamurai/transformers-from-scratch) or to be able to write attention
and a training loop yourself.

## Lessons

### Part 1. Encoders that understand text

| | Lesson | Notebook |
|---|---|---|
| 1 | [BERT from scratch with masked language modelling and fine-tuning](https://blog.korolevalab.com/blog/2026/transformer-architectures-01-bert/)<br>BERT written in PyTorch and loaded with Google's weights. Why reading in both directions needed a mask, what BERT knows, and a detector of personal data in long financial documents, set against regular expressions. | [Kaggle](https://www.kaggle.com/code/qmarva/transformer-architectures-01-bert) · [Colab](https://colab.research.google.com/github/mlsamurai/transformer-architectures/blob/main/notebooks/colab/a01_bert.ipynb) |
| 2 | [How BERT changed from RoBERTa to ModernBERT](https://blog.korolevalab.com/blog/2026/transformer-architectures-02-modernbert/)<br>What seven years of research changed in BERT, from RoBERTa's training recipe to ModernBERT's block. Each model is checked against its real weights, and all four sort GitHub issues from projects they never saw. | [Kaggle](https://www.kaggle.com/code/qmarva/transformer-architectures-02-roberta-modernbert) · [Colab](https://colab.research.google.com/github/mlsamurai/transformer-architectures/blob/main/notebooks/colab/a02_modernbert.ipynb) |
| 3 | [Semantic search with Sentence-BERT, cross-encoders and ColBERT](https://blog.korolevalab.com/blog/2026/transformer-architectures-03-search/)<br>BM25 from scratch, a Sentence-BERT bi-encoder trained on in-batch negatives, a cross-encoder reranker and ColBERT. Together they become a search engine over 58,000 machine-learning papers. | [Kaggle](https://www.kaggle.com/code/qmarva/transformer-architectures-03-embeddings-search) · [Colab](https://colab.research.google.com/github/mlsamurai/transformer-architectures/blob/main/notebooks/colab/a03_search.ipynb) |

### Part 2. Decoders and the encoder–decoder

| | Lesson | Notebook |
|---|---|---|
| 4 | [GPT-2 from scratch and a small GPT that learns chess](https://blog.korolevalab.com/blog/2026/transformer-architectures-04-gpt/)<br>GPT-2 written from scratch with OpenAI's weights, the KV cache and the logit lens. Then a small GPT trained on Lichess games learns legal chess, and a linear probe finds the board inside it. | [Kaggle](https://www.kaggle.com/code/qmarva/transformer-architectures-04-gpt-chess) · [Colab](https://colab.research.google.com/github/mlsamurai/transformer-architectures/blob/main/notebooks/colab/a04_gpt.ipynb) |
| 5 | [T5 and BART from scratch and a T5 that writes SQL](https://blog.korolevalab.com/blog/2026/transformer-architectures-05-t5/)<br>T5 written from scratch with Google's weights, with its relative positions, cross-attention and span corruption, next to BART's text infilling. A T5 fine-tuned on Spider writes SQL, and every query is run on SQLite to check it. | [Kaggle](https://www.kaggle.com/code/qmarva/transformer-architectures-05-t5-bart) · [Colab](https://colab.research.google.com/github/mlsamurai/transformer-architectures/blob/main/notebooks/colab/a05_t5.ipynb) |
| 6 | [Modern LLM architecture and Qwen3 from scratch](https://blog.korolevalab.com/blog/2026/transformer-architectures-06-modern/)<br>What Mistral, Gemma and Qwen changed after Llama, read from their configs. Qwen3-0.6B written from scratch with its official weights, and LoRA that turns voice commands into JSON in English and Russian. | [Kaggle](https://www.kaggle.com/code/qmarva/transformer-architectures-06-modern-llms) · [Colab](https://colab.research.google.com/github/mlsamurai/transformer-architectures/blob/main/notebooks/colab/a06_modern.ipynb) |

### Part 3. Scale and efficiency

| | Lesson | Notebook |
|---|---|---|
| 7 | [Mixture-of-experts LLMs from scratch with OLMoE](https://blog.korolevalab.com/blog/2026/transformer-architectures-07-moe/)<br>How mixtures of experts from Mixtral to gpt-oss are built. OLMoE with its official weights shows which experts English, Python, maths and Russian use, and then gets pruned for one kind of text. | [Kaggle](https://www.kaggle.com/code/qmarva/transformer-architectures-07-mixture-of-experts) · [Colab](https://colab.research.google.com/github/mlsamurai/transformer-architectures/blob/main/notebooks/colab/a07_moe.ipynb) |
| 8 | [The KV cache budget and DeepSeek's latent attention](https://blog.korolevalab.com/blog/2026/transformer-architectures-08-kv/)<br>What one token costs in the KV cache of ten models, and how DeepSeek's latent attention cuts it. MLA in both of its forms, MiniCPM3-4B with its official weights, and Qwen3 compressed without training. | [Kaggle](https://www.kaggle.com/code/qmarva/transformer-architectures-08-kv-cache-and-mla) · [Colab](https://colab.research.google.com/github/mlsamurai/transformer-architectures/blob/main/notebooks/colab/a08_kv.ipynb) |
| 9 | [Long context with RoPE scaling, YaRN and FlashAttention](https://blog.korolevalab.com/blog/2026/transformer-architectures-09-long/)<br>Why a language model breaks past the length it was trained on, and how RoPE scaling fixes it. YaRN and Llama 3.1's scaling compared on TinyLlama, FlashAttention built from the online softmax, and Qwen3 searching a novel for a needle. | [Kaggle](https://www.kaggle.com/code/qmarva/transformer-architectures-09-long-context) · [Colab](https://colab.research.google.com/github/mlsamurai/transformer-architectures/blob/main/notebooks/colab/a09_long.ipynb) |
| 10 | [Mamba and state space models instead of attention](https://blog.korolevalab.com/blog/2026/transformer-architectures-10-mamba/)<br>State space models from the continuous equation to Mamba's selective scan, checked against mamba-130m. Mamba against Pythia on the same data, Mamba-2 as masked attention, and the layouts of hybrid models. | [Kaggle](https://www.kaggle.com/code/qmarva/transformer-architectures-10-mamba-and-hybrids) · [Colab](https://colab.research.google.com/github/mlsamurai/transformer-architectures/blob/main/notebooks/colab/a10_mamba.ipynb) |

### Part 4. Beyond text

| | Lesson | Notebook |
|---|---|---|
| 11 | [The Vision Transformer reads images as sequences of patches](https://blog.korolevalab.com/blog/2026/transformer-architectures-11-vit/)<br>Every step from a photo to the 197 vectors the encoder reads, checked against Google's ViT-B/16. Where its heads look, how DINO finds objects without labels, and ViT against a CNN on satellite images with few labels. | [Kaggle](https://www.kaggle.com/code/qmarva/transformer-architectures-11-vision-transformer) · [Colab](https://colab.research.google.com/github/mlsamurai/transformer-architectures/blob/main/notebooks/colab/a11_vit.ipynb) |
| 12 | [CLIP and SigLIP put images and text in one space](https://blog.korolevalab.com/blog/2026/transformer-architectures-12-clip/)<br>Both towers of CLIP written from scratch with OpenAI's weights, and SigLIP's sigmoid loss next to the contrastive one. Zero-shot geolocation over 211 countries, the modality gap, and a word on a photo that changes the answer. | [Kaggle](https://www.kaggle.com/code/qmarva/transformer-architectures-12-clip-and-siglip) · [Colab](https://colab.research.google.com/github/mlsamurai/transformer-architectures/blob/main/notebooks/colab/a12_clip.ipynb) |
| 13 | [How vision-language models read images, from LLaVA to Qwen2-VL](https://blog.korolevalab.com/blog/2026/transformer-architectures-13-vlm/)<br>LLaVA's tiles, projector and splice written from scratch and matched to LLaVA-OneVision. What an image costs in tokens, Qwen2-VL's dynamic resolution, and receipts and charts read under shrinking token budgets. | [Kaggle](https://www.kaggle.com/code/qmarva/transformer-architectures-13-vision-language) · [Colab](https://colab.research.google.com/github/mlsamurai/transformer-architectures/blob/main/notebooks/colab/a13_vlm.ipynb) |
| 14 | [Whisper turns speech into text with an encoder–decoder Transformer](https://blog.korolevalab.com/blog/2026/transformer-architectures-14-whisper/)<br>Whisper's log-mel front end, encoder and decoder written from scratch and checked against whisper-tiny. Word error rates in English and Russian, subtitle credits written over silence, and word timings from cross-attention. | [Kaggle](https://www.kaggle.com/code/qmarva/transformer-architectures-14-whisper) · [Colab](https://colab.research.google.com/github/mlsamurai/transformer-architectures/blob/main/notebooks/colab/a14_whisper.ipynb) |
| 15 | Diffusion Transformers from DiT to Stable Diffusion 3<br>Denoising diffusion and flow matching, the adaLN-Zero block, a small DiT trained on butterflies, and the MM-DiT of Stable Diffusion 3. | coming |

## What is here

`notebooks/` holds every lesson twice, for Colab and for Kaggle, and `assets/` the figures, covers and photos they
show. The notebooks install their small helper package, `tfs`, from
[the first course's repository](https://github.com/mlsamurai/transformers-from-scratch/tree/main/src/tfs).

## Citing

[CITATION.cff](CITATION.cff) has the reference; GitHub shows it under "Cite this repository".

## License

Code under the [MIT License](LICENSE).
