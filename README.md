# **"Polarity-Aware Probing for Quantifying Latent Alignment in Language Models."**

Source code for the paper ["Polarity-Aware Probing for Quantifying Latent Alignment in Language Models"]().

![img](https://raw.githubusercontent.com/SadSabrina/ccs_2/refs/heads/main/data/teaser/ccs_teaser.png)

### **Abstract**

Recent progress in unsupervised probing methods, notably <span style="color:blue">[Contrast‑Consistent Search (CCS)](https://arxiv.org/pdf/2309.06991)</span>, has enabled the extraction of latent beliefs in language models without relying on token-level outputs. As these probes offer lightweight diagnostic tools with low alignment tax, a central question arises: *can they effectively assess model alignment?* We investigate this by examining CCS's sensitivity to harmful vs. safe statements and introducing Polarity‑Aware CCS (PA‑CCS), which evaluates whether a model's internal representations remain consistent under polarity inversion. We propose two alignment-oriented metrics — **Polar‑Consistency** and **Contradiction Index** — to quantify the semantic robustness of a model's latent knowledge. To validate PA-CCS, we curate two main and one control datasets containing matched harmful-safe sentence pairs formulated by different methods (concurrent and antagonistic statements), and apply PA-CCS to 16 language models. Our results demonstrate that PA‑CCS reveals both architectural and layer-specific differences in the encoding of latent harmful knowledge. Interestingly, replacing the negation token with a meaningless marker degrades the PA‑CCS scores of models with aligned representations. In contrast, models lacking robust internal calibration do not show this degradation. Our findings highlight the potential of unsupervised probing for alignment evaluation and call on the community to incorporate structural robustness checks into interpretability benchmarks.

# `code/` — Source Code for PA-CCS Evaluation

This directory contains all the necessary scripts for reproducing the experiments described in our paper on **Polarity-Aware Probing for Quantifying Latent Alignment in Language Models** for analyzing internal representations in language models.

The code is organized to cleanly separate:

*  Feature extraction (`extract.py`, `extract_llama.py`)
*  Probing & analysis (`ccs.py`)
*  Evaluation & reporting (`format_results.py`)

## Files

### `ccs.py`

Base implementation of the **Contrast Consistent Search (CCS)** method for linear probing of model representations.

* Implements the linear probing training and evaluation procedure.
* Used to compute CCS directions and evaluate the empirical separation accuracy (ESA) and introduced metrics: polar consistency (PC), and contradiction index (CI).

### `extract.py`

Script for extracting hidden states from encoder, decoder and encoder-decoder based models.

### `extract_llama.py`

Script for extracting representations from LLaMA — just `extract.py` adapted for MetaLlama8b models.

### `format_results.py`

Post-processing utility that:

* Aggregates layerwise metric results (ESA, PC, CI).
* Computes group statistics (means, std, confidence intervals).
* Formats results for plotting or tabular reporting.

## Usage Notes

* All scripts assume that the input data is already formatted as sentence pairs with polarity labels.
* The dataset with pairs should be organized as follows. From $0, 1, 2 .... N$ sentences, the first $\frac{N}{2}$ harm (or safe) and the next $\frac{N}{2}$ safe (or harm). For 0, the pair index is $\frac{N}{2}$, for 1, the pair index is $\frac{N}{2} + 1$ and so on.
* The model interface is built on top of HuggingFace converters.

# **`notebooks/` — PA-CCS Evaluation Notebooks**

This folder contains Jupyter notebooks for running and plotting **Polarity-Aware CCS (PA-CCS)** on various language models. The notebooks are intended to reproduce key experiments and visualize model alignment behavior.

## Files

* **`ccs_deberta_pretr.ipynb`**
  Runs CCS and PA-CCS on **DeBERTa-large-FT** (`Elron/deberta-v3-large-hate`). Includes:

 * Extract hidden representations
 * PA-CCC training
 * Calculate and format ESA/PC/CI metrics
 * Visual diagnostics of metrics and separations

* **`ccs_Meta-Llama-3-8B-Instruct.ipynb`**
  Applies PA-CCS to **Meta-LLaMA-3 8B Instruct** using reformulated harmful-safe statement pairs. Includes:

 * Extract hidden representations
 * PA-CCC training
 * Calculate and format ESA/PC/CI metrics
 * Visual diagnostics of metrics and separations

# **`data/` — datasets introduced in paper**


#### Requirements

* Python ≥ 3.9
* `transformers`, `torch`, `datasets`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `sentencepiece
`, `accelerate`, `tqdm`
* For LLaMA models: use HF-compatible checkpoints (with proper access)
