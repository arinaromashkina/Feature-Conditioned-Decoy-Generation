# ENGPE: Empirical Null Generation-based Performance Evaluation

**Label-free performance estimation for deep neural networks under data distribution shift.**

> Arina Romashkina, Viktoria Fokina, Attila Kertesz-Farkas  
> Laboratory on AI for Computational Biology, HSE University

---

## Overview

Modern deep learning models can silently degrade in production when test data shifts from the training distribution. Standard evaluation requires ground-truth labels — but in deployment, labels are often unavailable.

**ENGPE** estimates key performance metrics (accuracy, precision, recall, F1, FDR) on unlabeled test sets, even under heavy distribution shift, without retraining or reaccessing the original data.

### How it works

ENGPE augments a pretrained classifier $F = F_C \circ F_R$ with a shallow generator $G$ that produces **empirical null scores** conditioned on the latent representation $h = F_R(t)$:

```
Input t
   │
   ├──► F_R (feature extractor) ──► h ──► F_C ──► prediction logits F(t)
   │                                  │
   │                                  └──► G (CNF generator) ──► null logits G(t)
   │
   └─────────────────────────────────────────────────────────────────────────────
                                Mix-Max FDR control
                                TP / FP / TN / FN estimation
                                → Accuracy, Precision, Recall, F1 (label-free)
```

Because both $F$ and $G$ are conditioned on the same latent $h$, any distribution shift in $F_R$ propagates equally to both branches — so the null tracks the shifted test distribution without retraining.

### Key components

| Component | Description |
|---|---|
| **Conditional Normalizing Flow (CNF)** | Shallow generator trained to approximate the empirical null distribution of classifier logits |
| **Robust Feature Normalization** | Median/IQR-based normalization + tanh compression to handle OOD feature blow-up |
| **Mix-Max FDR control** | Rigorous FDR estimation at every decision threshold (Keich et al.) |
| **ENGPE-TA** | ENGPE with threshold adaptation — finds the optimal threshold to maximize accuracy |

---

## Repository Structure

```
project_decoy_gen/
│
├── flows/                          # Conditional Normalizing Flow implementation
│   ├── flow_FN.py                        # Canonical CNF model (RobustFeatureNormalizer +
│   │                                     #   CouplingLayer + ActNorm + ScoreShiftFlowWrapper)
│   └── shift_flow.py                     # Backward-compat re-export from flow_FN.py
│
├── fdr/                            # FDR estimation and control
│   ├── fdr_control.py              # Mix-Max, BH, q-value computation, accuracy estimation
│   └── plot_fdr.py                 # FDR curve plotting utilities
│
├── data_processing/                # Dataset utilities
│   ├── score_feature_dataset.py    # Dataset class for (logit, feature) pairs
│   └── negative_scores_pool.py     # Construction of null logit training targets
│
├── utils/
│   ├── other_methods.py            # Baseline methods (ATC, DOC, MaNo, etc.)
│   └── visualize_distributions.py  # Score distribution plots
│
├── camelyon17/                     # Camelyon17 experiment (binary, hospital shift)
├── cifar-10-c/                     # CIFAR-10-C experiment (multi-class, corruption shift)
├── BCSS/                           # BCSS experiment (pixel classification, subpopulation shift)
├── BREEDS/                         # BREEDS experiment (ImageNet subpopulation shift)
│
├── results_ds/                     # Saved numerical results (CSV)
├── figures/publication/            # Publication-ready figures
│
├── mnist_pipeline.py               # MNIST end-to-end pipeline (entry point)
├── bcss_pipeline.py                # BCSS end-to-end pipeline
├── make_breeds.py                  # BREEDS dataset preparation and evaluation
├── make_cifar-10-c.py              # CIFAR-10-C evaluation script
├── publication_plots.py            # Script to reproduce all publication figures
│
└── TD-GAN_v1.tex                   # Paper draft (NeurIPS 2026 submission format)
```

---

## Method Details

### Conditional Normalizing Flow (CNF)

The generator $G_K : \mathbb{R}^K \to \mathbb{R}^K$ is a chain of $N_f = 12$ affine coupling layers with ActNorm, conditioned on the encoded feature vector $e = \text{Encoder}(\tilde{h})$.

**Robust feature normalization** prevents OOD blow-up:
$$\tilde{h}_j = \tanh\!\left(\frac{h_j - \hat{m}_j}{c\,\hat{q}_j + \varepsilon}\right)$$
where $\hat{m}_j$, $\hat{q}_j$ are running median and IQR over the training set, $c = 5$, $\varepsilon = 10^{-6}$.

The flow is trained with maximum log-likelihood on **null logit vectors** constructed by replacing the class-$k$ coordinate of each training sample's logit with a score drawn from training samples that do not belong to class $k$.

### FDR-based Performance Estimation

At test time, for each sample $t$, the generator produces a null logit vector $n_t = G_K^{-1}(z \mid e)$, $z \sim \mathcal{N}(0, I)$.

The **Mix-Max algorithm** uses these null scores to estimate FDP at every decision threshold $s_{th}$, which in turn gives label-free estimates of TP, FP, TN, FN:

$$TP(s_{th}) = |\{t : F(t) \ge s_{th}\}| \cdot (1 - \widehat{FDP}(s_{th}))$$

**ENGPE-TA** selects the threshold that maximizes estimated accuracy:
$$ACC_{TA} = \max_{s_{th}} \frac{\widehat{TP}(s_{th}) + \widehat{TN}(s_{th})}{|T|}$$

---

## Datasets and Experiments

| Dataset | Task | Shift type | Backbone |
|---|---|---|---|
| **Camelyon17** | Binary (tumour detection) | Hospital / domain shift | EfficientNet-B0 |
| **CIFAR-10-C** | 10-class (image classification) | Corruption shift (19 types × 5 severities) | Wide ResNet-28-10 |
| **BCSS** | 5-class pixel classification | Subpopulation + covariate shift | FCN ResNet-50 UNet |
| **BREEDS** | 13–30 class (ImageNet subsets) | Subpopulation shift | ResNet-50 |

---

## Installation

```bash
pip install torch torchvision numpy pandas matplotlib scipy statsmodels wilds
```

For BREEDS experiments, additionally install the [BREEDS benchmark](https://github.com/MadryLab/BREEDS-Benchmarks).

---

## Usage

### 1. Prepare score/feature datasets

Extract logits and penultimate-layer features from a pretrained classifier and save them as `(scores, features, labels)` tensors. See `data_processing/score_feature_dataset.py`.

### 2. Train the CNF generator

```python
from flows.flow_FN import ScoreShiftFlowWrapper

flow = ScoreShiftFlowWrapper(num_classes=K, n_flows=12, feature_dim=D,
                              hidden_dim=256, encoder_dim=128, clip_val=5.0)
flow.train_flow(train_dataset, epochs=30, lr=3e-4, device=device)
```

### 3. Generate null scores and estimate performance

```python
from fdr.fdr_control import control_fdr_mixmax, compute_method_estimation_curve

target_scores, decoy_scores, labels = flow.generate_decoys(test_score_dataset, device=device)

df = control_fdr_mixmax(target_scores, labels, decoy_scores)
curve = compute_method_estimation_curve(df, q_value_column='q_values_mixmax', pi0=0.0)

print(f"Estimated accuracy: {curve['Accuracy_est'].max():.3f}")
```

### End-to-end example (MNIST)

```bash
python mnist_pipeline.py
```

---

## Results

ENGPE-TA reduces mean absolute error (MAE) in accuracy estimation by **36%** compared to current state-of-the-art methods (ATC, DOC, MaNo, Nuclear Norm, Energy Score) across all benchmarks.

Publication figures are in [`figures/publication/`](figures/publication/) and can be reproduced with:

```bash
python publication_plots.py
```

---

## Citation

```bibtex
@article{romashkina2026engpe,
  title   = {Label-free performance evaluation with empirical null generation 
             under data distribution shift in multi-class classification problems},
  author  = {Romashkina, Arina and Fokina, Viktoria and Kertesz-Farkas, Attila},
  journal = {NeurIPS},
  year    = {2026}
}
```
