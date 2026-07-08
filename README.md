# FairVision 🎯

**Detecting and Mitigating Bias in a CNN-Based Age Group Classification System Using FairFace**

FairVision is a from-scratch PyTorch CNN that predicts age group from a facial image — and, more importantly, audits and mitigates demographic bias in its own predictions. Built for the IJSE Certified AI & ML Engineer programme, the project treats fairness as a first-class objective alongside accuracy: every model is evaluated not just on overall accuracy, but on how consistently it performs across race and gender subgroups.

🔗 **Live demo:** [fairvision-sinesh-gimshan.streamlit.app](https://fairvision-sinesh-gimshan.streamlit.app/)

---

## Overview

Face-based AI systems are widely deployed in surveillance, biometric authentication, attendance tracking, and smart retail — yet research consistently shows they can perform unevenly across demographic groups. FairVision investigates this problem directly by:

1. Building a CNN that classifies faces into **9 age groups**: `0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+`
2. Auditing subgroup performance across **7 race categories** and **2 gender categories**
3. Applying practical, lightweight bias-mitigation techniques and measuring their impact
4. Shipping the result as an interactive, deployable web application

Two models were trained and compared:

| Model | Description |
|---|---|
| **Baseline CNN (model-1)** | Standard training — CrossEntropyLoss, uniform sampling, Adam optimizer |
| **Mitigated CNN (model-2)** | Fairness-aware training — WeightedRandomSampler, label smoothing (0.1), AdamW + ReduceLROnPlateau |

## Key Features

- ✅ Custom **FiveBlockCNN** architecture trained entirely from scratch (no pretrained backbone)
- ✅ Full EDA covering age, race, and gender distribution imbalance
- ✅ Stratified train/validation splitting to preserve class proportions
- ✅ Data augmentation pipeline (flip, rotation, brightness/contrast) for generalization
- ✅ Class-imbalance mitigation via inverse-sqrt-frequency **WeightedRandomSampler**
- ✅ Multi-GPU training support (`DataParallel`)
- ✅ Subgroup fairness auditing across race and gender
- ✅ Confusion matrix, misclassification, and prediction-confidence analysis
- ✅ Deployed **Streamlit** inference app with Test-Time Augmentation, temperature scaling, Top-K predictions, and uncertainty scoring

## Dataset

- **Source:** [FairFace](https://huggingface.co/datasets/HuggingFaceM4/FairFace) via Hugging Face (`0.25` config, 224×224 cropped images)
- **Full dataset:** 86,744 training images / 10,954 validation images (held out as the final unseen test set)
- **Labels used:** age (9 classes), race (7 classes), gender (2 classes) — `service_test` field ignored

| Model | Train samples | Internal val samples | Test samples (held-out) |
|---|---|---|---|
| Baseline | 20,000 | 5,000 | 10,954 |
| Mitigated | 24,000 | 6,000 | 10,954 |

**Known dataset characteristics:**
- Strong age-group imbalance (middle-age classes 20–39 dominate; `0–2`, `60–69`, `70+` are minority classes)
- Moderate race-group imbalance (White and Latino/Hispanic slightly overrepresented)
- Near-balanced gender distribution
- Inherent label ambiguity near age-group boundaries (e.g., 29 vs. 31)

## Model Architecture

**FiveBlockCNN** — a 5-block convolutional network built for hierarchical facial feature extraction:

| Block | Filters |
|---|---|
| 1 | 32 |
| 2 | 64 |
| 3 | 128 |
| 4 | 256 |
| 5 | 512 |

- 3×3 convolution kernels throughout
- Batch Normalization + ReLU after every convolution
- MaxPooling for spatial compression at the end of each block
- Adaptive Average Pooling before the classifier head
- Fully connected classifier: Flatten → Linear → ReLU → Dropout → 9-way output
- Dropout2D + standard Dropout for regularization

The architecture was deliberately kept lightweight and interpretable rather than maximally deep, to keep training from scratch tractable and to isolate the effect of fairness interventions from architectural changes.

## Methodology

1. **Preprocessing:** resize to 224×224, ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), custom collate + DataLoader pipeline (batch size 32)
2. **Augmentation** (train only): horizontal flip, slight rotation, brightness/contrast jitter
3. **Training:** 40 epochs, mini-batch learning
   - Baseline: CrossEntropyLoss, Adam, uniform sampling
   - Mitigated: CrossEntropyLoss with label smoothing = 0.1, AdamW, ReduceLROnPlateau, WeightedRandomSampler (weights ∝ 1/√class frequency)
4. **Evaluation:** held-out FairFace validation split used as the final unseen test set

## Results

| Metric | Baseline (model-1) | Mitigated (model-2) |
|---|---|---|
| Best validation accuracy | 51.50% (epoch 38) | 51.60% (epoch 39) |
| Final training accuracy | 60.57% | 63.35% |
| Final validation accuracy | 50.80% | 50.95% |
| Convergence behavior | Validation-loss spikes (~epoch 20, 28), moderate overfitting | Smoother convergence, fewer spikes, more stable |

**Final test evaluation (mitigated model):**

| Metric | Value |
|---|---|
| Test Accuracy | 50.37% |
| Test Loss | 1.4283 |
| Macro F1-score | 0.48 |
| Weighted F1-score | 0.50 |
| Top-3 Accuracy | 92.10% |

- **Strongest classes:** `3–9`, `0–2`
- **Weakest class:** `70+` (fewest training samples)
- **Most common confusions:** between adjacent age groups (e.g., `20–29 ↔ 30–39`, `30–39 ↔ 40–49`, `10–19 ↔ 20–29`) — indicating the model learns real age-progression patterns but struggles with precise boundaries
- Mean confidence: **0.560** for correct predictions vs. **0.472** for incorrect predictions (overlapping distributions — occasional confident errors)

## Fairness Audit

| Gender | Accuracy |
|---|---|
| Male | 49.27% |
| Female | 51.61% |

**Gender gap: 2.34 pp** — relatively small, indicating fairly consistent performance across gender.

| Race Group | Accuracy |
|---|---|
| East Asian | 53.6% |
| Indian | 52.0% |
| Southeast Asian | 51.7% |
| Latino/Hispanic | 50.5% |
| Middle Eastern | 50.7% |
| White | 48.6% |
| Black | 46.3% |

**Race gap: 7.3 pp** — East Asian subgroup performs best, Black subgroup performs worst. This is a measurable, non-trivial disparity even on a dataset explicitly designed for demographic balance.

## Demo Application

A Streamlit app provides real-time inference on uploaded images, including:
- Top prediction + confidence score
- Top-K predicted classes with probabilities
- Full probability distribution & radar-chart view
- Test-Time Augmentation (TTA) toggle
- Temperature scaling for calibration
- Uncertainty scoring

▶️ **Try it:** [fairvision-sinesh-gimshan.streamlit.app](https://fairvision-sinesh-gimshan.streamlit.app/)

## Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch, torchvision
- **Data:** Hugging Face `datasets`, pandas, scikit-learn (stratified splitting)
- **Visualization:** Matplotlib
- **Deployment:** Streamlit / Streamlit Cloud
- **Compute:** Multi-GPU training via `DataParallel`

## Project Structure

```
fairvision/
├── data/                   # Dataset loading & preprocessing scripts
├── notebooks/              # EDA, training, and evaluation notebooks
├── models/
│   ├── fiveblockcnn.py     # Custom CNN architecture
│   ├── baseline/           # model-1 checkpoints
│   └── mitigated/          # model-2 checkpoints (best_fairface_model.pth)
├── fairness/                # Subgroup auditing scripts
├── app/
│   └── streamlit_app.py    # Deployed inference app
├── reports/
│   └── Fair-Vision_Technical_Report.pdf
├── requirements.txt
└── README.md
```

> ⚠️ Adjust this structure to match your actual repo layout before publishing.

## Installation

```bash
git clone https://github.com/<your-username>/fairvision.git
cd fairvision
pip install -r requirements.txt
```

**requirements.txt should include (at minimum):**
```
torch
torchvision
datasets
streamlit
pandas
scikit-learn
matplotlib
```

## Usage

**Train the mitigated model:**
```bash
python train.py --mode mitigated --epochs 40 --batch-size 32
```

**Evaluate on the held-out test set:**
```bash
python evaluate.py --checkpoint models/mitigated/best_fairface_model.pth
```

**Run the fairness audit:**
```bash
python fairness_audit.py --checkpoint models/mitigated/best_fairface_model.pth
```

**Launch the demo app locally:**
```bash
streamlit run app/streamlit_app.py
```

## Limitations

- Trained on a **stratified subset** (20K–24K images) of the full 86,744-image FairFace training set due to compute constraints — full-dataset training may improve generalization
- `70+` age class remains difficult due to very limited samples
- Age-group boundaries are inherently ambiguous (gradual aging vs. discrete labels)
- Overall accuracy (~50%) is moderate — reflects the difficulty of 9-way age classification from a lightweight from-scratch CNN, not a bug
- Race-based fairness gap (7.3 pp) persists despite mitigation and should be reduced further before any real-world use

See Section 8 of the [technical report](reports/Fair-Vision_Technical_Report.pdf) for the full ethical discussion.

## Author

**Sinesh Gimshan Senarathne**


---

*For the full methodology, code walkthroughs, and complete evaluation charts, see the [Technical Report](reports/Fair-Vision_Technical_Report.pdf).*
