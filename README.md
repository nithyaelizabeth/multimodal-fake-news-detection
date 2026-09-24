# Multimodal Fake News Detection Using Text and Images

This repository contains the implementation and experimental results of my M598 dissertation project on multimodal fake-news detection. The project compares text-based and multimodal machine-learning models and investigates whether combining textual and visual information improves fake-news classification.

## Project Aim

The main aim is to evaluate text-based and multimodal models for fake-news detection and investigate how well they perform on internal and external datasets.

## Research Questions

1. How effectively can text-based and multimodal deep-learning models classify fake and real news?
2. Does combining text and image information improve performance compared with using only one modality?
3. How well do the evaluated models generalise to unseen misinformation data?

## Models

The following models and configurations were evaluated:

- TF-IDF with Logistic Regression
- BERT
- CLIP
- ViLT
- BLIP with BERT
- Text-only BERT features
- Image-only CLIP features
- BERT–CLIP feature fusion
- Fusion with modality dropout

## Experiments

The project includes:

- Leakage-aware dataset splitting
- TF-IDF baseline classification
- Reproducible BERT training
- Common-subset model comparison
- Modality-ablation experiments
- Controlled multimodal fusion
- Confidence-calibration analysis
- Bootstrap 95% confidence intervals
- McNemar statistical tests
- Robustness analysis
- External evaluation using VERITE

## Dataset Preparation

The project uses labelled multimodal fake-news data containing news text, associated images and binary labels.

Duplicate and related records were grouped before splitting to reduce information leakage between the training, validation and test sets.

| Dataset division | Number of samples |
|---|---:|
| Training | 399,342 |
| Validation | 80,724 |
| Test | 80,210 |
| Total | 560,276 |

The complete datasets and downloaded images are not included in this repository because of their size and redistribution restrictions.

## Main Results

### Reproducible Text Experiments

| Model | Macro F1 |
|---|---:|
| TF-IDF with Logistic Regression | 0.7721 |
| BERT | 0.8342 |

BERT improved macro F1 by 0.0621, corresponding to 6.21 percentage points or an approximate relative improvement of 8.04% over the TF-IDF baseline.

### Controlled Multimodal Fusion

| Configuration | Mean macro F1 | Standard deviation |
|---|---:|---:|
| Image-only CLIP features | 0.7888 | 0.0015 |
| Text-only BERT features | 0.8296 | 0.0006 |
| Modality-dropout fusion | 0.8482 | 0.0019 |
| Normal text-image fusion | 0.8541 | 0.0031 |

Normal text-image fusion produced the strongest result in the controlled experiment. It improved macro F1 by approximately 2.45 percentage points compared with the text-only configuration.

### External Generalisation

| Model | VERITE macro F1 |
|---|---:|
| BLIP+BERT | 0.5629 |
| BERT | 0.5261 |
| CLIP | 0.3112 |
| ViLT | 0.1288 |

All models performed worse on VERITE than on the internal dataset. This indicates that strong internal performance does not necessarily guarantee successful generalisation to data from a different source.

Results obtained from different sample sets or experimental conditions should not be interpreted as direct model rankings.

## Repository Structure

```text
multimodal-fake-news-detection/
├── notebooks/
│   └── Multimodal.ipynb
├── src/
│   ├── build_reproducible_splits.py
│   ├── train_tfidf_baseline.py
│   ├── train_bert_reproducible.py
│   ├── predict_bert.py
│   ├── predict_clip.py
│   ├── predict_vilt.py
│   ├── predict_blip_caption_bert.py
│   ├── extract_fusion_features.py
│   ├── train_controlled_fusion.py
│   ├── modality_ablation.py
│   ├── analyze_calibration.py
│   ├── significance_test.py
│   └── evaluate_verite.py
├── results/
│   ├── controlled_fusion_2k/
│   ├── figures/
│   ├── predictions/
│   └── tables/
├── EXPERIMENTS.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/nithyaelizabeth/multimodal-fake-news-detection.git
cd multimodal-fake-news-detection
```

Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Project

### Create the reproducible dataset splits

Place the source data in the required local data directories and run:

```bash
python src/build_reproducible_splits.py
```

This creates the training, validation and test manifests inside `data/splits/`.

### Train the TF-IDF baseline

```bash
python src/train_tfidf_baseline.py
```

### Train BERT

```bash
python src/train_bert_reproducible.py \
  --max-train-samples 20000 \
  --max-validation-samples 5000 \
  --max-test-samples 5000 \
  --epochs 3
```

For an Apple Silicon Mac with limited memory:

```bash
python src/train_bert_reproducible.py \
  --epochs 2 \
  --patience 1 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --optimizer adafactor
```

### Evaluate Predictions

```bash
python src/evaluate_predictions.py \
  results/predictions/bert_test.csv \
  --model-name BERT
```

### Controlled Multimodal Fusion

First extract the BERT and CLIP features:

```bash
python src/extract_fusion_features.py \
  --input data/splits/fakeddit_train.csv \
  --output data/features/fusion_train.npz \
  --failures results/tables/fusion_train_failures.csv \
  --max-samples 2000
```

Repeat the feature-extraction step for the validation and test sets.

Then train the controlled-fusion configurations:

```bash
python src/train_controlled_fusion.py \
  --train data/features/fusion_train.npz \
  --validation data/features/fusion_validation.npz \
  --test data/features/fusion_test.npz
```

Further instructions are available in [EXPERIMENTS.md](EXPERIMENTS.md).

## Evaluation Measures

The following measures are used:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- Macro F1
- Weighted F1
- Bootstrap 95% confidence intervals
- Brier score
- Expected Calibration Error
- McNemar’s exact test

Macro F1 is used as the main classification measure because it gives equal importance to the fake and real classes.

## Reproducibility

The main random seed is 42. The controlled-fusion experiment uses seeds 42, 43 and 44.

Prediction files use the following format:

```text
sample_id,true_label,prediction,probability
```

The class labels are:

- `0`: real news
- `1`: fake news

Processing failures are recorded separately so that evaluation coverage can be reported.

## Limitations

- The complete datasets and images are not included.
- Some experiments use limited samples because transformer and vision-language models require substantial computational resources.
- Model performance may be influenced by dataset-specific patterns.
- Performance decreases on external data.
- The system does not retrieve external evidence or generate fact-checking explanations.
- Automated predictions should not replace professional human fact-checking.

## Ethical Considerations

This project uses previously labelled secondary data. The models may produce incorrect or biased predictions. Therefore, the system should be treated as a research prototype and not as a final decision-making tool.

## Author

**Nithya Elizabeth**

M598 Dissertation Project

## Repository

[https://github.com/nithyaelizabeth/multimodal-fake-news-detection](https://github.com/nithyaelizabeth/multimodal-fake-news-detection)