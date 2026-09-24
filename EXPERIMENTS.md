# Reproducible experiment workflow

## 1. Create the shared splits

Run from the project root:

```bash
.venv/bin/python src/build_reproducible_splits.py
```

This creates:

- `data/splits/fakeddit_train.csv`
- `data/splits/fakeddit_validation.csv`
- `data/splits/fakeddit_test.csv`
- `data/splits/split_metadata.json`

Rows with repeated normalized text or repeated image URLs are kept together.
All models must train and evaluate on these exact manifests. Use validation data
for model selection and test data only once for the final reported comparison.

## 2. Prediction-file contract

Each model should save one row per test sample with at least:

```text
sample_id,true_label,prediction,probability
```

Use labels `0` for real and `1` for fake. Do not silently skip failed samples.
Record failures separately and report the coverage rate.

## 3. Evaluate predictions

```bash
.venv/bin/python src/evaluate_predictions.py \
  results/predictions/bert_test.csv \
  --model-name BERT
```

For the existing wide prediction table, select the appropriate column:

```bash
.venv/bin/python src/evaluate_predictions.py \
  results/tables/model_predictions.csv \
  --prediction-column bert_prediction \
  --model-name BERT
```

The evaluator reports accuracy, balanced accuracy, macro/weighted F1,
class-specific metrics, and bootstrap 95% confidence intervals.

## 4. Train the reproducible BERT baseline

First run a small pipeline check:

```bash
.venv/bin/python src/train_bert_reproducible.py \
  --max-train-samples 20000 \
  --max-validation-samples 5000 \
  --max-test-samples 5000 \
  --epochs 3
```

Then evaluate its saved predictions:

```bash
.venv/bin/python src/evaluate_predictions.py \
  results/predictions/bert_test.csv \
  --model-name BERT
```

After the pilot completes successfully, omit the three `--max-*-samples`
arguments to train and evaluate on the complete shared manifests.

For an Apple Silicon Mac with limited MPS memory, use a small physical batch
and gradient accumulation:

```bash
.venv/bin/python src/train_bert_reproducible.py \
  --epochs 2 \
  --patience 1 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --optimizer adafactor \
  --output-dir models/bert_full \
  --predictions results/predictions/bert_full_test.csv \
  --run-summary results/tables/bert_full_training_summary.json
```

## 5. Minimum reporting standard

For every experiment, record the dataset manifest, random seed, model checkpoint,
epochs, learning rate, batch size, training sample count, evaluation coverage,
metrics with confidence intervals, and hardware used.

## 6. Controlled multimodal fusion extension

This experiment compares text-only, image-only, normal fusion, and fusion with
modality dropout on identical samples. First cache frozen features (start with
small values to check image availability):

```bash
.venv/bin/python src/extract_fusion_features.py --input data/splits/fakeddit_train.csv --output data/features/fusion_train.npz --failures results/tables/fusion_train_failures.csv --max-samples 20000
.venv/bin/python src/extract_fusion_features.py --input data/splits/fakeddit_validation.csv --output data/features/fusion_validation.npz --failures results/tables/fusion_validation_failures.csv --max-samples 5000
.venv/bin/python src/extract_fusion_features.py --input data/splits/fakeddit_test.csv --output data/features/fusion_test.npz --failures results/tables/fusion_test_failures.csv --max-samples 5000
```

Then train all four controlled classifiers with three seeds:

```bash
.venv/bin/python src/train_controlled_fusion.py --train data/features/fusion_train.npz --validation data/features/fusion_validation.npz --test data/features/fusion_test.npz
```

Results are written to `results/controlled_fusion/summary.csv`. All four
conditions use only rows with successfully loaded images, so their results are
paired and directly comparable. Feature extraction is the slow stage; training
the small classifiers is comparatively fast.
