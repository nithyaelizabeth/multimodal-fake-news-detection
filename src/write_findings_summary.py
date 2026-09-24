from pathlib import Path
import json

import pandas as pd


OUTPUT_PATH = Path("results/tables/final_findings_summary.md")


def table(path):
    df = pd.read_csv(path)
    return df.to_markdown(index=False)


def main():
    sections = []

    sections.append("# Final Findings Summary\n")

    bert_evaluation = json.loads(
        Path("results/tables/bert_20k_evaluation.json").read_text(encoding="utf-8")
    )
    bert_metrics = bert_evaluation["metrics"]

    sections.append("## 1. Primary Result: Reproducible BERT Baseline\n")
    primary_rows = []
    metric_labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "macro_f1": "Macro F1",
        "fake_precision": "Fake precision",
        "fake_recall": "Fake recall",
        "fake_f1": "Fake F1",
        "real_recall": "Real recall",
    }
    for metric, label in metric_labels.items():
        result = bert_metrics[metric]
        primary_rows.append({
            "Metric": label,
            "Value": result["value"],
            "95% CI lower": result["lower_95"],
            "95% CI upper": result["upper_95"],
        })
    sections.append(pd.DataFrame(primary_rows).to_markdown(index=False, floatfmt=".4f"))
    sections.append(
        "\n\nThe reproducible BERT baseline was trained on 20,000 samples, selected "
        "using 5,000 validation samples, and evaluated on 5,000 test samples. It "
        "achieved a macro F1-score of 0.8342 (95% CI: 0.8236–0.8444). Fake-news "
        "recall was 0.8452 and real-news recall was 0.8327, indicating relatively "
        "balanced class performance. This larger evaluation is the primary "
        "quantitative result of the study.\n"
    )

    tfidf_evaluation = json.loads(
        Path("results/tables/tfidf_evaluation.json").read_text(encoding="utf-8")
    )
    tfidf_metrics = tfidf_evaluation["metrics"]
    baseline_rows = [
        {
            "Model": "TF-IDF + Logistic Regression",
            "Test samples": tfidf_evaluation["samples"],
            "Accuracy": tfidf_metrics["accuracy"]["value"],
            "Balanced accuracy": tfidf_metrics["balanced_accuracy"]["value"],
            "Macro F1": tfidf_metrics["macro_f1"]["value"],
        },
        {
            "Model": "BERT",
            "Test samples": bert_evaluation["samples"],
            "Accuracy": bert_metrics["accuracy"]["value"],
            "Balanced accuracy": bert_metrics["balanced_accuracy"]["value"],
            "Macro F1": bert_metrics["macro_f1"]["value"],
        },
    ]
    sections.append("\n## 2. Traditional Lexical Baseline Comparison\n")
    sections.append(pd.DataFrame(baseline_rows).to_markdown(index=False, floatfmt=".4f"))
    sections.append(
        "\n\nTF-IDF with class-balanced Logistic Regression was trained on the same "
        "20,000 samples, tuned on the same 5,000 validation samples, and evaluated "
        "on exactly the same 5,000 test sample IDs as BERT. TF-IDF achieved macro F1 "
        "of 0.7721 (95% CI: 0.7606–0.7839), while BERT achieved 0.8342 (95% CI: "
        "0.8236–0.8444). The non-overlapping confidence intervals and 0.0621 absolute "
        "macro-F1 advantage provide evidence that contextual transformer features add "
        "value beyond surface-level lexical patterns.\n"
    )

    sections.append("\n## 3. Expanded Four-Model Comparison: Fakeddit\n")
    sections.append(table("results/tables/model_comparison.csv"))
    sections.append(
        "\n\nThe expanded comparison requested 1,000 balanced test samples and "
        "successfully processed 957 (95.7% coverage). BERT achieved the strongest "
        "performance, with accuracy of 0.9112 and F1-score of 0.9119. "
        "ViLT was the strongest direct multimodal classifier, while CLIP + MLP "
        "and BLIP Caption + BERT showed weaker fake-news recall.\n"
    )

    sections.append("\n## 4. Class Error Analysis\n")
    sections.append(table("results/tables/class_error_analysis.csv"))
    sections.append(
        "\n\nClass-specific analysis showed that CLIP + MLP and BLIP Caption + BERT "
        "were biased toward predicting Real, causing many fake posts to be missed. "
        "BERT produced the most balanced Real and Fake recall.\n"
    )

    sections.append("\n## 5. Robustness Under Text Perturbations\n")
    sections.append(table("results/tables/robustness_summary.csv"))
    sections.append(
        "\n\nThe text robustness experiment requested 500 samples and successfully "
        "processed 480 (96.0% coverage). It measured whether "
        "predictions changed after "
        "minor text modifications. Lower change rates indicate greater prediction "
        "stability, but stability must be interpreted together with accuracy because "
        "a biased model can appear stable by predicting the same class repeatedly.\n"
    )

    sections.append("\n## 6. Robustness Under Image Transformations\n")
    sections.append(table("results/tables/visual_robustness_accuracy.csv"))
    sections.append(
        "\n\nThe visual robustness experiment requested 200 samples and successfully "
        "processed 194 (97.0% coverage). It showed how "
        "multimodal models respond when "
        "the image is blurred, cropped, darkened, or converted to grayscale. BERT is "
        "excluded from this test because it does not use image input.\n"
    )

    sections.append("\n## 7. Cross-Dataset Generalization: Fakeddit to VERITE\n")
    sections.append(table("results/tables/generalization_gap.csv"))
    sections.append(
        "\n\nThis comparison evaluated all 1,001 VERITE samples with 100% coverage. "
        "BERT performed best on Fakeddit but showed a substantial "
        "drop on VERITE. "
        "BLIP Caption + BERT generalized better to VERITE, likely because VERITE "
        "focuses on image-caption consistency, which matches the caption-based "
        "architecture.\n"
    )

    sections.append("\n## 8. VERITE Type Analysis\n")
    sections.append(table("results/tables/verite_type_analysis.csv"))
    sections.append(
        "\n\nVERITE type analysis separates performance across truthful, miscaptioned, "
        "and out-of-context samples. BLIP + BERT detected out-of-context items best "
        "(0.5046 accuracy) and also performed best on miscaptioned items (0.4615). "
        "ViLT correctly classified most truthful items (0.9408) but detected very few "
        "miscaptioned (0.0769) or out-of-context (0.0646) items, indicating a strong "
        "tendency to predict the truthful class on VERITE. No model exceeded 0.51 on "
        "either misinformation type, demonstrating that cross-dataset multimodal "
        "misinformation detection remains difficult.\n"
    )

    sections.append("\n## 9. Modality Ablation Analysis\n")
    sections.append(table("results/tables/modality_ablation_summary.csv"))
    sections.append(
        "\n\nThe modality-ablation experiment requested 200 samples and retained 188 "
        "paired successful samples for every condition and model. ViLT depended "
        "strongly on visual alignment: macro F1 fell from 0.7300 with the correct "
        "image to 0.3761 with a shuffled image and 0.4418 with a blank image. Its "
        "fake-news recall fell from 0.5591 to 0.0430 with shuffled images. In "
        "contrast, CLIP + MLP produced identical macro F1 (0.6335) for correct and "
        "shuffled images, suggesting that its classifier relied predominantly on "
        "text or failed to exploit image-text correspondence. For BLIP + BERT, "
        "caption-only input achieved macro F1 of 0.5107, compared with 0.4737 for "
        "text plus the correct caption and 0.3989 for text only. This indicates that "
        "generated captions contain useful signal, although combining them with the "
        "original text did not improve performance in this checkpoint.\n"
    )

    sections.append("\n## 10. Controlled Multimodal Fusion Extension\n")
    sections.append(table("results/controlled_fusion_2k/summary.csv"))
    sections.append(
        "\n\nThis controlled extension used frozen BERT text features and frozen CLIP "
        "image features. After image-retrieval failures were excluded, all four "
        "conditions used exactly the same 1,892 training, 479 validation, and 481 "
        "test samples. Results were averaged across seeds 42, 43, and 44. Normal "
        "text-image fusion achieved the strongest mean macro F1 (0.8541, SD "
        "0.0031), compared with 0.8296 (SD 0.0006) for text-only features and "
        "0.7888 (SD 0.0015) for image-only features. The 0.0245 absolute "
        "macro-F1 improvement over text-only features indicates that visual "
        "features added useful information under matched experimental conditions. "
        "Modality dropout achieved macro F1 of 0.8482 (SD 0.0019): it improved on "
        "both unimodal conditions but did not outperform ordinary fusion. The "
        "proposed regularization therefore did not provide an additional benefit "
        "under this configuration.\n"
    )

    sections.append("\n## 11. Confidence Calibration Analysis\n")
    sections.append(table("results/tables/calibration/calibration_summary.csv"))
    sections.append(
        "\n\nLower Brier scores and expected calibration error indicate more reliable "
        "probabilities. The reproducible BERT model was comparatively well "
        "calibrated (Brier score 0.1158; ECE 0.0467), and confidence was lower for "
        "incorrect than correct predictions. ViLT was strongly overconfident: it "
        "produced an ECE of 0.2407 and averaged 0.9014 confidence even when wrong. "
        "CLIP + MLP was also poorly calibrated, with greater average confidence on "
        "incorrect than correct predictions. BLIP Caption + BERT obtained low ECE "
        "because its probabilities remained close to 0.5, but its Brier score of "
        "0.2503 was approximately equivalent to an uninformative balanced binary "
        "predictor. Calibration metrics must therefore be interpreted together with "
        "discrimination and accuracy.\n"
    )

    sections.append("\n## 12. Structured Error-Analysis Progress\n")
    sections.append(table("results/tables/error_annotation_progress.csv"))
    sections.append(
        "\n\nA deterministic, class-aware sample of 100 incorrect predictions was "
        "created, containing 25 errors from each model. The table reports objective "
        "properties that can be calculated automatically. The qualitative causes "
        "of these errors have intentionally not been assigned automatically: the "
        "text and image must be inspected manually using the accompanying codebook. "
        "Until that review is completed, this is an annotation protocol and progress "
        "report rather than a completed qualitative finding.\n"
    )

    sections.append("\n## 13. Overall Conclusion\n")
    sections.append(
        "The reproducible text-only BERT baseline achieved a macro F1-score of "
        "0.8342 on 5,000 test samples. In the matched four-model comparison on 957 "
        "successfully processed samples, BERT again performed best, while the "
        "multimodal models showed lower fake-class recall. Evaluation on all 1,001 "
        "VERITE samples demonstrated substantial domain sensitivity. Robustness "
        "testing on 480 text samples and 194 image samples further showed that "
        "stability depends on both the model and perturbation type. The paired "
        "modality ablation provides direct evidence that ViLT uses image-text "
        "alignment, whereas CLIP + MLP showed little sensitivity to image identity. "
        "In a separate controlled three-seed experiment, normal fusion of frozen "
        "BERT and CLIP features improved mean macro F1 from 0.8296 for text-only "
        "features to 0.8541, while modality dropout did not improve on normal fusion. "
        "Calibration analysis further showed that high model confidence does not "
        "necessarily imply reliability, particularly for ViLT and CLIP + MLP.\n"
    )

    sections.append("\n## 14. Evidence Limitations\n")
    sections.append(
        "The four-model comparison processed 957 of 1,000 requested samples, text "
        "robustness processed 480 of 500, and visual robustness processed 194 of "
        "200 because some remote images were unavailable. Although coverage exceeded "
        "95% in each case, failed image retrieval may introduce selection bias. The "
        "models were originally trained with different sample budgets, so performance "
        "differences should not be attributed only to model architecture. The "
        "controlled fusion extension retained 1,892 of 2,000 requested training "
        "samples (94.6%), 479 of 500 validation samples (95.8%), and 481 of 500 test "
        "samples (96.2%). It used frozen encoders and a moderate test sample, so its "
        "results support a controlled comparison of feature fusion but do not "
        "establish that the fusion architecture is universally superior.\n"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(sections), encoding="utf-8")

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
