# Final Findings Summary

## 1. Primary Result: Reproducible BERT Baseline

| Metric            |   Value |   95% CI lower |   95% CI upper |
|:------------------|--------:|---------------:|---------------:|
| Accuracy          |  0.8378 |         0.8274 |         0.8476 |
| Balanced accuracy |  0.8390 |         0.8287 |         0.8488 |
| Macro F1          |  0.8342 |         0.8236 |         0.8444 |
| Fake precision    |  0.7771 |         0.7595 |         0.7938 |
| Fake recall       |  0.8452 |         0.8290 |         0.8605 |
| Fake F1           |  0.8098 |         0.7965 |         0.8220 |
| Real recall       |  0.8327 |         0.8188 |         0.8458 |


The reproducible BERT baseline was trained on 20,000 samples, selected using 5,000 validation samples, and evaluated on 5,000 test samples. It achieved a macro F1-score of 0.8342 (95% CI: 0.8236–0.8444). Fake-news recall was 0.8452 and real-news recall was 0.8327, indicating relatively balanced class performance. This larger evaluation is the primary quantitative result of the study.


## 2. Traditional Lexical Baseline Comparison

| Model                        |   Test samples |   Accuracy |   Balanced accuracy |   Macro F1 |
|:-----------------------------|---------------:|-----------:|--------------------:|-----------:|
| TF-IDF + Logistic Regression |           5000 |     0.7752 |              0.7800 |     0.7721 |
| BERT                         |           5000 |     0.8378 |              0.8390 |     0.8342 |


TF-IDF with class-balanced Logistic Regression was trained on the same 20,000 samples, tuned on the same 5,000 validation samples, and evaluated on exactly the same 5,000 test sample IDs as BERT. TF-IDF achieved macro F1 of 0.7721 (95% CI: 0.7606–0.7839), while BERT achieved 0.8342 (95% CI: 0.8236–0.8444). The non-overlapping confidence intervals and 0.0621 absolute macro-F1 advantage provide evidence that contextual transformer features add value beyond surface-level lexical patterns.


## 3. Expanded Four-Model Comparison: Fakeddit

| Model       |   Samples |   Accuracy |   Precision |   Recall |   F1-score |
|:------------|----------:|-----------:|------------:|---------:|-----------:|
| BERT        |       957 |   0.911181 |    0.903491 | 0.920502 |   0.911917 |
| CLIP + MLP  |       957 |   0.643678 |    0.858639 | 0.343096 |   0.490284 |
| ViLT        |       957 |   0.680251 |    0.807143 | 0.472803 |   0.596306 |
| BLIP + BERT |       957 |   0.529781 |    0.589744 | 0.192469 |   0.290221 |


The expanded comparison requested 1,000 balanced test samples and successfully processed 957 (95.7% coverage). BERT achieved the strongest performance, with accuracy of 0.9112 and F1-score of 0.9119. ViLT was the strongest direct multimodal classifier, while CLIP + MLP and BLIP Caption + BERT showed weaker fake-news recall.


## 4. Class Error Analysis

| Model               |   True Real |   False Fake |   False Real |   True Fake |   Real Recall |   Fake Recall |   Macro F1 |
|:--------------------|------------:|-------------:|-------------:|------------:|--------------:|--------------:|-----------:|
| BERT                |         432 |           47 |           38 |         440 |      0.901879 |      0.920502 |   0.911175 |
| CLIP + MLP          |         452 |           27 |          314 |         164 |      0.943633 |      0.343096 |   0.608194 |
| ViLT                |         425 |           54 |          252 |         226 |      0.887265 |      0.472803 |   0.6658   |
| BLIP Caption + BERT |         415 |           64 |          386 |          92 |      0.866388 |      0.192469 |   0.469329 |


Class-specific analysis showed that CLIP + MLP and BLIP Caption + BERT were biased toward predicting Real, causing many fake posts to be missed. BERT produced the most balanced Real and Fake recall.


## 5. Robustness Under Text Perturbations

| model               | perturbation       |   count |       mean |   change_rate_percent |
|:--------------------|:-------------------|--------:|-----------:|----------------------:|
| BERT                | add_noise          |     480 | 0.216667   |             21.6667   |
| BERT                | remove_one_word    |     480 | 0.0479167  |              4.79167  |
| BERT                | remove_punctuation |     480 | 0          |              0        |
| BLIP Caption + BERT | add_noise          |     480 | 0.0520833  |              5.20833  |
| BLIP Caption + BERT | remove_one_word    |     480 | 0.0458333  |              4.58333  |
| BLIP Caption + BERT | remove_punctuation |     480 | 0          |              0        |
| CLIP + MLP          | add_noise          |     480 | 0.00833333 |              0.833333 |
| CLIP + MLP          | remove_one_word    |     480 | 0.0166667  |              1.66667  |
| CLIP + MLP          | remove_punctuation |     480 | 0          |              0        |
| ViLT                | add_noise          |     480 | 0.0770833  |              7.70833  |
| ViLT                | remove_one_word    |     480 | 0.0625     |              6.25     |
| ViLT                | remove_punctuation |     480 | 0          |              0        |


The text robustness experiment requested 500 samples and successfully processed 480 (96.0% coverage). It measured whether predictions changed after minor text modifications. Lower change rates indicate greater prediction stability, but stability must be interpreted together with accuracy because a biased model can appear stable by predicting the same class repeatedly.


## 6. Robustness Under Image Transformations

| Model               | Change    |   Samples |   Original Accuracy |   Changed Accuracy |   Prediction Change Rate |
|:--------------------|:----------|----------:|--------------------:|-------------------:|-------------------------:|
| BLIP Caption + BERT | blur      |       194 |            0.546392 |           0.525773 |                12.3711   |
| BLIP Caption + BERT | crop      |       194 |            0.546392 |           0.551546 |                10.8247   |
| BLIP Caption + BERT | darken    |       194 |            0.546392 |           0.561856 |                12.8866   |
| BLIP Caption + BERT | grayscale |       194 |            0.546392 |           0.536082 |                14.433    |
| CLIP + MLP          | blur      |       194 |            0.659794 |           0.664948 |                 0.515464 |
| CLIP + MLP          | crop      |       194 |            0.659794 |           0.659794 |                 0        |
| CLIP + MLP          | darken    |       194 |            0.659794 |           0.659794 |                 1.03093  |
| CLIP + MLP          | grayscale |       194 |            0.659794 |           0.659794 |                 0        |
| ViLT                | blur      |       194 |            0.737113 |           0.680412 |                11.8557   |
| ViLT                | crop      |       194 |            0.737113 |           0.737113 |                 8.24742  |
| ViLT                | darken    |       194 |            0.737113 |           0.737113 |                 5.15464  |
| ViLT                | grayscale |       194 |            0.737113 |           0.680412 |                11.8557   |


The visual robustness experiment requested 200 samples and successfully processed 194 (97.0% coverage). It showed how multimodal models respond when the image is blurred, cropped, darkened, or converted to grayscale. BERT is excluded from this test because it does not use image input.


## 7. Cross-Dataset Generalization: Fakeddit to VERITE

| Model       |   Fakeddit |   VERITE |    F1 Gap |
|:------------|-----------:|---------:|----------:|
| BERT        |   0.911917 | 0.526129 |  0.385788 |
| BLIP + BERT |   0.290221 | 0.562885 | -0.272664 |
| CLIP + MLP  |   0.490284 | 0.311213 |  0.179071 |
| ViLT        |   0.596306 | 0.128767 |  0.467539 |


This comparison evaluated all 1,001 VERITE samples with 100% coverage. BERT performed best on Fakeddit but showed a substantial drop on VERITE. BLIP Caption + BERT generalized better to VERITE, likely because VERITE focuses on image-caption consistency, which matches the caption-based architecture.


## 8. VERITE Type Analysis

| Model       | VERITE Type    |   Samples |   Accuracy |
|:------------|:---------------|----------:|-----------:|
| BERT        | miscaptioned   |       338 |  0.399408  |
| BERT        | out-of-context |       325 |  0.498462  |
| BERT        | true           |       338 |  0.5       |
| CLIP + MLP  | miscaptioned   |       338 |  0.180473  |
| CLIP + MLP  | out-of-context |       325 |  0.230769  |
| CLIP + MLP  | true           |       338 |  0.778107  |
| ViLT        | miscaptioned   |       338 |  0.0769231 |
| ViLT        | out-of-context |       325 |  0.0646154 |
| ViLT        | true           |       338 |  0.940828  |
| BLIP + BERT | miscaptioned   |       338 |  0.461538  |
| BLIP + BERT | out-of-context |       325 |  0.504615  |
| BLIP + BERT | true           |       338 |  0.544379  |


VERITE type analysis separates performance across truthful, miscaptioned, and out-of-context samples. BLIP + BERT detected out-of-context items best (0.5046 accuracy) and also performed best on miscaptioned items (0.4615). ViLT correctly classified most truthful items (0.9408) but detected very few miscaptioned (0.0769) or out-of-context (0.0646) items, indicating a strong tendency to predict the truthful class on VERITE. No model exceeded 0.51 on either misinformation type, demonstrating that cross-dataset multimodal misinformation detection remains difficult.


## 9. Modality Ablation Analysis

| Model               | Condition                  |   Samples |   Accuracy |   Balanced Accuracy |   Macro F1 |   Real Recall |   Fake Recall |   Macro F1 Drop vs Correct |
|:--------------------|:---------------------------|----------:|-----------:|--------------------:|-----------:|--------------:|--------------:|---------------------------:|
| BLIP Caption + BERT | caption_only               |       188 |   0.542553 |            0.539898 |   0.510654 |      0.789474 |     0.290323  |                -0.0369558  |
| BLIP Caption + BERT | text_only                  |       188 |   0.409574 |            0.411036 |   0.398946 |      0.273684 |     0.548387  |                 0.0747521  |
| BLIP Caption + BERT | text_plus_blank_caption    |       188 |   0.553191 |            0.549066 |   0.471274 |      0.936842 |     0.16129   |                 0.00242431 |
| BLIP Caption + BERT | text_plus_correct_caption  |       188 |   0.542553 |            0.538766 |   0.473698 |      0.894737 |     0.182796  |                 0          |
| BLIP Caption + BERT | text_plus_shuffled_caption |       188 |   0.510638 |            0.507187 |   0.450921 |      0.831579 |     0.182796  |                 0.0227773  |
| CLIP + MLP          | blank_image                |       188 |   0.670213 |            0.667233 |   0.640558 |      0.947368 |     0.387097  |                -0.00702928 |
| CLIP + MLP          | correct_image              |       188 |   0.664894 |            0.661856 |   0.633528 |      0.947368 |     0.376344  |                 0          |
| CLIP + MLP          | shuffled_image             |       188 |   0.664894 |            0.661856 |   0.633528 |      0.947368 |     0.376344  |                 0          |
| ViLT                | blank_image                |       188 |   0.510638 |            0.506961 |   0.441848 |      0.852632 |     0.16129   |                 0.288155   |
| ViLT                | correct_image              |       188 |   0.739362 |            0.737465 |   0.730004 |      0.915789 |     0.55914   |                 0          |
| ViLT                | shuffled_image             |       188 |   0.515957 |            0.510979 |   0.376144 |      0.978947 |     0.0430108 |                 0.35386    |


The modality-ablation experiment requested 200 samples and retained 188 paired successful samples for every condition and model. ViLT depended strongly on visual alignment: macro F1 fell from 0.7300 with the correct image to 0.3761 with a shuffled image and 0.4418 with a blank image. Its fake-news recall fell from 0.5591 to 0.0430 with shuffled images. In contrast, CLIP + MLP produced identical macro F1 (0.6335) for correct and shuffled images, suggesting that its classifier relied predominantly on text or failed to exploit image-text correspondence. For BLIP + BERT, caption-only input achieved macro F1 of 0.5107, compared with 0.4737 for text plus the correct caption and 0.3989 for text only. This indicates that generated captions contain useful signal, although combining them with the original text did not improve performance in this checkpoint.


## 10. Controlled Multimodal Fusion Extension

| condition        |   accuracy_mean |   accuracy_std |   balanced_accuracy_mean |   balanced_accuracy_std |   macro_f1_mean |   macro_f1_std |   real_recall_mean |   real_recall_std |   fake_recall_mean |   fake_recall_std |
|:-----------------|----------------:|---------------:|-------------------------:|------------------------:|----------------:|---------------:|-------------------:|------------------:|-------------------:|------------------:|
| image_only       |        0.795565 |     0.00120031 |                 0.792985 |              0.00204667 |        0.78881  |    0.00152369  |           0.805269 |        0.00198402 |           0.780702 |        0.00607737 |
| modality_dropout |        0.853084 |     0.00120031 |                 0.853005 |              0.0040666  |        0.848167 |    0.00188173  |           0.853379 |        0.0110466  |           0.852632 |        0.0189766  |
| normal_fusion    |        0.859321 |     0.00317573 |                 0.857247 |              0.00262461 |        0.854061 |    0.00311269  |           0.867125 |        0.00524923 |           0.847368 |        0          |
| text_only        |        0.835759 |     0          |                 0.832598 |              0.00190132 |        0.829605 |    0.000587706 |           0.847652 |        0.00715349 |           0.817544 |        0.0109561  |


This controlled extension used frozen BERT text features and frozen CLIP image features. After image-retrieval failures were excluded, all four conditions used exactly the same 1,892 training, 479 validation, and 481 test samples. Results were averaged across seeds 42, 43, and 44. Normal text-image fusion achieved the strongest mean macro F1 (0.8541, SD 0.0031), compared with 0.8296 (SD 0.0006) for text-only features and 0.7888 (SD 0.0015) for image-only features. The 0.0245 absolute macro-F1 improvement over text-only features indicates that visual features added useful information under matched experimental conditions. Modality dropout achieved macro F1 of 0.8482 (SD 0.0019): it improved on both unimodal conditions but did not outperform ordinary fusion. The proposed regularization therefore did not provide an additional benefit under this configuration.


## 11. Confidence Calibration Analysis

| Model                        |   Samples |   Brier Score |   Expected Calibration Error |   Mean Confidence |   Mean Confidence Correct |   Mean Confidence Incorrect |
|:-----------------------------|----------:|--------------:|-----------------------------:|------------------:|--------------------------:|----------------------------:|
| BERT (reproducible 20k)      |      5000 |     0.115771  |                    0.0466706 |          0.855786 |                  0.879277 |                    0.734446 |
| TF-IDF + Logistic Regression |      5000 |     0.164217  |                    0.0900404 |          0.708035 |                  0.726649 |                    0.643847 |
| BERT (four-model subset)     |       957 |     0.0708798 |                    0.0408705 |          0.940056 |                  0.952134 |                    0.816144 |
| CLIP + MLP                   |       957 |     0.240544  |                    0.138241  |          0.662964 |                  0.65702  |                    0.673702 |
| ViLT                         |       957 |     0.277194  |                    0.240688  |          0.916338 |                  0.923353 |                    0.901413 |
| BLIP Caption + BERT          |       957 |     0.250309  |                    0.0247595 |          0.528717 |                  0.528037 |                    0.529482 |


Lower Brier scores and expected calibration error indicate more reliable probabilities. The reproducible BERT model was comparatively well calibrated (Brier score 0.1158; ECE 0.0467), and confidence was lower for incorrect than correct predictions. ViLT was strongly overconfident: it produced an ECE of 0.2407 and averaged 0.9014 confidence even when wrong. CLIP + MLP was also poorly calibrated, with greater average confidence on incorrect than correct predictions. BLIP Caption + BERT obtained low ECE because its probabilities remained close to 0.5, but its Brier score of 0.2503 was approximately equivalent to an uninformative balanced binary predictor. Calibration metrics must therefore be interpreted together with discrimination and accuracy.


## 12. Structured Error-Analysis Progress

| Model               | Error direction        |   Selected errors |   Mean confidence |   High-confidence errors (>=0.90) |   Manually annotated |
|:--------------------|:-----------------------|------------------:|------------------:|----------------------------------:|---------------------:|
| BERT                | Fake predicted as Real |                13 |          0.805008 |                                 3 |                    0 |
| BERT                | Real predicted as Fake |                12 |          0.781655 |                                 3 |                    0 |
| BLIP Caption + BERT | Fake predicted as Real |                13 |          0.537093 |                                 0 |                    0 |
| BLIP Caption + BERT | Real predicted as Fake |                12 |          0.516955 |                                 0 |                    0 |
| CLIP + MLP          | Fake predicted as Real |                13 |          0.683992 |                                 0 |                    0 |
| CLIP + MLP          | Real predicted as Fake |                12 |          0.559548 |                                 0 |                    0 |
| ViLT                | Fake predicted as Real |                13 |          0.948571 |                                11 |                    0 |
| ViLT                | Real predicted as Fake |                12 |          0.891444 |                                 7 |                    0 |


A deterministic, class-aware sample of 100 incorrect predictions was created, containing 25 errors from each model. The table reports objective properties that can be calculated automatically. The qualitative causes of these errors have intentionally not been assigned automatically: the text and image must be inspected manually using the accompanying codebook. Until that review is completed, this is an annotation protocol and progress report rather than a completed qualitative finding.


## 13. Overall Conclusion

The reproducible text-only BERT baseline achieved a macro F1-score of 0.8342 on 5,000 test samples. In the matched four-model comparison on 957 successfully processed samples, BERT again performed best, while the multimodal models showed lower fake-class recall. Evaluation on all 1,001 VERITE samples demonstrated substantial domain sensitivity. Robustness testing on 480 text samples and 194 image samples further showed that stability depends on both the model and perturbation type. The paired modality ablation provides direct evidence that ViLT uses image-text alignment, whereas CLIP + MLP showed little sensitivity to image identity. In a separate controlled three-seed experiment, normal fusion of frozen BERT and CLIP features improved mean macro F1 from 0.8296 for text-only features to 0.8541, while modality dropout did not improve on normal fusion. Calibration analysis further showed that high model confidence does not necessarily imply reliability, particularly for ViLT and CLIP + MLP.


## 14. Evidence Limitations

The four-model comparison processed 957 of 1,000 requested samples, text robustness processed 480 of 500, and visual robustness processed 194 of 200 because some remote images were unavailable. Although coverage exceeded 95% in each case, failed image retrieval may introduce selection bias. The models were originally trained with different sample budgets, so performance differences should not be attributed only to model architecture. The controlled fusion extension retained 1,892 of 2,000 requested training samples (94.6%), 479 of 500 validation samples (95.8%), and 481 of 500 test samples (96.2%). It used frozen encoders and a moderate test sample, so its results support a controlled comparison of feature fusion but do not establish that the fusion architecture is universally superior.
