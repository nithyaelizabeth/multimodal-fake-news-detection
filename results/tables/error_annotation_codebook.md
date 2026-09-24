# Error Annotation Codebook

Annotate 100 errors without looking at aggregate model results.
Use one primary category and, when necessary, one secondary category.

## Error categories

- `ambiguous_or_insufficient_text`
- `satire_or_humour`
- `misleading_caption`
- `out_of_context_image`
- `incorrect_generated_caption`
- `named_entity_or_event_confusion`
- `external_knowledge_required`
- `dataset_label_ambiguity`
- `image_quality_or_retrieval_problem`
- `real_class_prediction_bias`
- `other`

## Controlled fields

- `image_helpful`: yes, no, unclear, or inaccessible
- `caption_quality`: correct, partially_correct, incorrect, irrelevant, or not_applicable
- `label_ambiguous`: yes, no, or unclear

## Annotation guidance

Base the category on the likely reason for the model error, not merely on the
predicted class. Keep explanations concise and evidence-based. If an image
cannot be opened, mark it inaccessible rather than guessing its content.