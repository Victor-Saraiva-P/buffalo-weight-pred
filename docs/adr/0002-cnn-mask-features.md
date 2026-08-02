# Future hybrid mask and feature CNN

We want to evaluate a future `cnn_mask_features` Configuração de Modelo that combines the Máscara Binarizada pixels with geometric features from the Índice de Features. The goal is to compare three prediction strategies in the Etapa de Predição de Peso: Modelo Clássico de Predição from features only, Modelo de Predição por Máscara from pixels only, and a hybrid model that uses both signals.

The expected architecture is late fusion: the mask branch uses a CNN to produce a visual embedding, the feature branch uses a small MLP to produce a tabular embedding, then both embeddings are concatenated before the final regression head. This keeps the pixel path able to learn shape patterns while giving the model direct access to explicit geometric measurements such as area, bounding box dimensions, convex area, extent, and axis lengths.

This may help because the dataset is small and many useful geometric relationships are already known and cheaply computed from each Máscara Binarizada. Explicit features can act as stable inductive bias, while the CNN may still capture shape details that the handcrafted features miss.

We will not implement `cnn_mask_features` yet. The current `cnn_mask` support still needs cleanup because it was added inside a training pipeline originally shaped around feature arrays. In the current code, `training.feature_columns` and `features_index_path` are still present in the CNN config even though `cnn_mask` reads mask pixels directly and ignores the feature array. Building a hybrid model before clarifying this separation would make the model boundary confusing.

Before adding `cnn_mask_features`, we should first polish the CNN foundation:

1. Make `cnn_mask` clearly represent a pure Modelo de Predição por Máscara.
2. Avoid requiring `training.feature_columns` for pure mask models.
3. Clarify which inputs each model family consumes: Indice de Features, Mascara Binarizada pixels, or both.
4. Keep config validation aligned with those input contracts.
5. Add tests around mask loading, binary-mask validation, and CNN-specific training flow.

After that cleanup, `cnn_mask_features` can be introduced as a separate hybrid model instead of overloading `cnn_mask`. This keeps comparisons interpretable: `cnn_mask` means pixels only, `cnn_mask_features` means pixels plus geometric features.
