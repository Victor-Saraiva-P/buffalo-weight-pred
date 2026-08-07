# Registro de Experimentos Encerrados

Este documento constitui o inventário científico conciso das hipóteses e abordagens de modelagem descartadas durante o desenvolvimento do projeto. Cada entrada registra o nome da hipótese, uma breve descrição da abordagem explorada e o motivo objetivo do seu encerramento, servindo como memória científica sem integrar as evidências do relatório final.

---

## 1. PCA / Fusão de Feições (`pca_feature_fusion`, `dual_pca`)

- **Descrição curta**: Compressão de imagens de máscara via Análise de Componentes Principais (PCA) concatenada com descritores geométricos tabulares para entrada em regressores clássicos e ensembles.
- **Motivo objetivo**: Subestimou o peso vivo nos animais de peso extremo e apresentou desempenho OOF (MAE) inferior ao Random Forest direto e aos modelos convolucionais sob a Divisão Estratificada Canônica.

---

## 2. Modelos Híbridos & Embeddings Pré-treinados (`pretrained_mask_embedding`)

- **Descrição curta**: Extração de vetores de características profundas a partir de modelos pré-treinados (MobileNetV3 / EfficientNet) congelados ou parcialmente ajustados, combinados com vetores geométricos.
- **Motivo objetivo**: Alto sobreajuste (overfitting) no tamanho reduzido da amostra e maior variância entre dobras do que a ResNet-18 treinada sob a entrada convolucional direta.

---

## 3. Canais Derivados & Geometria Pura (`cnn_geometry_channels`, `pure_geometry`)

- **Descrição curta**: Inclusão de canais espaciais derivados (contornos, transformadas de distância) na entrada de redes CNN ou modelos treinados exclusivamente sobre descritores de contorno sem invariância canônica.
- **Motivo objetivo**: Demonstrou redundância estrutural com as feições geométricas já calculadas e não apresentou ganho preditivo relevante em relação à representação binarizada na Escala Canônica da Máscara.

---

## 4. Calibração Pós-hoc de Predições (`prediction_calibration`)

- **Descrição curta**: Modelos de pós-processamento e calibração linear/não-linear aplicados sobre as predições OOF para tentar corrigir viés em faixas extremas de peso.
- **Motivo objetivo**: Não reduziu o erro médio absoluto no protocolo de validação cruzada sem aumentar a distorção em instâncias fora da amostra; superado pelo ajuste fino controlado de hiperparâmetros.

---

## 5. Ensembles e Integrações Multimodelo (`ensemble_oof`)

- **Descrição curta**: Combinação por média ponderada ou metamodelos (stacking) das predições de múltiplos estimadores (fusão, regressão linear, árvores).
- **Motivo objetivo**: Adicionou complexidade significativa e acoplamento ao pipeline sem produzir melhoria estatisticamente relevante (superando a Equivalência Prática de 1 kg) frente à Abordagem de Maior Potencial isolada.

---

## 6. Algoritmos Clássicos Fora do Protocolo (`extra_trees`, `hist_gradient_boosting`, `xgboost`, `pca_svr_mask`)

- **Descrição curta**: Avaliação exploratória de algoritmos alternativos de aprendizado de máquina (Extra Trees, HistGradientBoosting, XGBoost, PCA-SVR) sobre o espaço de feições ou pixels.
- **Motivo objetivo**: Apresentaram estabilidade inferior ou MAE OOF consistentemente pior do que o Random Forest Baseline no Conjunto Compartilhado de Features.

---

## 7. Transformações de Alvo e Reponderação de Extremos (`target_transform`, `heavy_weighting`)

- **Descrição curta**: Aplicação de transformações logarítmicas, raiz cúbica ou reponderação da função de perda com base nas faixas quantílicas de peso.
- **Motivo objetivo**: Pioraram o erro quadrático e o erro absoluto médio (MAE) na escala original em quilogramas (kg) sem estabilizar os erros percentuais relativos.
