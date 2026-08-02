# Registro de Experimentos Encerrados

Este registro preserva hipóteses descartadas sem manter suas implementações como parte do Núcleo Reprodutível do Relatório.

## Fusão tardia entre CNN e geometria pura

- **Hipótese**: combinar uma representação aprendida da máscara com dez descritores geométricos melhoraria a predição de peso.
- **Abordagem**: concatenar embeddings da ResNet-18 e das features geométricas antes da regressão final.
- **Encerramento**: a combinação não melhorou a CNN equivalente que usa somente canais derivados da máscara.

## Modelo Híbrido PCA Canônico

- **Hipótese**: combinar escala aparente, forma alinhada e descritores geométricos reduziria o erro de predição.
- **Abordagem**: aplicar PCA à máscara original e à máscara canônica, concatenar 16 features calculadas automaticamente e ajustar um ExtraTrees.
- **Encerramento**: a vantagem nominal sobre a ResNet-18 não foi conclusiva e não justificou ampliar o contrato de processamento e interpretação do método final.
