# Minuta de Seleção da Abordagem de Maior Potencial

Esta minuta organiza o MAE OOF Pós-Seleção e evidências descritivas para revisão humana. A recomendação abaixo não constitui uma decisão automática.

As métricas globais são calculadas diretamente sobre as 132 Predições OOF reunidas; não são médias simples dos folds.

## Quatro abordagens candidatas

| Abordagem | MAE (kg) | RMSE (kg) | Viés (kg) | R² |
|---|---:|---:|---:|---:|
| Random Forest | 53.33 | 73.52 | 4.03 | 0.756 |
| Rede Densa por Feições | 53.33 | 76.33 | -2.14 | 0.737 |
| CNN compacta | 66.07 | 88.83 | -9.56 | 0.644 |
| ResNet-18 pré-treinada | 73.57 | 96.66 | 33.49 | 0.578 |

## Evidência descritiva nos extremos

| Abordagem | B1 MAE (kg) | B1 viés (kg) | B10 MAE (kg) | B10 viés (kg) |
|---|---:|---:|---:|---:|
| Random Forest | 31.71 | 31.71 | 126.92 | -126.92 |
| Rede Densa por Feições | 34.14 | 23.39 | 133.35 | -133.35 |
| CNN compacta | 21.66 | 15.17 | 177.27 | -177.27 |
| ResNet-18 pré-treinada | 26.12 | 13.44 | 64.47 | -17.47 |

## Referência

O preditor da média do treino de cada fold obteve MAE de 120.84 kg. Ele é uma referência trivial e não uma quinta candidata.

## Recomendação revisável

Pelo critério principal predefinido, `Rede Densa por Feições` apresenta o menor MAE OOF Pós-Seleção (53.33 kg) e deve ser priorizada na revisão. RMSE, viés, R² e os extremos permanecem evidências descritivas; não há pontuação combinada, bootstrap ou custo computacional como critério.

## Limitações

As mesmas 132 máscaras orientaram seleção de features e comparação. Estes valores são evidência de desenvolvimento, não validação independente em animais novos. B10 está confundida com fazenda e aquisição na amostra atual.

## Registro de revisão humana

- Status: revisado
- Interpretações aceitas, corrigidas ou rejeitadas: foram aceitas as evidências comparativas OOF. Confirmou-se humanamente a abordagem `random_forest` (Random Forest Baseline, `random_forest_baseline`) como a Abordagem de Maior Potencial. Embora empatada em MAE (53,33 kg) com a Rede Densa, a Random Forest apresentou menor RMSE (73,52 kg vs 76,33 kg), maior R² (0,756 vs 0,737) e excelente estabilidade estrutural sob o Conjunto Compartilhado de Features de 25 feições.
- Abordagem confirmada: `random_forest` (`random_forest_baseline`).
- Orçamento de ajuste fino: no máximo 3 variações adicionais pré-registradas.
- Limitação aceita: decisão baseada em evidência de desenvolvimento da Divisão Estratificada Canônica; B10 confundida com a fazenda Faco.
- Revisor: Victor Alexandre Saraiva Pimentel
- Revisado em: 2026-08-06
- Decisão auditável: https://github.com/Victor-Saraiva-P/buffalo-weight-pred/issues/22#issuecomment-5208834642
