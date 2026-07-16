# Investigação empírica do limite de MAE

Esta análise investiga por que os modelos atuais não chegam perto de MAE zero usando somente as 132 máscaras binárias disponíveis. As conclusões abaixo se limitam a testes executáveis com o conjunto atual; fatores não observados não são apresentados como causas confirmadas.

## Resultado de referência

O modelo individual `dual_pca24_canonical16` obteve MAE OOF de 50,88 kg, com IC95% bootstrap por animal de 42,31–59,65 kg. A melhoria de 0,54 kg sobre a fusão anterior não é conclusiva: o IC95% pareado da diferença é −2,28 a +1,26 kg.

## Quantidade de dados

A learning curve repetida produziu:

| Amostras de treino | MAE treino | MAE validação | MAE dos 20% mais pesados |
|---:|---:|---:|---:|
| 26,0 | 9,85 | 61,57 | 138,50 |
| 52,6 | 7,98 | 57,36 | 125,87 |
| 78,6 | 7,26 | 55,20 | 118,71 |
| 105,6 | 6,85 | 51,80 | 107,40 |

De 75% para 100% do treino, o ganho pareado foi 3,40 kg, IC95% 1,86–4,99 kg, com melhora em 84% das combinações seed/fold. Portanto, a curva ainda não atingiu plateau: mais exemplos têm efeito mensurável, principalmente no extremo pesado.

Com `min_samples_leaf=1`, o ExtraTrees atingiu MAE de treino praticamente zero (`3,2e-12 kg`) e MAE de validação de 51,63 kg. Isso demonstra que a família tem capacidade para memorizar os dados; o limite atual é generalização, não incapacidade de ajuste do estimador.

## Extremos de peso

A faixa B10 contém somente 13 animais. Seu MAE é 146,11 kg e o viés é −146,11 kg: todos são subestimados. A faixa B9 também apresenta MAE de 69,42 kg e viés de −67,77 kg. O quintil mais pesado tem MAE de 107,76 kg, contra 21,53 kg no quintil mais leve.

Ponderar os 20% mais pesados reduziu o MAE de B10 de 146,11 para aproximadamente 139 kg, mas aumentou o MAE global de 50,88 para 51,25–54,06 kg. Calibração linear cross-fitted reduziu B10 para 122,82 kg e removeu o viés global, mas aumentou o MAE global para 51,93 kg. Com os exemplos atuais, corrigir o extremo desloca erro para os demais grupos.

## Informação disponível na máscara

No espaço completo usado pelo modelo, incluindo geometria, PCA da máscara original e PCA canônica, o vizinho mais próximo apresenta diferença mediana de peso de 52,5 kg e P90 de 202,6 kg. Na representação canônica existem pares com IoU de 0,765 e diferença de 507 kg.

Não foram encontrados nomes, máscaras em pixels ou vetores completos de features duplicados. Também não existem pares canônicos com IoU ≥ 0,95; o máximo observado foi 0,856. Logo, a ambiguidade observada não é causada por duplicatas exatas.

## Escala e aquisição

Uma ablação OOF com ExtraTrees obteve:

| Representação | MAE |
|---|---:|
| Escala + forma | 55,36 kg |
| Somente área | 68,46 kg |
| Somente forma invariante | 71,51 kg |

O modelo precisa simultaneamente de escala aparente e forma. As features classificam a resolução de aquisição com balanced accuracy de 0,970, evidenciando que carregam informação do protocolo/câmera. Transferência entre fazendas, restrita ao suporte comum de peso, apresentou MAE de 62,41 e 72,02 kg.

A fazenda é classificada com balanced accuracy de 0,932 usando features, mas somente o peso já alcança 0,921. No suporte comum, usando apenas forma, a balanced accuracy cai para 0,495, equivalente ao acaso. Assim, a associação fazenda-peso é comprovada; um estilo visual residual de fazenda não foi detectado nesse controle.

## Qualidade e robustez da segmentação

Entre as 132 máscaras, 6 possuem múltiplos componentes, 5 tocam a borda e 73 contêm algum buraco. Componentes, buracos e contato com borda não apresentaram associação significativa com erro OOF. `foreground_ratio` apresentou Spearman 0,316, p=0,00022, mas também representa escala aparente e peso, não apenas qualidade.

Perturbações sintéticas mostraram alta sensibilidade a mudanças de área:

| Cenário | MAE | Mudança média da predição |
|---|---:|---:|
| Original | 51,08 | 0,00 kg |
| Erosão 1 px | 63,95 | 38,46 kg |
| Erosão 2 px | 86,77 | 65,95 kg |
| Dilatação 1 px | 66,99 | 41,64 kg |
| Translação 4 px | 51,49 | 5,16 kg |
| Rotação 5° | 50,33 | 7,20 kg |

Isso comprova sensibilidade ao contorno/área. Não comprova que as máscaras reais estejam erradas, pois não há máscaras manuais de referência.

## Limite compartilhado pelos modelos

As correlações dos erros OOF variam aproximadamente de 0,70 a 0,998. As três fusões principais ficam acima de 0,987, indicando que repetem quase os mesmos erros. Um oracle que escolhe o melhor dos seis modelos por animal ainda tem MAE de 28,58 kg; há complementaridade, mas nenhuma combinação atual contém informação suficiente para chegar perto de zero.

## Conclusão baseada nos testes

Os dados sustentam quatro fatores para o erro atual:

1. Quantidade insuficiente para generalização: learning curve ainda descendente e grande gap treino-validação.
2. Falta crítica de cobertura dos pesos altos: somente 13 animais em B10 e erro sistemático de 146 kg.
3. Ambiguidade da entrada: representações muito próximas podem corresponder a diferenças grandes de peso.
4. Dependência de escala aparente e do protocolo de aquisição, sem calibração física observada.

Não há evidência de que duplicatas exatas ou defeitos morfológicos grosseiros sejam a causa principal. Ruído de pesagem, qualidade real da segmentação, repetição do mesmo animal e generalização prospectiva não podem ser validados com os artefatos atuais.

## Artefatos

- `generated/diagnostics/report.md`
- `generated/diagnostics/plots/learning_curve.png`
- `generated/diagnostics/plots/residuals.png`
- `generated/diagnostics/plots/mae_by_weight_category.png`
- `generated/diagnostics/plots/error_correlation.png`
- `generated/diagnostics/plots/mask_quality_vs_error.png`
- `generated/diagnostics/plots/nearest_visual_pairs.png`
- CSVs correspondentes em `generated/diagnostics/`
