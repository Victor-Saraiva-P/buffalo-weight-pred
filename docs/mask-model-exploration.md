# Exploração de modelos a partir da máscara binária

Todos os modelos desta rodada recebem somente a máscara binária ou descritores calculados diretamente dela. O protocolo mantém os mesmos cinco folds do restante do projeto.

## Resultados no split fixo

| Abordagem | MAE OOF (kg) |
|---|---:|
| Fusão geometria + PCA, stretch, folha 2 | 52,11 |
| Fusão geometria + PCA, letterbox, folha 1 | 52,46 |
| Fusão original, letterbox, folha 2 | 53,52 |
| Pixels + PCA + Ridge | 59,15 |
| Perfis da silhueta + ExtraTrees | 59,66 |
| Embedding MobileNet + PCA + Ridge | 62,99 |
| Embedding ResNet-18 + PCA + Ridge | 75,31 |

O melhor ensemble equal-weight combinou a fusão `letterbox/leaf1`, HistGradientBoosting e ResNet-18 ajustada, com MAE OOF de 51,38 kg. Como a combinação foi escolhida no mesmo OOF usado para comparação, esse valor serve para priorização exploratória, não como estimativa final de generalização.

## Estabilidade

Em cinco seeds, as duas melhores variantes apresentaram:

| Configuração | MAE médio (kg) | Desvio entre seeds |
|---|---:|---:|
| Fusão stretch, folha 2 | 53,65 | 1,57 |
| Fusão letterbox, folha 1 | 53,78 | 1,19 |

O ganho do `stretch` no split fixo diminui entre seeds. As duas variantes devem continuar como candidatas até uma avaliação confirmatória.

## Leitura exploratória

- Descritores geométricos globais continuam sendo o sinal individual mais forte.
- PCA dos pixels adiciona informação complementar quando fundida à geometria.
- Aumentar PCA de 24 para 32 ou 48 componentes piorou o resultado.
- Embeddings ImageNet congelados não transferiram bem para silhuetas binárias sem ajuste supervisionado.
- Perfis de borda são úteis, mas não superaram a fusão entre geometria e pixels comprimidos.

## Reprodução

```bash
make train-mask-experiments
make train-fusion-experiments
make ensemble
```

Artefatos principais:

- `generated/mask_classical/model_comparison.csv`
- `generated/ensemble/model_comparison.csv`
- `generated/stability_fusion_candidates/model_comparison.csv`

## Segunda rodada dirigida

Foram adicionadas oito features derivadas da máscara: dois proxies de volume, três medidas regionais de espessura, razão entre espessuras e posição relativa do centroide. Também foram avaliadas transformações `log(peso)` e `peso^(1/3)`, além de CNNs com três canais derivados da máscara: preenchimento, contorno e distância interna normalizada.

| Abordagem | MAE OOF (kg) |
|---|---:|
| Fusão original + `log(peso)` | 51,42 |
| Fusão original + raiz cúbica | 51,49 |
| Fusão alométrica + raiz cúbica | 53,20 |
| ResNet-18 com máscara/contorno/distância | 53,92 |
| ExtraTrees alométrico + `log(peso)` | 54,48 |

As features alométricas adicionais não melhoraram a fusão. A transformação do alvo, isolada sobre as 16 features originais, foi a melhoria mais consistente. Em cinco seeds de divisão, a fusão com `log(peso)` obteve `52,20 ± 1,19 kg`; a variante com raiz cúbica obteve `52,33 ± 1,45 kg`.

O ensemble entre seeds do mesmo modelo não trouxe ganho. O melhor ensemble de arquiteturas diferentes combinou fusões com `log` e raiz cúbica com a ResNet geométrica, chegando a MAE OOF de 50,46 kg. Esse valor continua exploratório por ter sido selecionado no mesmo conjunto OOF.

Artefatos da segunda rodada:

- `generated/allometric/model_comparison.csv`
- `generated/ensemble/directed_comparison.csv`
- `generated/ensemble/fusion_seed_comparison.csv`
- `generated/stability_target_transform/model_comparison.csv`

## Tuning dirigido e máscara canônica

A busca dirigida variou resolução, componentes PCA, tamanho mínimo das folhas, fração de features e potência do alvo. A redução de 128 para 96 pixels foi a melhor alteração no split fixo; reduzir `max_features` consistentemente piorou o modelo.

Também foi implementado um segundo ramo de forma. A máscara é pré-reduzida, alinhada pelo eixo principal, recortada e redimensionada antes de um PCA próprio. O ramo original continua preservando enquadramento e escala.

| Configuração | MAE OOF fixo (kg) | MAE em 5 seeds (kg) |
|---|---:|---:|
| Fusão `log`, 128 px | 51,42 | 52,20 ± 1,19 |
| Fusão `log`, 96 px | 50,90 | 52,08 ± 1,34 |
| Fusão canônica de dois ramos | 50,90 | **51,65 ± 0,85** |

A fusão canônica apresentou o melhor resultado estável e menor variação entre seeds. O ensemble equal-weight entre fusão canônica, fusão 96 px e ResNet geométrica atingiu 50,06 kg no OOF fixo. Como os componentes foram selecionados nesse mesmo OOF, o modelo individual canônico é a referência mais confiável desta etapa.

Artefatos:

- `generated/tuning/model_comparison.csv`
- `generated/canonical/model_comparison.csv`
- `generated/stability_fusion_tuning/model_comparison.csv`
- `generated/ensemble/final_tuning_comparison.csv`
