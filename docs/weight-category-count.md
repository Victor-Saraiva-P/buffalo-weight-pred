# Quantidade de Categorias de Peso

## Decisão

O baseline usa `10` Categorias de Peso na Divisão Estratificada.

## Motivo

A comparação de granularidade testou `4`, `6`, `8`, `10`, `12` e `16` Categorias de Peso com 30 seeds de divisão. O objetivo não foi otimizar diretamente o MAE de validação k-fold, mas encontrar uma divisão mais bem distribuída entre faixas de peso sem deixar cada categoria pequena demais dentro de cada fold.

Resultado resumido:

| Categorias | MAE médio | Desvio entre seeds | Range entre seeds | Desvio entre folds | Pior fold | Exemplos por categoria/fold |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 55.827 | 1.960 | 7.817 | 10.142 | 82.491 | 6-7 |
| 6 | 55.790 | 2.079 | 10.431 | 9.872 | 86.106 | 4-5 |
| 8 | 55.851 | 1.882 | 7.183 | 8.262 | 77.062 | 3-4 |
| 10 | 55.651 | 1.469 | 6.034 | 9.342 | 79.329 | 2-3 |
| 12 | 55.881 | 1.640 | 5.634 | 8.133 | 79.434 | 2-3 |
| 16 | 56.418 | 1.981 | 8.013 | 9.731 | 80.199 | 1-2 |

`10` foi escolhido como ponto de equilíbrio:

- Menor MAE médio observado na comparação.
- Menor desvio entre seeds entre as opções testadas.
- Mantém 2-3 exemplos por Categoria de Peso em cada fold.
- Evita o limite frágil observado em `16`, onde algumas categorias ficam com 1-2 exemplos por fold.

## Interpretação

`8`, `10` e `12` formam a faixa saudável para o dataset atual. `10` é o sweet spot operacional: melhora a estabilidade entre seeds sem empurrar a granularidade até uma zona frágil.

Com o tamanho atual do dataset, não devemos passar de `12` Categorias de Peso sem nova evidência. `16` já indica perda de robustez da Divisão Estratificada.

## Comando de Referência

```bash
CATEGORY_COUNTS=4,6,8,10,12,16 SEED_COUNT=30 make compare-categories
```
