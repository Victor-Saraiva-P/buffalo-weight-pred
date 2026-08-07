# Relatório de Diagnóstico: Curvas de Aprendizado Controladas

Total de pontos de avaliação processados: 60
Total de configurações avaliadas: 4

## Resumo das Curvas de Aprendizado (MAE OOF Médio em kg)

| Configuração | 50% Treino | 75% Treino | 100% Treino |
| :--- | :---: | :---: | :---: |
| compact_cnn_baseline | 70.81 | 65.01 | 66.22 |
| dense_baseline | 61.93 | 57.73 | 53.42 |
| random_forest_baseline | 52.87 | 53.15 | 53.50 |
| resnet18_pretrained_partial | 73.33 | 59.06 | 73.62 |

Nota: Subconjuntos de 50% e 75% foram gerados de forma aninhada, estratificada e determinística com seed 45.
Os pontos de 100% reutilizam os artefatos existentes apenas quando a proveniência é válida e atual.
