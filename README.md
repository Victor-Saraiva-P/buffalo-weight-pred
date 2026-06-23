# buffalo-weight-pred

Base Python para treinamento de modelo de predição de peso de búfalos a partir de máscaras binarizadas.

## Dados

`data/` contém dois itens esperados:

- `indice.xlsx`: índice de máscaras com nome do arquivo, fazenda, peso e tag de uso.
- `conjunto-de-mascaras/`: máscaras binarizadas usadas no treinamento.

## Máscaras Binarizadas

O treinamento usa máscaras geradas pelo modelo de segmentação `birefnet-general` com método de binarização `LimiarFixoBaixa`.

Somente máscaras presentes no `indice.xlsx` devem ser usadas no treinamento.

## Configuração

Cada experimento deve ter sua própria configuração em `configs/`.

O primeiro experimento usa `configs/baseline.yaml`. Essa configuração define os caminhos dos dados, coluna-alvo e colunas do arquivo gerado com as features calculadas.

Essa separação permite gerar bases futuras com outras features sem misturar decisões entre experimentos.

## Arquivos Gerados

`generated/` deve conter artefatos derivados de `data/`, como índices de features calculadas a partir das máscaras binarizadas.

Esses arquivos podem ser recriados a partir de `data/` e das configurações em `configs/`.

## Comandos

Criar ambiente e instalar dependências:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Gerar índice de features:

```bash
PYTHON=.venv/bin/python make features
```

Gerar divisão estratificada e gráfico de categorias de peso:

```bash
PYTHON=.venv/bin/python make split
```

Treinar baseline Random Forest com kfold estratificado:

```bash
PYTHON=.venv/bin/python make train
```

Medir estabilidade da divisão estratificada entre diferentes `split.random_state`:

```bash
PYTHON=.venv/bin/python make stability
```

Esse diagnóstico gera arquivos em `generated/diagnostics/`:

- `split_stability_fold_metrics.csv`: métricas por seed e fold.
- `split_stability_seed_summary.csv`: média, desvio, mínimo e máximo de MAE por seed.
- `split_stability_overall.csv`: variação do MAE médio entre seeds.
- `split_stability_hard_examples.csv`: máscaras com maior erro absoluto médio quando aparecem em validação.
- `split_stability_seed_mae.png`: gráfico do MAE médio por seed com faixa mínimo-máximo entre folds.
- `split_stability_fold_mae.png`: gráfico do MAE de cada fold em cada seed.
- `split_stability_hard_examples.png`: gráfico das máscaras com maior erro absoluto médio.

Rodar testes:

```bash
PYTHON=.venv/bin/python make test
```
