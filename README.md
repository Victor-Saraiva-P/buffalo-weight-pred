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

Treinar baseline com Random Forest e XGBoost usando kfold estratificado:

```bash
PYTHON=.venv/bin/python make train
```

O treino gera arquivos em `generated/train/`, com um diretório por Configuração de Modelo e `model_comparison.csv`/`model_comparison.png` na raiz para comparação geral.

Medir estabilidade da divisão estratificada entre diferentes `split.random_state`:

```bash
PYTHON=.venv/bin/python make stability
```

Esse diagnóstico gera arquivos em `generated/stability/`:

- `<configuração_de_modelo>/fold_metrics.csv`: métricas por seed e fold.
- `<configuração_de_modelo>/seed_summary.csv`: média, desvio, mínimo e máximo de MAE por seed.
- `<configuração_de_modelo>/overall.csv`: variação do MAE médio entre seeds.
- `<configuração_de_modelo>/hard_examples.csv`: máscaras com maior erro absoluto médio quando aparecem em validação.
- `<configuração_de_modelo>/seed_mae.png`: gráfico do MAE médio por seed com faixa mínimo-máximo entre folds.
- `<configuração_de_modelo>/fold_mae.png`: gráfico do MAE de cada fold em cada seed.
- `<configuração_de_modelo>/hard_examples.png`: gráfico das máscaras com maior erro absoluto médio.
- `model_comparison.csv` e `model_comparison.png`: comparação entre Configurações de Modelo.

Comparar granularidades de `Categoria de Peso` na divisão estratificada:

```bash
PYTHON=.venv/bin/python make compare-categories
```

Por padrão, esse diagnóstico compara 4, 6 e 8 categorias com 30 seeds. Use `CATEGORY_COUNTS` para explorar mais granularidades. Ele gera arquivos em `generated/compare-categories/`:

- `<configuração_de_modelo>/overall.csv`: resumo por quantidade de categorias.
- `<configuração_de_modelo>/fold_metrics.csv`: métricas por quantidade de categorias, seed e fold.
- `<configuração_de_modelo>/mae.png`: MAE médio por quantidade de categorias.
- `<configuração_de_modelo>/seed_variation.png`: variação do MAE entre seeds por quantidade de categorias.
- `split_balance.csv`: distribuição de validação por categoria, seed e fold.
- `weight_scatter_c4.png`, `c6`, `c8`: distribuição de pesos por fold.
- `weight_heatmap_c4.png`, `c6`, `c8`: contagem por fold e categoria.
- `model_comparison.csv` e `model_comparison.png`: comparação entre Configurações de Modelo.

Para rodadas rápidas, ajuste as variáveis do Makefile:

```bash
SEED_COUNT=2 PYTHON=.venv/bin/python make compare-categories
```

A configuração baseline usa 10 `Categorias de Peso`; veja `docs/weight-category-count.md` para a justificativa.

Rodar testes:

```bash
PYTHON=.venv/bin/python make test
```
