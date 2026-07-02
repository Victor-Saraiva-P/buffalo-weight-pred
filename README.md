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

`configs/shared.yaml` define dados, Índice de Features, Divisão Estratificada e diretório de treino compartilhados.

`configs/classical_models.yaml` define Configurações de Modelo para Modelos Clássicos de Predição e as colunas de features usadas por eles.

`configs/cnn_mask_models.yaml` define Configurações de Modelo para Modelos de Predição por Máscara. Essas configurações usam pixels da Máscara Binarizada diretamente e não usam `feature_columns`.

## Arquivos Gerados

`generated/` deve conter artefatos derivados de `data/`, como o Índice de Features e a Divisão Estratificada.

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

Treinar modelos usando kfold estratificado:

```bash
PYTHON=.venv/bin/python make train
```

`make train` valida os artefatos já gerados, treina primeiro os Modelos Clássicos de Predição, depois os Modelos de Predição por Máscara, e por fim recria `model_comparison.csv`/`model_comparison.png` em `generated/train/`.

O treino gera um diretório por Configuração de Modelo em `generated/train/`. Cada diretório inclui `predicted_vs_actual.png`, que marca os maiores erros no gráfico de peso real vs predito.

Também é possível testar um Modelo de Predição por Máscara com `model: cnn_mask`. Esse modelo usa a Máscara Binarizada diretamente e requer PyTorch instalado no ambiente. Exemplo de Configuração de Modelo:

```yaml
cnn_mask_baseline:
  model: cnn_mask
  params:
    epochs: 80
    batch_size: 16
    learning_rate: 0.001
    image_size: 128
    random_state: 42
    weight_decay: 0.0001
    patience: 10
    augment: true
```

Também é possível rodar cada família diretamente:

```bash
PYTHONPATH=src .venv/bin/python -m buffalo_weight.train_classical --shared-config configs/shared.yaml --models-config configs/classical_models.yaml
PYTHONPATH=src .venv/bin/python -m buffalo_weight.train_cnn_mask --shared-config configs/shared.yaml --models-config configs/cnn_mask_models.yaml
```

Antes do treino, os arquivos gerados devem estar consistentes com o Índice de Máscaras: o Índice de Features, a Divisão Estratificada e `data/conjunto-de-mascaras/` devem conter exatamente os mesmos nomes de arquivo definidos em `data/indice.xlsx`. Máscaras extras, faltantes, duplicadas por extensão ou com valores diferentes de `0/255` são rejeitadas.

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
