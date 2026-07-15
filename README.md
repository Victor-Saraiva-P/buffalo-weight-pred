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

`configs/cnn_mask_models.yaml` define Configurações de Modelo para Modelos de Predição por Máscara. Essas configurações usam pixels da Máscara Binarizada diretamente e não usam `feature_columns`. O parâmetro `resize_mode` permite comparar `letterbox`, que preserva a proporção original, com `stretch`, que ocupa todo o quadro quadrado.

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

`make train` valida os artefatos já gerados, treina primeiro os Modelos Clássicos de Predição pendentes, depois os Modelos de Predição por Máscara pendentes, e por fim recria `model_comparison.csv`/`model_comparison.png` em `generated/train/`. Uma configuração é considerada concluída quando seu diretório contém `fold_metrics.csv` e `predictions.csv`.

Por padrão, `configs/cnn_mask_models.yaml` reúne todas as configurações de máscara avaliadas no projeto, incluindo ablações da CNN, PCA+SVR, MobileNetV3, EfficientNet-B0 e ResNet-18. Os arquivos de experimento menores continuam disponíveis para executar somente subconjuntos específicos.

Apagar todo o cache de treinamento para forçar uma nova execução de todos os modelos:

```bash
make clean
```

Apagar somente uma ou mais configurações específicas:

```bash
make clean MODELS="resnet18_pretrained_last_block mobilenet_v3_pretrained_last_block"
```

Também é possível separar os nomes por vírgula. A limpeza seletiva remove os diretórios indicados e invalida `model_comparison.csv`/`model_comparison.png`. Como o cache é identificado pelo nome da configuração, alterações nos parâmetros de uma configuração existente exigem sua limpeza manual antes de `make train`.

O treino gera um diretório por Configuração de Modelo em `generated/train/`. Cada diretório inclui `predicted_vs_actual.png`, que marca os maiores erros no gráfico de peso real vs predito.

Também é possível testar um Modelo de Predição por Máscara com `model: cnn_mask`. Esse modelo usa a Máscara Binarizada diretamente e requer PyTorch instalado no ambiente. Exemplo de Configuração de Modelo:

```yaml
cnn_mask_baseline:
  model: cnn_mask
  params:
    architecture: baseline
    epochs: 80
    batch_size: 16
    learning_rate: 0.001
    image_size: 128
    resize_mode: letterbox
    random_state: 42
    weight_decay: 0.0001
    patience: 10
    augment: true
```

`architecture` aceita `baseline`, `residual`, `mobilenet_v3_small`, `efficientnet_b0` ou `resnet18`. O experimento isolado de resolução e arquitetura está em `configs/cnn_mask_next_experiment.yaml` e pode ser executado sem alterar a configuração da ablação anterior:

```bash
CNN_MASK_MODELS_CONFIG=configs/cnn_mask_next_experiment.yaml make train
```

`architecture: mobilenet_v3_small` usa a MobileNetV3 Small do `torchvision`. A máscara é repetida nos três canais e normalizada como entrada ImageNet, mas nenhum dado RGB é adicionado. O experimento compara pesos pré-treinados com a cabeça congelada, fine-tuning do último bloco e um controle com pesos aleatórios:

```bash
CNN_MASK_MODELS_CONFIG=configs/cnn_mask_transfer_experiment.yaml make train
```

O comparativo seguinte mantém o fine-tuning do último estágio e avalia MobileNetV3 Small, EfficientNet-B0 e ResNet-18 sob o mesmo protocolo:

```bash
CNN_MASK_MODELS_CONFIG=configs/cnn_mask_pretrained_models_experiment.yaml make train
```

O modelo `pca_svr_mask` oferece um baseline não neural que aplica PCA diretamente aos pixels binários e usa SVR para regressão. A PCA e o SVR são ajustados novamente dentro do treino de cada fold:

```yaml
pca_svr_mask_baseline:
  model: pca_svr_mask
  params:
    image_size: 128
    resize_mode: letterbox
    n_components: 32
    random_state: 42
    c: 10.0
    epsilon: 0.1
    gamma: scale
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
