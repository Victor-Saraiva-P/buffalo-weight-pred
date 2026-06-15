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

O primeiro experimento usa `configs/baseline.yaml`. Essa configuração define os caminhos dos dados, coluna-alvo, tag válida, features incluídas e arquivo gerado com as features calculadas.

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

Rodar testes:

```bash
PYTHON=.venv/bin/python make test
```
