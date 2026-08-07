# buffalo-weight-pred

Base Python para treinamento de modelo de predição de peso de búfalos a partir de máscaras binarizadas.

## Dados

`data/` contém dois itens esperados:

- `indice.xlsx`: índice de máscaras com nome do arquivo, fazenda, peso e tag de uso.
- `conjunto-de-mascaras/`: máscaras binarizadas usadas no treinamento.

## Máscaras Binarizadas

O treinamento usa máscaras geradas pelo modelo de segmentação `birefnet-general` com método de binarização `LimiarFixoBaixa`.

Somente máscaras presentes no `indice.xlsx` devem ser usadas no treinamento.

A qualidade da segmentação e da binarização é documentada em
[`docs/mask-segmentation-reference.md`](docs/mask-segmentation-reference.md),
com base no relatório upstream
`Relatório PIBIC_ PROCESSAMENTO DE IMAGEM PARA ESTIMATIVA DO PESO DE BÚFALO .pdf`.

## Configuração

`configs/shared.yaml` define dados, Índice de Features, Divisão Estratificada e diretório de treino compartilhados.

`configs/report.yaml` define a configuração do relatório e o diretório de artefatos da reprodução oficial.

## Arquivos Gerados

`generated/` contém artefatos derivados de `data/`, como o Índice de Features e a Divisão Estratificada.

Esses arquivos podem ser recriados a partir de `data/` e das configurações em `configs/`.

## Comandos

O Núcleo Reprodutível do Relatório usa a interface pública `python main.py`.
Depois de preparar o ambiente oficial, o estágio de entradas deve preceder a
Seleção Manual de Features:

```bash
PYTHON=.venv/bin/python make setup
PYTHON=.venv/bin/python make inputs
PYTHON=.venv/bin/python make feature-selection
```

`feature-selection` executa o Random Forest congelado em CPU e a Rede Densa por
Feições em CUDA sobre a Divisão Estratificada Canônica. O resultado provisório
fica em `generated/report/feature_selection/`, com as duas tabelas tidy, três
figuras a 300 DPI, a minuta auditável e um contrato que permanece sem
`selected_features` ou decisão humana. Use
`python main.py feature-selection --dry-run --config configs/report.yaml` para
consultar `blocked`, `absent`, `obsolete` ou `reusable` sem gravar arquivos.

Depois da revisão científica, crie manualmente o contrato descrito em
[`docs/schemas/feature-selection.md`](docs/schemas/feature-selection.md) e uma
versão revisada da minuta. Ambos devem estar versionados em um worktree limpo.
Confira o portão sem gravar e então promova o pacote confirmed:

```bash
python main.py confirm-features --dry-run --config configs/report.yaml \
  --contract caminho/contrato.json --report caminho/relatorio-revisado.md
python main.py confirm-features --config configs/report.yaml \
  --contract caminho/contrato.json --report caminho/relatorio-revisado.md
python main.py baselines --dry-run --config configs/report.yaml
```

O snapshot confirmado é escrito em
`evidence/confirmed/feature_selection/v1/`. O comando `baselines` permanece
bloqueado antes de treino enquanto esse pacote estiver ausente, provisório,
adulterado ou incompatível com as entradas atuais.

Depois que as quatro Configurações Baseline e a referência estiverem atuais,
gere a comparação controlada sem iniciar novo treino:

```bash
python main.py compare-baselines --dry-run --config configs/report.yaml
PYTHON=.venv/bin/python make compare-baselines
```

O comando recusa configurações ausentes, obsoletas ou incompatíveis. O pacote
provisório fica em `generated/report/approach_selection/` com
`baseline_metrics.csv`, três figuras canônicas, a minuta de seleção, o modelo de
contrato ainda sem decisão humana e um manifesto escrito por último. O schema é
documentado em [`docs/schemas/baselines.md`](docs/schemas/baselines.md).

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

Executar reprodução integral do relatório:

```bash
PYTHON=.venv/bin/python make reproduce
```

Rodar testes:

```bash
PYTHON=.venv/bin/python make test
```
