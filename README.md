# buffalo-weight-pred

Núcleo Reprodutível do Relatório PIBIC para estimativa do peso vivo de búfalos a partir de máscaras binarizadas.

## Objetivo

Fornecer uma superfície pública limpa, determinística e auditável para treinamento, avaliação e reprodução integral do pipeline de predição de peso de búfalos a partir de feições geométricas extraídas de máscaras binárias.

## Limitações do Estudo

- Os resultados numéricos de desempenho preditivo reportados constituem métricas de **MAE OOF Pós-Seleção** (out-of-fold cross-validation) obtidas na amostra curada de 132 máscaras válidas.
- Os resultados representam evidência de desenvolvimento e seleção metodológica, **não reivindicam validação independente em animais novos** ou provenientes de outras fazendas.

## Requisitos de Ambiente (Python / CUDA)

- **Python**: 3.11+.
- **CUDA**: Ambiente com suporte a GPU/CUDA para execução das etapas neurais (Rede Densa por Feições, CNN compacta e ResNet-18) via PyTorch.

## Instalação e Setup

Criar ambiente virtual e instalar dependências:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Preparar e auditar o ambiente oficial:

```bash
PYTHON=.venv/bin/python make setup
# ou diretamente via CLI Python:
.venv/bin/python main.py setup
```

## Entradas Curadas

A pasta `data/` contém os dados de entrada oficiais do repositório:

- `data/mask_index.csv` (ou `data/indice.xlsx`): índice curado com nome do arquivo, fazenda, peso contínuo (kg) e tag de uso para as 132 máscaras válidas.
- `data/conjunto-de-mascaras/`: repositório das 132 máscaras binarizadas em formato PNG (geradas pelo modelo `birefnet-general` com método `LimiarFixoBaixa`).

A documentação de proveniência e qualidade de segmentação está em [`docs/mask-segmentation-reference.md`](docs/mask-segmentation-reference.md).

## Configuração e Artefatos Reconstruíveis

- `configs/report.yaml`: arquivo de configuração canônico do relatório, especificando contagem de folds (5), sementes e diretório de saída.
- `generated/report/`: diretório onde são gravados os artefatos intermediários e provisórios reconstruíveis.

Para limpar artefatos reconstruíveis de um estágio específico:

```bash
.venv/bin/python main.py clean inputs --config configs/report.yaml
# ou via Makefile:
PYTHON=.venv/bin/python make report-clean STAGE=inputs
```

## Portões de Decisão Humana

O pipeline exige revisão humana em pontos de transição metodológica antes de avançar para estágios dependentes. Os contratos e minutas revisados devem ser promovidos para snapshots confirmados versionados:

- **Seleção de Atributos** (`confirm-features`): promove o snapshot em `evidence/confirmed/feature_selection/v1/`. Schema em [`docs/schemas/feature-selection.md`](docs/schemas/feature-selection.md).
- **Escolha da Abordagem** (`confirm-approach`): promove o snapshot em `evidence/confirmed/approach_selection/v1/`. Schema em [`docs/schemas/baselines.md`](docs/schemas/baselines.md).
- **Diagnósticos Expandidos** (`confirm-diagnostics`): promove o snapshot em `evidence/confirmed/diagnostics/v1/`.

## Comandos Públicos da CLI

Todos os comandos públicos aceitam `--dry-run` para inspecionar ações e estados sem gravar artefatos em disco.

### 1. Entradas e Feições
```bash
.venv/bin/python main.py inputs --config configs/report.yaml
.venv/bin/python main.py feature-selection --config configs/report.yaml
.venv/bin/python main.py confirm-features --config configs/report.yaml --contract <caminho/contrato.json> --report <caminho/relatorio.md>
```

### 2. Baselines e Comparação de Abordagens
```bash
.venv/bin/python main.py baselines --config configs/report.yaml
.venv/bin/python main.py compare-baselines --config configs/report.yaml
.venv/bin/python main.py confirm-approach --config configs/report.yaml --contract <caminho/contrato.json> --report <caminho/relatorio.md>
```

### 3. Ajuste Fino e Diagnósticos Expandidos
```bash
.venv/bin/python main.py tuning --config configs/report.yaml
.venv/bin/python main.py diagnostics-descriptive --config configs/report.yaml
.venv/bin/python main.py diagnostics-learning --config configs/report.yaml
.venv/bin/python main.py diagnostics-sensitivity --config configs/report.yaml
.venv/bin/python main.py confirm-diagnostics --config configs/report.yaml --contract <caminho/contrato.json> --report <caminho/relatorio.md>
```

### 4. Reprodução Integral
Orquestra o grafo completo de 11 nós entre estágios e portões:

```bash
.venv/bin/python main.py reproduce --config configs/report.yaml --dry-run
PYTHON=.venv/bin/python make reproduce
```

## Suíte de Testes

O comando `make test` é o único ponto de entrada para a suíte de testes do projeto:

```bash
PYTHON=.venv/bin/python make test
```

## Evidências Confirmadas do Núcleo Reprodutível

Artefatos de evidência confirmados e imutáveis estão mantidos sob versionamento:

- `evidence/confirmed/feature_selection/v1/`: pacote confirmado da seleção de 25 atributos.
- `evidence/confirmed/approach_selection/v1/`: pacote confirmado da escolha da Random Forest.
