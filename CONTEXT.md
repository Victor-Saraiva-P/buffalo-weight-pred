# Predição de Peso de Búfalos

Este contexto define linguagem do domínio para treinamento de modelos de predição de peso de búfalos a partir de máscaras derivadas de imagens digitais.

## Language

**Máscara Binarizada**:
Imagem em preto e branco que representa a região do búfalo separada do fundo após segmentação e binarização.
_Avoid_: Combo, máscara preta e branca

**Conjunto de Máscaras**:
Coleção de máscaras binarizadas produzidas pela mesma combinação de modelo de segmentação e método de binarização.
_Avoid_: Combo

**Avaliação de Segmentação**:
Etapa anterior que compara modelos pré-treinados de segmentação e métodos de binarização para escolher o conjunto de máscaras mais adequado ao projeto.
_Avoid_: Treinamento de segmentação, ajuste fino de segmentação

**Máscara Preditiva**:
Máscara binarizada produzida por um modelo de segmentação e método de binarização, usada como entrada para extração de features geométricas.
_Avoid_: Ground truth, anotação manual

**Índice de Máscaras**:
Planilha que define quais máscaras binarizadas têm rótulo válido para treinamento, associando nome do arquivo, fazenda, peso e tag de uso.
_Avoid_: Lista de fotos, tabela de imagens

**Índice de Features**:
Arquivo derivado do índice de máscaras e do conjunto de máscaras, contendo uma linha por máscara válida e colunas com rótulos e features geométricas calculadas.
_Avoid_: Banco de dados, tabela temporária

**Etapa de Predição de Peso**:
Etapa que usa features geométricas extraídas de máscaras binarizadas para avaliar modelos de predição do peso vivo dos búfalos.
_Avoid_: Avaliação de segmentação, treinamento de segmentação

**Categoria de Peso**:
Grupo definido por quartis globais do peso dos animais no dataset inteiro, usado para balancear a avaliação entre faixas de peso absoluto.
_Avoid_: Categoria da fazenda, balde por fazenda

**Divisão Estratificada**:
Arquivo que associa cada máscara válida a uma categoria de peso e a um fold de avaliação, preservando a distribuição das categorias de peso entre folds.
_Avoid_: Separação aleatória, split temporário
