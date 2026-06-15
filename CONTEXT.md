# Predição de Peso de Búfalos

Este contexto define linguagem do domínio para treinamento de modelos de predição de peso de búfalos a partir de máscaras derivadas de imagens digitais.

## Language

**Máscara Binarizada**:
Imagem em preto e branco que representa a região do búfalo separada do fundo após segmentação e binarização.
_Avoid_: Combo, máscara preta e branca

**Conjunto de Máscaras**:
Coleção de máscaras binarizadas produzidas pela mesma combinação de modelo de segmentação e método de binarização.
_Avoid_: Combo

**Índice de Máscaras**:
Planilha que define quais máscaras binarizadas têm rótulo válido para treinamento, associando nome do arquivo, fazenda, peso e tag de uso.
_Avoid_: Lista de fotos, tabela de imagens

**Índice de Features**:
Arquivo derivado do índice de máscaras e do conjunto de máscaras, contendo uma linha por máscara válida e colunas com rótulos e features geométricas calculadas.
_Avoid_: Banco de dados, tabela temporária
