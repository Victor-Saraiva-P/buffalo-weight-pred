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

**Máscara Válida**:
Máscara binarizada representada por uma única linha do Índice de Máscaras, com peso válido e correspondendo a um único animal.
_Avoid_: Duplicata do animal, múltiplas fotos do mesmo animal como amostras independentes

**Índice de Features**:
Arquivo derivado do índice de máscaras e do conjunto de máscaras, contendo uma linha por máscara válida e colunas com rótulos e features geométricas calculadas.
_Avoid_: Banco de dados, tabela temporária

**Feature Preditiva Útil**:
Feature geométrica que contribui para estimar o peso vivo de forma estável em validação fora da amostra.
_Avoid_: Feature correta, variável boa sem critério de validação

**Seleção Manual de Features**:
Decisão humana sobre quais features geométricas entram na avaliação de modelos, tomada a partir de evidências comparativas geradas pelo projeto.
_Avoid_: Seleção automática, otimização automática de features

**Evidência Comparativa de Feature**:
Resultado usado para julgar uma feature geométrica comparando seu desempenho isolado, sua ausência no conjunto de features e o impacto de embaralhar seus valores fora da amostra.
_Avoid_: Escolha automática de feature, importância sem validação

**Redundância Entre Features**:
Situação em que duas ou mais features geométricas carregam sinais semelhantes sobre a máscara válida, exigindo interpretação conjunta das evidências comparativas.
_Avoid_: Feature duplicada como sinônimo de feature inútil

**Etapa de Predição de Peso**:
Etapa que usa features geométricas extraídas de máscaras binarizadas para avaliar modelos de predição do peso vivo dos búfalos.
_Avoid_: Avaliação de segmentação, treinamento de segmentação

**Modelo Clássico de Predição**:
Modelo supervisionado tradicional usado na Etapa de Predição de Peso para estimar peso vivo a partir de features geométricas.
_Avoid_: IA, modelo de segmentação, rede neural quando o modelo avaliado não for uma rede neural

**Configuração de Modelo**:
Variação nomeada de um Modelo Clássico de Predição, usada para comparar diferentes hiperparâmetros ou tipos de modelo na mesma avaliação.
_Avoid_: Modelo quando o texto se refere à variação nomeada, IA

**Modelo de Predição por Máscara**:
Modelo supervisionado que estima peso vivo diretamente a partir da Máscara Binarizada, sem depender do Índice de Features como entrada principal.
_Avoid_: Modelo Clássico de Predição quando o modelo lê pixels da máscara

**Categoria de Peso**:
Grupo definido por faixas quantílicas globais do peso dos animais no dataset inteiro, usado para balancear a avaliação entre faixas de peso absoluto.
As categorias são nomeadas com códigos neutros (`B1`, `B2`, ...), acompanhados por rótulos como `Faixa 1`, `Faixa 2`, ... para evitar assumir quartis quando a quantidade de faixas é configurável.
_Avoid_: Categoria da fazenda, balde por fazenda, quartil quando a quantidade de faixas for configurável, leve/pesado quando a quantidade de faixas tornar o rótulo ambíguo

**Divisão Estratificada**:
Arquivo que associa cada máscara válida a uma categoria de peso e a um fold de avaliação, preservando a distribuição das categorias de peso entre folds.
_Avoid_: Separação aleatória, split temporário

**Estabilidade Entre Divisões**:
Consistência do desempenho preditivo quando a avaliação é repetida em diferentes divisões estratificadas das mesmas máscaras válidas.
_Avoid_: Resultado de uma única seed como evidência suficiente
