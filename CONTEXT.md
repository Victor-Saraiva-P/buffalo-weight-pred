# Predição de Peso de Búfalos

Este contexto define linguagem do domínio para treinamento de modelos de predição de peso de búfalos a partir de máscaras derivadas de imagens digitais.

## Language

**Máscara Binarizada**:
Imagem em preto e branco que representa a região do búfalo separada do fundo após segmentação e binarização.
_Avoid_: Combo, máscara preta e branca

A qualidade da segmentação e da binarização deve ser interpretada com base no
relatório upstream `Relatório PIBIC_ PROCESSAMENTO DE IMAGEM PARA ESTIMATIVA DO
PESO DE BÚFALO .pdf` e nas regras de `docs/mask-segmentation-reference.md`.

As 132 Máscaras Binarizadas da Etapa de Predição de Peso não incluem o grupo
`Fotos 1`. Todas pertencem aos grupos `Fotos 2` ou `Fotos 3` e foram
manualmente validadas com a tag `ok`.

`Fotos 2` e `Fotos 3` correspondem a fazendas diferentes. A categoria `B10`
está concentrada na fazenda `Faco`, conforme o Índice de Máscaras. Portanto,
fazenda e extremo de peso estão confundidos na amostra atual e não podem ser
separados apenas pela análise dos B10.

A inspeção manual das 13 Máscaras Binarizadas da categoria `B10` confirmou boa
correspondência visual com as fotografias originais, sem recortes grosseiros,
inclusão relevante de fundo ou perda do animal. Portanto, defeitos visuais
óbvios de segmentação não são a principal hipótese para a subestimação dos
extremos. Diferenças geométricas sutis ainda podem afetar features sensíveis.

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
Feature geométrica cuja contribuição para estimar o peso vivo é sustentada pelas Predições OOF da Divisão Estratificada Canônica, sem implicar estabilidade entre outras divisões ou validação em animais novos.
_Avoid_: Feature correta, variável estável em qualquer divisão, variável boa sem critério de validação

**Universo de Features Candidatas**:
União das 26 features geométricas já calculadas pelo projeto ou sugeridas pelo orientador, mantida para definir e investigar as medições sem presumir que todas entrarão nos modelos de predição.
_Avoid_: Conjunto final de features, todas as features do modelo

**Conjunto Compartilhado de Features**:
Subconjunto definido somente após a análise de redundância e dos Testes de Retirada de Features do Universo de Features Candidatas. Ele é congelado antes da escolha e do ajuste fino das configurações finais e fornecido de forma idêntica ao Random Forest e à Rede Densa por Feições. O ajuste fino não reabre a seleção de features.
_Avoid_: Universo de features candidatas, features específicas de um dos modelos

**Roundness**:
Descritor de alongamento calculado como quatro vezes a área dividida por pi vezes o quadrado do eixo maior da elipse ajustada. É distinto da circularidade, que depende do perímetro e também responde à irregularidade do contorno.
_Avoid_: Circularidade, sinônimo genérico de forma arredondada

**Elipse Equivalente**:
Elipse definida pelos momentos centrais da Máscara Binarizada, representada no Universo de Features Candidatas pelos comprimentos de seus eixos maior e menor. Seu centro e seu ângulo não integram essa representação.
_Avoid_: Fit ellipse como uma feature única, ângulo de postura

**Centróide Relativo**:
Representação do centróide da Máscara Binarizada pela distância horizontal absoluta ao centro do retângulo delimitador, normalizada pela largura, e pela posição vertical dentro desse retângulo, normalizada pela altura. Não representa a posição absoluta do animal na imagem.
_Avoid_: Coordenadas absolutas do centróide, posição no quadro

**Bounding Rectangle**:
Menor retângulo alinhado aos eixos da imagem que contém a Máscara Binarizada, representado por largura, altura, área, proporção largura/altura e extensão da máscara dentro do retângulo. Suas medições podem variar com a postura ou inclinação do animal.
_Avoid_: Retângulo rotacionado, caixa delimitadora sem declarar o alinhamento

**Diâmetro de Feret**:
Maior distância euclidiana entre dois pontos do contorno da Máscara Binarizada, equivalente ao calibre máximo e calculável a partir do fecho convexo.
_Avoid_: Feret mínimo, ângulo de Feret, diâmetro alinhado aos eixos

**Perímetro de Crofton**:
Estimativa do comprimento do contorno da Máscara Binarizada pela fórmula de Crofton em quatro direções, adotada para reduzir a sensibilidade da medição à grade e à inclinação do animal. É a definição de perímetro usada também por circularidade e convexidade.
_Avoid_: Contagem de lados expostos dos pixels, perímetro sem declarar o estimador

**Escala Canônica da Máscara**:
Sistema de unidades no qual a maior dimensão da imagem mede 1024 pixels canônicos, sem reamostrar a Máscara Binarizada. Comprimentos originais são multiplicados pelo fator de escala, áreas pelo quadrado e proxies volumétricos pelo cubo; proporções e descritores invariantes permanecem adimensionais.
_Avoid_: Pixels originais como unidades comparáveis, redimensionamento destrutivo da máscara

**Fecho Convexo da Máscara**:
Máscara preenchida do menor conjunto convexo que contém a Máscara Binarizada. Sua quantidade de pixels define a área convexa; solidez é a razão entre área e área convexa; convexidade é a razão entre os Perímetros de Crofton do fecho e da máscara original.
_Avoid_: Polígono contínuo misturado com contagem de pixels, solidez como sinônimo de convexidade

**Ocupação Vertical Regional**:
Média da quantidade de pixels da Máscara Binarizada por coluna em cada terço horizontal do Bounding Rectangle. A representação usa `center_vertical_occupancy`, `end_vertical_occupancy_min`, `end_vertical_occupancy_max` e `center_to_end_occupancy_ratio`, sem pressupor a direção do animal.
_Avoid_: Espessura anatômica, amostragem fixa de nove colunas

**Momentos de Hu Selecionados**:
Primeiro e segundo invariantes de Hu calculados diretamente dos momentos centrais normalizados da Máscara Binarizada, sem transformação logarítmica. Os demais cinco invariantes não integram o Universo de Features Candidatas.
_Avoid_: Todos os sete momentos de Hu, momentos com log sem declaração

**Proxies Volumétricos Geométricos**:
`area_power_1_5` e `area_major_axis_product`, relações em pixels canônicos cúbicos mantidas como hipóteses exploratórias sobre porte tridimensional. Não constituem medições de volume corporal e permanecem sujeitas aos Testes de Retirada de Features.
_Avoid_: Volume estimado, medida corporal tridimensional

**Seleção Manual de Features**:
Decisão humana sobre quais features geométricas entram na avaliação de modelos, tomada a partir de evidências comparativas geradas pelo projeto. Uma agente de IA pode organizar os resultados e recomendar interpretações, mas não declara o conjunto final sem confirmação humana explícita.
_Avoid_: Seleção automática, otimização automática de features

**Relatório de Seleção do Conjunto Compartilhado de Features**:
Documento versionado que liga as evidências comparativas à Seleção Manual de Features. Uma agente de IA produz a minuta a partir dos resultados e da documentação do protocolo; as interpretações e o Conjunto Compartilhado de Features somente se tornam definitivos após revisão e confirmação humana em uma sessão de grilling. O documento resume todos os Testes de Retirada de Features, inclusive os neutros e rejeitados, mas referencia artefatos separados para resultados detalhados por baseline e fold. Ele apresenta mapas de calor de redundância e dos efeitos da retirada, além de um gráfico dos efeitos de permutação. Ele serve como fonte intermediária auditável e concisa para o futuro relatório PIBIC.
_Avoid_: Saída automática do modelo, decisão autônoma da IA, relatório PIBIC final

**Evidência Comparativa de Feature**:
Conjunto de resultados usado para interpretar a contribuição de uma feature geométrica em cada classe de modelo: desempenho isolado, retreinamento após remoção individual, impacto da permutação fora da amostra e Testes de Retirada de Features para grupos redundantes. Remoção mede se o modelo consegue se adaptar sem a feature; permutação mede quanto o modelo treinado depende da associação entre a feature e cada Máscara Válida.
_Avoid_: Escolha automática de feature, importância sem validação

**Redundância Entre Features**:
Situação em que duas ou mais features geométricas carregam sinais semelhantes sobre a máscara válida, exigindo interpretação conjunta das evidências comparativas.
_Avoid_: Feature duplicada como sinônimo de feature inútil

**Redundância Estrutural Entre Features**:
Caso de Redundância Entre Features determinado pelas próprias definições matemáticas, no qual uma feature pode ser recuperada exatamente de outra, independentemente das máscaras observadas. Diâmetro equivalente e área elevada a 1,5 são exemplos em relação à área.
_Avoid_: Correlação alta observada na amostra, evidência de desempenho preditivo

**Teste de Retirada de Features**:
Comparação controlada entre o Universo de Features Candidatas e uma variação que omite uma feature ou grupo cuja redundância foi identificada, mantendo iguais os folds e as demais condições de avaliação. A redundância motiva o teste, mas não determina a exclusão.
_Avoid_: Remoção automática por correlação, descarte definitivo de feature

**Regra Conservadora de Remoção**:
Critério aplicado ao Conjunto Compartilhado de Features pelo qual uma feature ou grupo pode ser recomendado para remoção quando sua ausência melhora o MAE em mais de 1 kg em pelo menos uma das duas Configurações Baseline baseadas em features e não piora a outra em mais de 1 kg. Se qualquer baseline piorar mais de 1 kg, ou se ambas permanecerem neutras, a feature é mantida. A decisão final continua humana.
_Avoid_: Média do efeito entre modelos, remoção automática, conjunto específico por classe de modelo

**Equivalência Prática de Feature**:
Diferença de até 1 kg no MAE entre o Conjunto Compartilhado de Features e um Teste de Retirada de Features, tratada como efeito praticamente neutro em vez de melhora ou piora relevante.
_Avoid_: Igualdade numérica exata, significância sem tamanho de efeito

**Etapa de Predição de Peso**:
Etapa que avalia em que medida informações derivadas das Máscaras Binarizadas permitem predizer o peso vivo, compara abordagens geométricas e convolucionais sob um protocolo controlado e identifica limitações da amostra e da aquisição.
_Avoid_: Avaliação de segmentação, treinamento de segmentação

**Modelo Clássico de Predição**:
Modelo supervisionado tradicional usado na Etapa de Predição de Peso para estimar peso vivo a partir de features geométricas.
_Avoid_: IA, modelo de segmentação, rede neural quando o modelo avaliado não for uma rede neural

**Rede Densa por Feições**:
Rede neural inicializada com pesos aleatórios que estima o peso vivo a partir de um vetor de feições calculadas automaticamente da Máscara Binarizada, sem receber seus pixels diretamente. Modelos fundacionais tabulares pré-treinados constituem outra classe de modelo.
_Avoid_: CNN, Modelo Híbrido de Predição

**Rede Convolucional por Máscara**:
Rede neural convolucional que estima o peso vivo a partir de representações espaciais derivadas dos pixels da Máscara Binarizada.
_Avoid_: Rede Densa por Feições, modelo de segmentação

**Configuração de Modelo**:
Variação nomeada de uma abordagem de predição, definida por sua arquitetura, hiperparâmetros e procedimento de treinamento.
_Avoid_: Modelo quando o texto se refere à variação nomeada, IA

**Configuração Baseline**:
Configuração de Modelo predefinida que representa uma abordagem antes do ajuste fino e permite compará-la sob o mesmo protocolo controlado.
_Avoid_: Preditor trivial da média, configuração escolhida depois de observar resultados

**Ajuste Fino de Configuração**:
Refinamento posterior da Configuração Baseline da Abordagem de Maior Potencial, realizado somente depois de congelado o Conjunto Compartilhado de Features. O orçamento permite no máximo três variações adicionais, cujas receitas devem ser registradas antes de executar o ajuste e não podem ser ampliadas após observar resultados. O ajuste não pode alterar ou reabrir a seleção de features.
_Avoid_: Ajuste simultâneo de features e hiperparâmetros, busca aberta, comparação controlada entre abordagens

**Abordagem de Maior Potencial**:
Uma das quatro abordagens baseline — Random Forest, Rede Densa por Feições, CNN compacta inicializada do zero ou ResNet-18 pré-treinada — escolhida somente depois de congelado o Conjunto Compartilhado de Features. A escolha humana usa o MAE OOF como evidência principal e as demais métricas predefinidas como evidências descritivas, sem pontuação combinada automática. Somente essa abordagem avança para o Ajuste Fino de Configuração.
_Avoid_: Vencedora automática pelo menor MAE, abordagem escolhida antes do congelamento das features, modelo final ajustado

**Relatório de Seleção da Abordagem de Maior Potencial**:
Documento versionado, separado do Relatório de Seleção do Conjunto Compartilhado de Features, que compara as quatro Configurações Baseline depois do congelamento do contrato de features. Uma agente de IA organiza o MAE OOF primário e as evidências descritivas predefinidas; uma sessão de grilling revisa as interpretações e confirma humanamente uma Abordagem de Maior Potencial.
_Avoid_: Relatório de seleção de features, ranking autônomo da IA, relatório do ajuste fino

**Artefato de Avaliação Atual**:
Resultado de uma Configuração de Modelo cuja proveniência corresponde à mesma configuração, Divisão Estratificada, entradas e procedimento de treinamento da avaliação solicitada.
_Avoid_: Cache válido apenas porque os arquivos existem, resultado anterior sem proveniência verificada

**Modelo de Predição por Máscara**:
Modelo supervisionado que estima peso vivo diretamente a partir da Máscara Binarizada, sem depender do Índice de Features como entrada principal.
_Avoid_: Modelo Clássico de Predição quando o modelo lê pixels da máscara

**Modelo Híbrido de Predição**:
Modelo supervisionado que estima peso vivo combinando pixels da Máscara Binarizada com um conjunto declarado de features do Índice de Features.
_Avoid_: Modelo de Predição por Máscara, fusão sem identificar as entradas

**Categoria de Peso**:
Grupo definido por faixas quantílicas globais do peso dos animais no dataset inteiro, usado para balancear a avaliação entre faixas de peso absoluto.
As categorias são nomeadas com códigos neutros (`B1`, `B2`, ...), acompanhados por rótulos como `Faixa 1`, `Faixa 2`, ... para evitar assumir quartis quando a quantidade de faixas é configurável.
_Avoid_: Categoria da fazenda, balde por fazenda, quartil quando a quantidade de faixas for configurável, leve/pesado quando a quantidade de faixas tornar o rótulo ambíguo

**Divisão Estratificada**:
Arquivo que associa cada máscara válida a uma categoria de peso e a um fold de avaliação, preservando a distribuição das categorias de peso entre folds.
_Avoid_: Separação aleatória, split temporário

**Divisão Estratificada Canônica**:
Única Divisão Estratificada de cinco folds usada em toda a avaliação controlada atual. Ela torna as comparações pareadas e reproduzíveis, mas não mede sensibilidade a outras divisões possíveis da amostra.
_Avoid_: Cinco divisões repetidas, validação independente, evidência de estabilidade entre seeds

**Predição OOF**:
Predição de peso de uma Máscara Válida produzida por uma instância do modelo que não usou essa máscara no próprio treinamento.
_Avoid_: Predição do conjunto de treino, predição de um modelo treinado com todas as máscaras

**MAE OOF Pós-Seleção**:
MAE calculado a partir de Predições OOF das mesmas 132 Máscaras Válidas cujos resultados orientaram a Seleção Manual de Features, a escolha da Abordagem de Maior Potencial e o Ajuste Fino de Configuração. Cada predição permanece fora do treino de sua instância, mas a métrica resume a amostra usada para escolher o procedimento e pode ser otimista em relação a animais novos. Ela constitui evidência de desenvolvimento, não validação independente.
_Avoid_: MAE de treino, estimativa independente de generalização, desempenho confirmado em animais novos

**Caso Difícil Compartilhado**:
Máscara Válida situada no pior décimo de erro absoluto de pelo menos três das quatro Configurações Baseline comparadas.
_Avoid_: Outlier confirmado, máscara errada, causa comprovada do erro

**Caso Divergente Entre Abordagens**:
Máscara Válida situada no pior décimo da amplitude entre as quatro Predições OOF baseline produzidas para o mesmo animal.
_Avoid_: Caso em que uma abordagem está necessariamente correta, prova de complementaridade causal

**Seleção Interna Isolada**:
Escolha de parâmetros ou do momento de parada usando apenas as Máscaras Válidas disponíveis para treino, sem consultar o fold reservado para avaliação.
_Avoid_: Seleção independente, ajuste usando o fold de avaliação

**Estabilidade Entre Divisões**:
Consistência do desempenho preditivo quando a avaliação é repetida em diferentes divisões estratificadas das mesmas máscaras válidas.
_Avoid_: Resultado de uma única seed como evidência suficiente

**Núcleo Reprodutível do Relatório**:
Conjunto mínimo de entradas, métodos e evidências necessário para reproduzir todos os resultados apresentados no relatório final.
_Avoid_: Histórico completo de experimentos, arquivo de tentativas descartadas

**Registro de Experimentos Encerrados**:
Inventário conciso das hipóteses e abordagens descartadas, mantido como memória científica sem integrar as evidências do relatório final.
_Avoid_: Arquivo de código obsoleto, relatório de resultados negativos
