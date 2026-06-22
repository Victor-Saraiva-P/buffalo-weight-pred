# Contexto do Projeto

Este projeto integra uma pesquisa PIBIC/PIC sobre processamento de imagem para estimativa do peso vivo de bubalinos. A etapa atual parte de mascaras binarizadas ja selecionadas em uma etapa anterior de avaliacao de segmentacao.

## Documentos de Referencia

**`Modelo de plano de trabalho dos PICs atual - Plano 1.pdf`**:
Plano original do trabalho, com escopo formal de avaliacao de metodos de segmentacao semantica por deteccao de objeto saliente em imagens digitais de bubalinos. O plano lista familias de modelos como U2Net, ISNet, SAM e BiRefNet, alem da avaliacao por area, contorno e associacao com a regiao de interesse manual.

**`Relatorio PIBIC_ PROCESSAMENTO DE IMAGEM PARA ESTIMATIVA DO PESO DE BUFALO .pdf`**:
Relatorio parcial da etapa de avaliacao de segmentacao. O documento descreve curadoria manual, criacao de ground truths, tags de dificuldade, avaliacao de segmentacao bruta, estrategias de binarizacao e validacao das melhores combinacoes. Essa etapa indicou a familia BiRefNet, especialmente combinada com LimiarFixoBaixa, como base adequada para gerar mascaras binarizadas preditivas.

## Relacao Entre Etapas

1. Avaliacao de Segmentacao: compara modelos pre-treinados e metodos de binarizacao para selecionar um conjunto de mascaras adequado.
2. Etapa de Predicao de Peso: usa as mascaras binarizadas selecionadas para extrair features geometricas e avaliar modelos de predicao do peso vivo.

## Recorte Atual

A etapa atual usa 132 mascaras com peso valido. O objetivo nao e treinar ou ajustar modelos de segmentacao, mas avaliar modelos de predicao de peso a partir das features geometricas extraidas dessas mascaras.
