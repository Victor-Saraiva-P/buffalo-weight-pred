# Referência da segmentação e das máscaras binarizadas

## Fonte autoritativa

Toda afirmação sobre a qualidade da segmentação, a binarização ou a origem das
Máscaras Binarizadas deve usar primeiro:

`Relatório PIBIC_ PROCESSAMENTO DE IMAGEM PARA ESTIMATIVA DO PESO DE BÚFALO .pdf`

O relatório documenta a etapa upstream de Avaliação de Segmentação, incluindo:

- curadoria de 387 imagens de campo;
- criação de ground truths manuais no GIMP;
- tags de dificuldade visual;
- avaliação de máscaras contínuas e binarizadas;
- comparação de modelos e estratégias de binarização;
- validação final por IoU, Precision, Recall, Area Similarity e Perímetro Similarity.

## Resultado relevante

No recorte mais favorável, as combinações `birefnet-hrsod`,
`birefnet-general` e `birefnet-massive`, todas com `LimiarFixoBaixa`,
atenderam simultaneamente aos thresholds definidos no relatório. O relatório
também indica impacto negativo consistente de `baixo_contraste`,
`multi_bufalos` e `ocluido`.

As 132 Máscaras Binarizadas atuais estão incluídas nas 387 imagens avaliadas
nesse estudo upstream e correspondem ao cenário manualmente validado como
`apenas ok`. Portanto, a hipótese de que a segmentação possa ser uma causa
importante do erro de peso deve ser tratada com mais cautela: o recorte atual
foi escolhido justamente para excluir dificuldades visuais anotadas. A ligação
individual entre cada máscara atual, seu resultado de segmentação e seu ground
truth correspondente ainda deve ser preservada como artefato de proveniência.

## Regra de interpretação

Ao documentar as máscaras:

1. Use o relatório PIBIC como referência principal para a qualidade da
   segmentação e da binarização.
2. Diferencie a avaliação upstream das 387 imagens da amostra atual de 132
   máscaras com peso válido e tag manual `ok`, lembrando que as 132 estão
   contidas nas 387.
3. Não use apenas componentes, buracos, contato com borda ou perturbações
   sintéticas como prova de que uma Máscara Binarizada real está errada.
4. Quando a correspondência entre as 132 máscaras e os ground truths não
   estiver comprovada por artefatos, declare essa limitação explicitamente.

## Relação com a predição de peso

A Avaliação de Segmentação ocorre antes da Etapa de Predição de Peso. Uma
Máscara Binarizada pode ter boa qualidade média contra o ground truth e ainda
assim perder informação de escala ou contorno suficiente para a predição de
peso. As duas perguntas devem ser mantidas separadas:

- a máscara representa corretamente o búfalo?
- a geometria disponível nessa representação permite estimar o peso?

Uma resposta positiva à primeira não garante MAE baixo na segunda.
