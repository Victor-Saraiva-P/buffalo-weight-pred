# Revisão Humana do Diagnóstico Expandido — Parte II

Este documento registra a revisão humana dos diagnósticos produzidos após a seleção da Random Forest como abordagem de continuidade. O escopo reúne o ajuste fino pré-registrado, a caracterização de cobertura e padrões de erro, as curvas de aprendizado controladas e a análise de sensibilidade das predições a perturbações das máscaras. A revisão não reabre a seleção de atributos nem a escolha da abordagem.

Os valores apresentados são tratados como **MAE OOF Pós-Seleção** e demais métricas associadas ao processo de desenvolvimento experimental sobre as 132 máscaras disponíveis.

## Ajuste da configuração selecionada

A configuração `random_forest_baseline` permaneceu como melhor resultado dentro do orçamento pré-registrado de três variações. O baseline apresentou MAE OOF de 53,33 kg, RMSE de 73,52 kg, viés de 4,03 kg e R² de 0,756. As variações `rf_tuning_estimators`, `rf_tuning_depth` e `rf_tuning_features` apresentaram MAE de 55,19 kg, 54,90 kg e 56,48 kg, respectivamente. Assim, nenhuma das três alterações avaliadas forneceu redução do MAE em relação à configuração de referência, e não há evidência neste experimento que justifique substituir a configuração confirmada.

## Cobertura da amostra e padrões de erro

A análise diagnóstica incluiu 132 máscaras, distribuídas em dez categorias auxiliares de peso, duas fazendas e três resoluções de imagem. Foram identificados cinco casos difíceis compartilhados entre múltiplas abordagens e 14 casos de divergência elevada entre as predições dos modelos.

Na Random Forest, o desempenho variou substancialmente entre as categorias de peso. A categoria B1 apresentou MAE de 31,71 kg e viés de +31,71 kg, enquanto B10 apresentou MAE de 126,92 kg e viés de -126,92 kg. O comportamento em B10 caracteriza forte tendência de subestimação dos animais mais pesados dentro da amostra estudada.

Na comparação por origem, a Random Forest apresentou MAE de 65,45 kg na fazenda Faco e 29,92 kg na fazenda Manezinho. A diferença permaneceu na faixa de peso compartilhada entre 92 e 265 kg, na qual os MAEs foram 80,49 kg e 31,30 kg, respectivamente. Esses resultados devem ser interpretados como associações observadas na amostra, pois origem, faixa de peso e condições de aquisição permanecem confundidas e não podem ser isoladas por este delineamento.

Os resíduos da Random Forest apresentaram correlação de Pearson de 0,788 com os da rede densa, 0,786 com os da CNN compacta e 0,490 com os da ResNet-18. A concordância parcial dos erros entre abordagens indica que parte dos casos difíceis não é exclusiva de um único modelo.

## Curvas de aprendizado controladas

Para a Random Forest, os MAEs OOF médios entre folds foram 52,87 kg com 50% do conjunto externo de treinamento, 53,15 kg com 75% e 53,50 kg com 100%. Dentro das frações e do protocolo avaliados, não foi observada redução monotônica do erro com o aumento do subconjunto de treinamento.

A rede densa apresentou redução progressiva do MAE médio, de 61,93 kg para 57,73 kg e 53,42 kg nas frações de 50%, 75% e 100%. A CNN compacta apresentou 70,81 kg, 65,01 kg e 66,22 kg, enquanto a ResNet-18 apresentou 73,33 kg, 59,06 kg e 73,62 kg. Essas curvas descrevem o comportamento no intervalo amostral avaliado e não sustentam extrapolação sobre quanto desempenho adicional seria obtido com novas coletas.

## Sensibilidade das predições às máscaras

Foram gerados 1.056 registros de perturbação para a Random Forest. Das 132 máscaras, 96 foram elegíveis para o par de operações morfológicas e 36 foram rejeitadas pelos critérios de topologia ou margem. Entre as rejeições morfológicas, 27 decorreram de violação topológica após contração, quatro após expansão e cinco de margem insuficiente para expansão.

Nas translações que preservaram integralmente o primeiro plano, a variação da predição foi de 0 kg. Esse comportamento é coerente com o conjunto de atributos confirmado, formado por medidas geométricas definidas em relação à própria silhueta e ao seu retângulo delimitador. Vinte e três deslocamentos foram rejeitados por moverem parte do primeiro plano para fora dos limites da imagem e, portanto, não entraram nessa interpretação de invariância translacional.

As mudanças de escala de ±5% produziram alterações maiores. O crescimento da silhueta apresentou variação absoluta média de 11,82 kg, enquanto a redução apresentou 14,60 kg. As operações morfológicas produziram respostas ainda mais intensas: a contração apresentou variação absoluta média de 43,86 kg e a expansão, 51,92 kg. Em média, a contração deslocou a predição em -42,67 kg e a expansão em +49,12 kg. Esses resultados indicam que a predição é sensível a mudanças que alteram o tamanho e o contorno da silhueta, aspecto relevante porque tais propriedades estão diretamente representadas entre os atributos geométricos usados pelo modelo.

## Síntese da revisão

A revisão mantém a `random_forest_baseline` como configuração da abordagem selecionada. O ajuste fino restrito não produziu ganho de MAE. Os diagnósticos mostram concentração de erros em partes específicas da distribuição de peso, especialmente no extremo superior, diferenças associadas à origem que não podem ser separadas das demais características da amostra, ausência de melhora monotônica da Random Forest entre as frações de treinamento avaliadas e sensibilidade relevante a alterações de escala e contorno das máscaras. Em contrapartida, deslocamentos puros que preservam a silhueta não alteraram as predições.

Esses achados devem fundamentar a discussão de limitações da Parte II, em especial quanto ao tamanho e à cobertura da amostra, à associação entre origem e distribuição de peso, à dependência da qualidade geométrica das máscaras e à necessidade de avaliação futura em uma coleta separada do conjunto usado para desenvolvimento.

## Registro de revisão humana

- Status: revisado
- Decisão: aprovar o pacote diagnóstico expandido sem reabrir a seleção de atributos ou a escolha da Random Forest.
- Revisor: Victor Alexandre Saraiva Pimentel
- Data da revisão: 2026-08-07
- Referência da decisão: GitHub Issue #27
