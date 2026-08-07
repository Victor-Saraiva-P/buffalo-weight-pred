# Handoff — Relatório Final PIBIC / buffalo-weight-pred

## Objetivo da próxima sessão

Continuar a redação de `docs/relatorio-final.md` somente quando surgirem novos resultados confirmados no repositório. O próximo bloco científico depende das issues de ajuste fino e diagnóstico expandido; não antecipar conclusões antes desses artefatos existirem.

## Estado atual

- Branch: `main`.
- Último pull feito em 2026-08-06; `main` foi atualizado até o commit `e456bac` (`Merge pull request #41 ... issue-22-human-gate-approach`).
- O relatório em andamento está em `docs/relatorio-final.md` e permanece local/não rastreado neste checkout.
- Também existe localmente um PDF de referência do orientador: `Relatorio_final_-_ok_assinado_assinado (1).pdf`.
- Não sobrescrever nem resumir a Parte I do relatório: ela foi integrada praticamente na íntegra a partir do relatório parcial aprovado.

## Decisões já consolidadas para o relatório

Consulte diretamente `docs/relatorio-final.md` para a redação vigente e `CONTEXT.md` para o domínio. Não duplicar essas informações em outro documento.

Pontos que devem ser preservados:

- O relatório final é um único documento com duas partes científicas autônomas.
- Parte I: trabalho de segmentação, preservado praticamente na íntegra.
- Parte II: “Predição do peso vivo de bubalinos a partir de máscaras binárias”.
- A Parte II é comparativa/exploratória; não prometer sistema preditivo definitivo.
- A amostra usada na Parte II tem 132 máscaras binárias válidas.
- A divisão experimental canônica usa cinco folds e categorias auxiliares B1–B10; o alvo continua sendo peso contínuo em kg.
- Foram avaliados quatro baselines: Random Forest, rede densa por feições, CNN compacta e ResNet-18 pré-treinada.
- A seleção de atributos está concluída: o conjunto compartilhado final contém 25 atributos; `area_power_1_5` foi removido. Evidências em `evidence/confirmed/feature_selection/v1/`.
- A comparação dos quatro baselines e a escolha humana da abordagem estão concluídas. Evidências em `evidence/confirmed/approach_selection/v1/`.
- A abordagem confirmada para continuidade é `random_forest` / `random_forest_baseline`.
- A rede densa possui MAE OOF numericamente menor por cerca de 0,007 kg, mas foi tratado como empate prático; a Random Forest foi escolhida com apoio de RMSE e R² melhores. A justificativa completa está no pacote confirmado de seleção da abordagem.
- Os resultados são evidência de desenvolvimento OOF pós-seleção, não validação independente em animais novos.

## Estado da redação da Parte II

Já estão redigidos no relatório:

- Introdução.
- Objetivos.
- Fundamentação teórica.
- Delineamento e origem das máscaras.
- Padronização das máscaras.
- Categorias de peso e divisão experimental.
- Extração dos 26 atributos candidatos.
- Modelos de referência.
- Treinamento, métricas e reprodutibilidade.
- Seleção de atributos.
- Comparação das abordagens e critério de escolha.
- Caracterização da amostra e dos folds.
- Evidências da seleção de atributos.
- Comparação dos quatro modelos.
- Escolha da Random Forest como abordagem de maior potencial.

Ainda não fechar:

- `2.4.9 Ajuste da Abordagem Selecionada`.
- `2.4.10 Diagnósticos, Curvas de Aprendizado e Sensibilidade`.
- `2.5.5 Desempenho Preditivo Consolidado e Padrões de Erro`.
- `2.5.6 Limitações`.
- `2.6 Considerações Finais da Parte II`.
- `2.7 Dificuldades Encontradas na Parte II`.

A seção de limitações foi deliberadamente mantida pendente porque várias limitações ainda serão caracterizadas experimentalmente.

## Próximas issues científicas

A ordem aprovada está registrada nas issues; consulte-as em vez de reconstruir o protocolo manualmente:

- #23 — Executar o Ajuste Fino de Configuração pré-registrado.
- #24 — Caracterizar cobertura e padrões de erro.
- #25 — Produzir curvas de aprendizado controladas.
- #26 — Medir a sensibilidade controlada das predições.
- #27 — Publicar o diagnóstico expandido confirmado.

A discussão conceitual das limitações e a ordem desses diagnósticos já foram definidas na issue #2. Importante: não transformar associações de fazenda, peso, resolução, aquisição, contorno ou tamanho da amostra em afirmações causais.

## Como continuar

1. Fazer `git pull` da `main` antes de qualquer nova redação baseada em resultados.
2. Inspecionar as issues #23–#27 e os novos pacotes em `evidence/confirmed/`.
3. Só escrever resultados que estejam confirmados/revisados; manter marcadores para etapas ainda abertas.
4. Atualizar primeiro metodologia/resultados da etapa recém-concluída e só depois perguntar pela próxima decisão.
5. Antes de fechar limitações e conclusão, exigir o pacote confirmado da issue #27.
6. Manter a bibliografia compartilhada no final do documento e não criar uma segunda lista de referências.
7. Executar `git diff --check` após editar o Markdown.

## Preferência de interação

Trabalhar em modo de decisão incremental: uma decisão por vez, com uma recomendação explícita e aguardando confirmação antes de avançar. Quando um fato puder ser obtido no repositório, investigá-lo diretamente em vez de perguntar ao usuário.

## Suggested skills

- `grilling` — para continuar a tomada de decisões metodológicas/argumentativas uma por vez.
- `handoff` — ao encerrar uma futura sessão longa.
- Skill de PDF apenas quando chegar a etapa de montagem/validação do PDF final; a fonte de redação atual permanece em Markdown.

## Referências operacionais

- Relatório atual: `docs/relatorio-final.md`
- Contexto científico: `CONTEXT.md`
- ADRs e proveniência: `docs/adr/`
- Seleção de atributos confirmada: `evidence/confirmed/feature_selection/v1/`
- Seleção da abordagem confirmada: `evidence/confirmed/approach_selection/v1/`
- Estrutura de issues e triagem: `docs/agents/issue-tracker.md`
