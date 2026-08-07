# Relatório de Diagnóstico: Sensibilidade Controlada das Predições

Total de máscaras avaliadas: 132
Elegíveis para morfologia: 96
Rejeitadas para morfologia: 36
Total de registros de perturbação: 1056
Configurações avaliadas: random_forest

## Resumo de Elegibilidade Morfológica

| Status | Contagem |
| :--- | :---: |
| Elegível | 96 |
| Rejeitada | 36 |

### Motivos de Rejeição

| Motivo | Contagem |
| :--- | :---: |
| contraction_topology_violation | 27 |
| expansion_topology_violation | 4 |
| insufficient_expansion_margin | 5 |

## Notas

- Deltas são sempre perturbado menos original.
- Contração e expansão formam par inseparável: ambas ou nenhuma.
- Perturbações de escala usam ±5% do foreground ao redor do centro.
- Deslocamentos de 5% que cortam o foreground são rejeitados da análise.
- Perturbações sintéticas medem sensibilidade local, não substituem máscaras manuais de referência.
