---
status: superseded by ADR-0016
---

# Freeze the shared feature contract before final configuration tuning

Feature evidence is produced from five predeclared five-fold Divisões Estratificadas over the 132 Máscaras Válidas. Redundancy motivates controlled removal tests but does not remove a feature automatically. Evidence from the representative Random Forest and Rede Densa por Feições configurations supports one human-confirmed Conjunto Compartilhado de Features, rather than a different subset inside each reserved fold.

Feature columns are standardized with means and standard deviations fitted only on the training partition of each fold. The resulting transformation is applied to that partition and its reserved animals, and the same standardized representation is supplied to Random Forest and Rede Densa por Feições. This gives the neural network comparable numeric scales without allowing a reserved animal to influence preprocessing for its prediction.

After feature evidence is reviewed, the resulting shared set is frozen and supplied identically to both model classes. The four coarse configurations of each class are then compared again using that frozen input contract. A separate AI-drafted Relatório de Seleção das Configurações de Maior Potencial presents MAE as the primary evidence and stability across Divisões Estratificadas as secondary but substantive evidence; the criteria need not be collapsed into one automatic score. A grilling session reviews the evidence and humanly confirms one Configuração de Maior Potencial per class. Only those two configurations undergo Ajuste Fino de Configuração to seek lower MAE and greater stability. This later stage cannot add, remove, or reopen features.

Because feature and configuration choices are guided by OOF results from the same 132 masks, the resulting metric is MAE OOF Pós-Seleção. OOF prediction still prevents an animal from training the instance that predicts it, but it does not make the final score an independent estimate after the sample has guided the procedure. Independent confirmation requires new animals, and the selection report must state this dataset limitation.
