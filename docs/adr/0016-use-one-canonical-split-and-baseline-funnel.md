# Use one canonical split and a baseline funnel

The controlled experiment uses one predeclared five-fold Divisão Estratificada
Canônica over the 132 Máscaras Válidas. Five repeated divisions crossed with
multiple configurations made the planned evidence computationally
disproportionate to this project's time and compute budget. The outer fold seed
is 42, the isolated inner-validation seed is 43, and one training seed, 44, is
shared by every fold and configuration. This makes the experiment reproducible;
it does not measure sensitivity to other split or training seeds.

One Random Forest baseline and one Rede Densa por Feições baseline produce all
feature evidence. For a removal test, `ΔMAE` is the MAE without the feature minus
the MAE with the complete feature set. Removal can be recommended when at least
one baseline improves by more than 1 kg and the other does not worsen by more
than 1 kg. Harm above 1 kg in either baseline, or neutrality in both, retains
the feature. The recommendation remains subject to human confirmation. The
protocol does not use bootstrap intervals or repeated divisions.

After the Conjunto Compartilhado de Features is frozen, four predeclared
baselines are compared: Random Forest, Rede Densa por Feições, a compact CNN
trained from scratch, and a partially fine-tuned pretrained ResNet-18. Global
pooled OOF MAE is primary. Pooled OOF RMSE, signed bias, R², MAE and bias for B1
and B10, and the training-fold mean predictor are descriptive evidence. Cost
and complexity are not selection criteria, and no combined score is calculated.

A human selects exactly one Abordagem de Maior Potencial. Only its baseline can
advance to adjustment, with a budget of at most three additional variants. The
exact variants are intentionally deferred until the approach is known, but must
be recorded before any tuning result is produced and cannot be expanded after
results are observed. The shared features remain frozen, and the tuned result is
reported separately from the controlled baseline comparison.

All resulting scores are labeled MAE OOF Pós-Seleção. They are development
evidence conditional on this split because the same 132 masks guide feature
selection, approach choice, and tuning. Repeated divisions, isolated training
seed sensitivity, exhaustive configuration grids, and confirmation on new
animals remain outside the current project scope.
