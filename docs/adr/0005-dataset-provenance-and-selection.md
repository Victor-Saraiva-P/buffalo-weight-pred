# Dataset provenance and selection

The 132 Máscaras Binarizadas used by the current Etapa de Predição de Peso are
the manually validated `ok` subset of the 387 images evaluated in the upstream
PIBIC segmentation study. The remaining images from that study were not
carried into this phase because the current phase requires a valid weight
label, the manual `ok` validation, and follows the dataset scope communicated
by the project supervisor.

The supervisor's email describes three source groups:

- `Fotos 1 - 224`: weights for animals from the three farms are in the Excel
  file. These are the first project photographs and do not follow distance or
  camera-position recommendations, so they are considered less reliable for
  those acquisition criteria.
- `Fotos 2 - 88`: the weight is encoded in the animal-photo folder name, and
  the recommended distance and camera position are followed.
- `Fotos 3 - 47`: the weight is encoded in the animal-photo folder name, and
  the recommended distance and camera position are followed.

The email is provenance for acquisition quality and label origin, not a
replacement for the manual-ground-truth evaluation in the PIBIC report.
Segmentation claims must continue to use
`Relatório PIBIC_ PROCESSAMENTO DE IMAGEM PARA ESTIMATIVA DO PESO DE BÚFALO .pdf`.

**Considered Options**: Treating the 132 masks as an independent dataset was
rejected because they originate within the 387-image upstream study. Treating
all 387 images as interchangeable training examples was rejected because this
phase requires valid weight labels and the acquisition groups have different
reliability characteristics.

**Consequences**: Analyses must distinguish source-image coverage from current
training inclusion. The exact mapping of each current mask to source group,
ground truth, `ok` validation, and supervisor-provided weight source should be
preserved when available. The counts in the email (`224 + 88 + 47 = 359`) do
not equal the 387 images reported in the PIBIC document, so this discrepancy
remains an open provenance issue rather than being silently resolved.
