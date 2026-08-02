# Treat segmentation as a completed upstream result

The final report will build on the completed PIBIC segmentation study, but this repository will not reproduce that study's model and binarization experiments. It will preserve only the provenance needed to identify the selected binary masks and will reproduce the subsequent weight-prediction evidence, keeping the repository focused without weakening the auditability of its new claims.

Feature extraction consumes each curated Máscara Binarizada exactly as validated with tag `ok`. It does not fill holes, discard disconnected components, or apply another morphological cleanup, because doing so would create a new unvalidated segmentation stage. Perimeter, Feret, and convex-hull measurements may therefore expose imperfections in the upstream predictive mask, which remain part of the prediction input and its documented limitations.
