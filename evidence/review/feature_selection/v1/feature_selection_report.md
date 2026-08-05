# Minuta de Seleção do Conjunto Compartilhado de Features

Este documento organiza evidências OOF pós-seleção para revisão humana. Nenhuma recomendação abaixo constitui uma decisão automática.

A tabela detalhada `feature_predictive_evidence.csv` contém 3816 resultados; `feature_redundancy.csv` contém os 325 pares. Os resultados por fold e repetição permanecem nesses artefatos auditáveis.

## Desempenho isolado

| Feature | MAE RF (kg) | MAE rede densa (kg) |
|---|---:|---:|
| `area` | 66.00 | 68.82 |
| `area_major_axis_product` | 63.67 | 67.78 |
| `area_power_1_5` | 65.99 | 68.34 |
| `aspect_ratio` | 94.06 | 105.74 |
| `bbox_area` | 75.48 | 68.57 |
| `bbox_height` | 80.72 | 74.72 |
| `bbox_width` | 71.65 | 72.03 |
| `center_to_end_occupancy_ratio` | 127.90 | 131.94 |
| `center_vertical_occupancy` | 73.39 | 74.62 |
| `centroid_x_offset` | 138.09 | 119.81 |
| `centroid_y_ratio` | 129.13 | 122.81 |
| `circularity` | 123.86 | 117.43 |
| `convex_area` | 71.85 | 69.98 |
| `convexity` | 138.15 | 123.10 |
| `end_vertical_occupancy_max` | 75.75 | 73.59 |
| `end_vertical_occupancy_min` | 78.87 | 86.71 |
| `equivalent_diameter` | 66.01 | 68.70 |
| `extent` | 105.63 | 103.90 |
| `feret_diameter` | 74.16 | 73.86 |
| `hu_moment_1` | 120.62 | 118.96 |
| `hu_moment_2` | 99.83 | 108.34 |
| `major_axis_length` | 65.83 | 69.38 |
| `minor_axis_length` | 80.53 | 74.01 |
| `perimeter` | 70.39 | 74.25 |
| `roundness` | 123.04 | 114.55 |
| `solidity` | 101.32 | 104.89 |

## Redundância estrutural e observada

Há 29 pares com relação estrutural declarada. O mapa completo está em `redundancy_heatmap.png`; todos os pares também permanecem no CSV.

| Par | Relação estrutural | Pearson | Spearman |
|---|---|---:|---:|
| `area` / `equivalent_diameter` | `area_bijection` | 0.994 | 1.000 |
| `area` / `area_power_1_5` | `area_bijection` | 0.994 | 1.000 |
| `equivalent_diameter` / `area_power_1_5` | `area_bijection` | 0.975 | 1.000 |
| `area_power_1_5` / `area_major_axis_product` | `none` | 0.999 | 0.997 |
| `area` / `area_major_axis_product` | `area_major_axis_product` | 0.992 | 0.997 |
| `equivalent_diameter` / `area_major_axis_product` | `none` | 0.973 | 0.997 |
| `convex_area` / `area_major_axis_product` | `none` | 0.985 | 0.994 |
| `bbox_area` / `convex_area` | `none` | 0.993 | 0.990 |
| `major_axis_length` / `area_major_axis_product` | `area_major_axis_product` | 0.968 | 0.992 |
| `area` / `convex_area` | `convex_solidity` | 0.991 | 0.992 |
| `equivalent_diameter` / `convex_area` | `none` | 0.986 | 0.992 |
| `convex_area` / `area_power_1_5` | `none` | 0.984 | 0.992 |
| `equivalent_diameter` / `major_axis_length` | `none` | 0.987 | 0.983 |
| `convex_area` / `major_axis_length` | `none` | 0.986 | 0.987 |
| `bbox_area` / `area_major_axis_product` | `none` | 0.973 | 0.984 |
| `area` / `major_axis_length` | `area_major_axis_product|ellipse_roundness` | 0.981 | 0.983 |
| `major_axis_length` / `area_power_1_5` | `none` | 0.963 | 0.983 |
| `bbox_width` / `feret_diameter` | `none` | 0.982 | 0.974 |
| `area` / `bbox_area` | `bbox_extent` | 0.981 | 0.980 |
| `bbox_area` / `major_axis_length` | `none` | 0.981 | 0.978 |
| `equivalent_diameter` / `bbox_area` | `none` | 0.979 | 0.980 |
| `bbox_area` / `area_power_1_5` | `none` | 0.972 | 0.980 |
| `equivalent_diameter` / `minor_axis_length` | `none` | 0.975 | 0.970 |
| `area` / `center_vertical_occupancy` | `none` | 0.936 | 0.974 |
| `equivalent_diameter` / `center_vertical_occupancy` | `none` | 0.943 | 0.974 |
| `area_power_1_5` / `center_vertical_occupancy` | `none` | 0.918 | 0.974 |
| `area` / `minor_axis_length` | `none` | 0.964 | 0.970 |
| `minor_axis_length` / `area_power_1_5` | `none` | 0.942 | 0.970 |
| `perimeter` / `equivalent_diameter` | `none` | 0.970 | 0.962 |
| `convex_area` / `minor_axis_length` | `none` | 0.964 | 0.970 |
| `bbox_area` / `feret_diameter` | `none` | 0.969 | 0.961 |
| `bbox_width` / `major_axis_length` | `none` | 0.968 | 0.958 |
| `major_axis_length` / `feret_diameter` | `none` | 0.967 | 0.948 |
| `bbox_width` / `bbox_area` | `bbox_area_product` | 0.966 | 0.956 |
| `minor_axis_length` / `center_vertical_occupancy` | `none` | 0.936 | 0.964 |
| `bbox_height` / `minor_axis_length` | `none` | 0.964 | 0.957 |
| `area_major_axis_product` / `center_vertical_occupancy` | `none` | 0.906 | 0.964 |
| `area` / `perimeter` | `area_contour_circularity` | 0.963 | 0.962 |
| `perimeter` / `area_power_1_5` | `none` | 0.944 | 0.962 |
| `minor_axis_length` / `area_major_axis_product` | `none` | 0.933 | 0.961 |
| `perimeter` / `bbox_area` | `none` | 0.960 | 0.960 |
| `bbox_area` / `minor_axis_length` | `none` | 0.958 | 0.960 |
| `perimeter` / `convex_area` | `none` | 0.960 | 0.957 |
| `perimeter` / `area_major_axis_product` | `none` | 0.940 | 0.959 |
| `convex_area` / `feret_diameter` | `none` | 0.958 | 0.950 |
| `perimeter` / `minor_axis_length` | `none` | 0.958 | 0.951 |
| `bbox_width` / `convex_area` | `none` | 0.957 | 0.949 |
| `roundness` / `hu_moment_2` | `none` | -0.856 | -0.955 |
| `convex_area` / `center_vertical_occupancy` | `none` | 0.901 | 0.954 |
| `perimeter` / `major_axis_length` | `none` | 0.954 | 0.943 |
| `bbox_height` / `bbox_area` | `bbox_area_product` | 0.947 | 0.950 |
| `equivalent_diameter` / `bbox_height` | `none` | 0.947 | 0.931 |
| `perimeter` / `center_vertical_occupancy` | `none` | 0.911 | 0.944 |
| `bbox_width` / `area_major_axis_product` | `none` | 0.925 | 0.944 |
| `major_axis_length` / `minor_axis_length` | `none` | 0.943 | 0.932 |
| `equivalent_diameter` / `bbox_width` | `none` | 0.939 | 0.929 |
| `minor_axis_length` / `end_vertical_occupancy_max` | `none` | 0.939 | 0.938 |
| `bbox_area` / `center_vertical_occupancy` | `none` | 0.881 | 0.938 |
| `equivalent_diameter` / `feret_diameter` | `none` | 0.938 | 0.921 |
| `bbox_height` / `convex_area` | `none` | 0.937 | 0.937 |
| `equivalent_diameter` / `end_vertical_occupancy_max` | `none` | 0.937 | 0.937 |
| `area` / `end_vertical_occupancy_max` | `none` | 0.930 | 0.937 |
| `area_power_1_5` / `end_vertical_occupancy_max` | `none` | 0.911 | 0.937 |
| `feret_diameter` / `area_major_axis_product` | `none` | 0.919 | 0.936 |
| `major_axis_length` / `center_vertical_occupancy` | `none` | 0.889 | 0.934 |
| `area` / `bbox_width` | `none` | 0.934 | 0.929 |
| `area` / `bbox_height` | `none` | 0.932 | 0.931 |
| `bbox_height` / `area_power_1_5` | `none` | 0.908 | 0.931 |
| `area` / `feret_diameter` | `none` | 0.930 | 0.921 |
| `bbox_width` / `area_power_1_5` | `none` | 0.918 | 0.929 |
| `area_major_axis_product` / `end_vertical_occupancy_max` | `none` | 0.903 | 0.927 |
| `perimeter` / `bbox_height` | `none` | 0.927 | 0.914 |
| `bbox_height` / `end_vertical_occupancy_max` | `none` | 0.927 | 0.927 |
| `convex_area` / `end_vertical_occupancy_max` | `none` | 0.919 | 0.925 |
| `bbox_height` / `area_major_axis_product` | `none` | 0.901 | 0.925 |
| `perimeter` / `bbox_width` | `none` | 0.924 | 0.911 |
| `bbox_height` / `major_axis_length` | `none` | 0.922 | 0.902 |
| `feret_diameter` / `area_power_1_5` | `none` | 0.911 | 0.921 |
| `perimeter` / `feret_diameter` | `none` | 0.918 | 0.901 |
| `bbox_height` / `center_vertical_occupancy` | `none` | 0.908 | 0.914 |
| `bbox_area` / `end_vertical_occupancy_max` | `none` | 0.905 | 0.909 |
| `center_vertical_occupancy` / `end_vertical_occupancy_max` | `vertical_occupancy_ratio` | 0.880 | 0.908 |
| `major_axis_length` / `end_vertical_occupancy_max` | `none` | 0.908 | 0.902 |
| `perimeter` / `end_vertical_occupancy_max` | `none` | 0.903 | 0.895 |
| `minor_axis_length` / `feret_diameter` | `none` | 0.903 | 0.886 |
| `bbox_width` / `minor_axis_length` | `none` | 0.898 | 0.880 |
| `bbox_height` / `feret_diameter` | `none` | 0.897 | 0.878 |
| `center_vertical_occupancy` / `end_vertical_occupancy_min` | `vertical_occupancy_ratio` | 0.876 | 0.895 |
| `minor_axis_length` / `end_vertical_occupancy_min` | `none` | 0.833 | 0.889 |
| `area` / `end_vertical_occupancy_min` | `none` | 0.822 | 0.885 |
| `equivalent_diameter` / `end_vertical_occupancy_min` | `none` | 0.826 | 0.885 |
| `area_power_1_5` / `end_vertical_occupancy_min` | `none` | 0.806 | 0.885 |
| `bbox_height` / `end_vertical_occupancy_min` | `none` | 0.845 | 0.880 |
| `area_major_axis_product` / `end_vertical_occupancy_min` | `none` | 0.793 | 0.874 |
| `convex_area` / `end_vertical_occupancy_min` | `none` | 0.788 | 0.867 |
| `bbox_width` / `center_vertical_occupancy` | `none` | 0.789 | 0.866 |
| `bbox_width` / `bbox_height` | `bbox_area_product|bbox_aspect_ratio` | 0.859 | 0.836 |
| `hu_moment_1` / `hu_moment_2` | `none` | 0.858 | 0.583 |
| `feret_diameter` / `end_vertical_occupancy_max` | `none` | 0.855 | 0.837 |
| `feret_diameter` / `center_vertical_occupancy` | `none` | 0.789 | 0.853 |
| `bbox_area` / `end_vertical_occupancy_min` | `none` | 0.763 | 0.846 |
| `end_vertical_occupancy_min` / `end_vertical_occupancy_max` | `vertical_occupancy_ratio` | 0.789 | 0.845 |
| `solidity` / `extent` | `none` | 0.842 | 0.676 |
| `major_axis_length` / `end_vertical_occupancy_min` | `none` | 0.771 | 0.840 |
| `perimeter` / `end_vertical_occupancy_min` | `none` | 0.775 | 0.837 |
| `bbox_width` / `end_vertical_occupancy_max` | `none` | 0.836 | 0.818 |
| `circularity` / `convexity` | `none` | 0.540 | 0.830 |
| `solidity` / `hu_moment_1` | `none` | -0.810 | -0.619 |
| `bbox_width` / `aspect_ratio` | `bbox_aspect_ratio` | 0.691 | 0.800 |
| `aspect_ratio` / `hu_moment_2` | `none` | 0.783 | 0.749 |
| `roundness` / `hu_moment_1` | `none` | -0.751 | -0.783 |
| `feret_diameter` / `end_vertical_occupancy_min` | `none` | 0.690 | 0.781 |
| `bbox_width` / `end_vertical_occupancy_min` | `none` | 0.639 | 0.745 |
| `extent` / `hu_moment_1` | `none` | -0.730 | -0.466 |
| `aspect_ratio` / `major_axis_length` | `none` | 0.527 | 0.722 |
| `center_to_end_occupancy_ratio` / `centroid_x_offset` | `none` | -0.713 | -0.375 |
| `aspect_ratio` / `feret_diameter` | `none` | 0.620 | 0.711 |
| `aspect_ratio` / `area_major_axis_product` | `none` | 0.446 | 0.678 |
| `aspect_ratio` / `convex_area` | `none` | 0.481 | 0.663 |
| `convexity` / `hu_moment_1` | `none` | 0.661 | 0.383 |
| `aspect_ratio` / `roundness` | `none` | -0.656 | -0.624 |
| `area` / `aspect_ratio` | `none` | 0.429 | 0.653 |
| `equivalent_diameter` / `aspect_ratio` | `none` | 0.431 | 0.653 |
| `aspect_ratio` / `area_power_1_5` | `none` | 0.422 | 0.653 |
| `bbox_width` / `hu_moment_2` | `none` | 0.512 | 0.650 |
| `hu_moment_1` / `centroid_x_offset` | `none` | 0.649 | 0.176 |
| `bbox_area` / `aspect_ratio` | `none` | 0.494 | 0.648 |
| `solidity` / `center_vertical_occupancy` | `none` | 0.642 | 0.605 |
| `perimeter` / `aspect_ratio` | `none` | 0.435 | 0.640 |
| `feret_diameter` / `hu_moment_2` | `none` | 0.510 | 0.639 |
| `major_axis_length` / `hu_moment_2` | `none` | 0.397 | 0.621 |
| `solidity` / `centroid_x_offset` | `none` | -0.619 | -0.096 |
| `solidity` / `convexity` | `none` | -0.616 | -0.273 |
| `extent` / `centroid_x_offset` | `none` | -0.595 | -0.145 |
| `hu_moment_1` / `center_to_end_occupancy_ratio` | `none` | -0.588 | -0.160 |
| `convexity` / `center_to_end_occupancy_ratio` | `none` | -0.583 | -0.239 |
| `convexity` / `hu_moment_2` | `none` | 0.582 | 0.298 |
| `solidity` / `end_vertical_occupancy_min` | `none` | 0.578 | 0.553 |
| `aspect_ratio` / `center_vertical_occupancy` | `none` | 0.174 | 0.576 |
| `extent` / `center_vertical_occupancy` | `none` | 0.576 | 0.545 |
| `aspect_ratio` / `hu_moment_1` | `none` | 0.573 | 0.167 |
| `solidity` / `center_to_end_occupancy_ratio` | `none` | 0.568 | 0.071 |
| `end_vertical_occupancy_min` / `centroid_x_offset` | `none` | -0.565 | -0.472 |
| `extent` / `end_vertical_occupancy_min` | `none` | 0.559 | 0.528 |
| `area` / `solidity` | `convex_solidity` | 0.450 | 0.552 |
| `solidity` / `equivalent_diameter` | `none` | 0.448 | 0.552 |
| `solidity` / `area_power_1_5` | `none` | 0.443 | 0.552 |
| `solidity` / `end_vertical_occupancy_max` | `none` | 0.455 | 0.551 |
| `bbox_area` / `hu_moment_2` | `none` | 0.354 | 0.541 |
| `aspect_ratio` / `minor_axis_length` | `none` | 0.347 | 0.537 |
| `hu_moment_2` / `area_major_axis_product` | `none` | 0.283 | 0.537 |
| `solidity` / `area_major_axis_product` | `none` | 0.422 | 0.533 |
| `perimeter` / `solidity` | `none` | 0.412 | 0.532 |
| `extent` / `convexity` | `none` | -0.531 | -0.164 |
| `convex_area` / `hu_moment_2` | `none` | 0.325 | 0.531 |
| `convexity` / `centroid_x_offset` | `none` | 0.529 | -0.043 |
| `extent` / `end_vertical_occupancy_max` | `none` | 0.421 | 0.518 |
| `area` / `extent` | `bbox_extent` | 0.412 | 0.516 |
| `equivalent_diameter` / `extent` | `none` | 0.403 | 0.516 |
| `extent` / `area_power_1_5` | `none` | 0.413 | 0.516 |
| `convexity` / `roundness` | `none` | -0.513 | -0.348 |
| `hu_moment_2` / `centroid_x_offset` | `none` | 0.508 | -0.025 |
| `solidity` / `minor_axis_length` | `none` | 0.414 | 0.502 |
| `extent` / `area_major_axis_product` | `none` | 0.398 | 0.500 |
| `bbox_width` / `roundness` | `none` | -0.438 | -0.496 |
| `area` / `hu_moment_2` | `none` | 0.254 | 0.493 |
| `equivalent_diameter` / `hu_moment_2` | `none` | 0.257 | 0.493 |
| `hu_moment_2` / `area_power_1_5` | `none` | 0.249 | 0.493 |
| `solidity` / `major_axis_length` | `none` | 0.347 | 0.492 |
| `aspect_ratio` / `extent` | `none` | -0.163 | 0.491 |
| `roundness` / `feret_diameter` | `none` | -0.440 | -0.491 |
| `solidity` / `hu_moment_2` | `none` | -0.487 | 0.038 |
| `aspect_ratio` / `centroid_x_offset` | `none` | 0.480 | -0.091 |
| `aspect_ratio` / `end_vertical_occupancy_max` | `none` | 0.286 | 0.474 |
| `extent` / `major_axis_length` | `none` | 0.316 | 0.473 |
| `solidity` / `convex_area` | `convex_solidity` | 0.335 | 0.471 |
| `extent` / `center_to_end_occupancy_ratio` | `none` | 0.467 | 0.035 |
| `major_axis_length` / `roundness` | `ellipse_roundness` | -0.380 | -0.466 |
| `solidity` / `bbox_area` | `none` | 0.307 | 0.465 |
| `extent` / `minor_axis_length` | `none` | 0.366 | 0.465 |
| `extent` / `convex_area` | `none` | 0.317 | 0.462 |
| `perimeter` / `hu_moment_2` | `none` | 0.249 | 0.455 |
| `extent` / `hu_moment_2` | `none` | -0.453 | 0.090 |
| `perimeter` / `extent` | `none` | 0.340 | 0.453 |
| `center_vertical_occupancy` / `centroid_x_offset` | `none` | -0.448 | -0.262 |
| `solidity` / `bbox_width` | `none` | 0.194 | 0.439 |
| `solidity` / `roundness` | `none` | 0.432 | 0.180 |
| `hu_moment_2` / `center_to_end_occupancy_ratio` | `none` | -0.429 | 0.035 |
| `hu_moment_1` / `center_vertical_occupancy` | `none` | -0.427 | -0.282 |
| `center_vertical_occupancy` / `center_to_end_occupancy_ratio` | `vertical_occupancy_ratio` | 0.426 | 0.232 |
| `solidity` / `bbox_height` | `none` | 0.372 | 0.421 |
| `aspect_ratio` / `end_vertical_occupancy_min` | `none` | 0.009 | 0.420 |
| `aspect_ratio` / `convexity` | `none` | 0.416 | 0.004 |
| `bbox_height` / `aspect_ratio` | `bbox_aspect_ratio` | 0.240 | 0.413 |
| `hu_moment_1` / `end_vertical_occupancy_min` | `none` | -0.407 | -0.298 |
| `bbox_width` / `extent` | `none` | 0.156 | 0.395 |
| `bbox_area` / `roundness` | `none` | -0.305 | -0.387 |
| `bbox_area` / `extent` | `bbox_extent` | 0.236 | 0.383 |
| `solidity` / `aspect_ratio` | `none` | -0.261 | 0.379 |
| `solidity` / `feret_diameter` | `none` | 0.161 | 0.379 |
| `bbox_height` / `hu_moment_2` | `none` | 0.185 | 0.377 |
| `convex_area` / `roundness` | `none` | -0.290 | -0.375 |
| `roundness` / `area_major_axis_product` | `none` | -0.257 | -0.371 |
| `hu_moment_2` / `center_vertical_occupancy` | `none` | -0.010 | 0.369 |
| `aspect_ratio` / `centroid_y_ratio` | `none` | -0.291 | -0.367 |
| `hu_moment_2` / `end_vertical_occupancy_max` | `none` | 0.159 | 0.360 |
| `extent` / `roundness` | `none` | 0.356 | 0.089 |
| `minor_axis_length` / `hu_moment_2` | `none` | 0.159 | 0.353 |
| `extent` / `feret_diameter` | `none` | 0.119 | 0.337 |
| `area` / `roundness` | `ellipse_roundness` | -0.227 | -0.324 |
| `equivalent_diameter` / `roundness` | `none` | -0.233 | -0.324 |
| `roundness` / `area_power_1_5` | `none` | -0.219 | -0.324 |
| `roundness` / `centroid_x_offset` | `none` | -0.324 | -0.043 |
| `circularity` / `end_vertical_occupancy_min` | `none` | 0.315 | 0.299 |
| `roundness` / `center_to_end_occupancy_ratio` | `none` | 0.301 | 0.024 |
| `circularity` / `extent` | `none` | 0.300 | 0.256 |
| `bbox_height` / `extent` | `none` | 0.246 | 0.289 |
| `perimeter` / `roundness` | `none` | -0.202 | -0.289 |
| `convexity` / `center_vertical_occupancy` | `none` | -0.287 | -0.099 |
| `hu_moment_2` / `end_vertical_occupancy_min` | `none` | -0.063 | 0.284 |
| `roundness` / `centroid_y_ratio` | `none` | 0.211 | 0.277 |
| `aspect_ratio` / `center_to_end_occupancy_ratio` | `none` | -0.255 | 0.272 |
| `circularity` / `center_vertical_occupancy` | `none` | 0.256 | 0.194 |
| `hu_moment_1` / `centroid_y_ratio` | `none` | -0.102 | -0.256 |
| `circularity` / `major_axis_length` | `none` | 0.255 | 0.229 |
| `circularity` / `end_vertical_occupancy_max` | `none` | 0.252 | 0.240 |
| `circularity` / `equivalent_diameter` | `none` | 0.248 | 0.217 |
| `area` / `circularity` | `area_contour_circularity` | 0.244 | 0.217 |
| `circularity` / `area_major_axis_product` | `none` | 0.243 | 0.219 |
| `bbox_height` / `roundness` | `none` | -0.151 | -0.242 |
| `perimeter` / `convexity` | `convexity_ratio` | -0.214 | -0.241 |
| `bbox_width` / `centroid_y_ratio` | `none` | -0.240 | -0.239 |
| `circularity` / `area_power_1_5` | `none` | 0.237 | 0.217 |
| `hu_moment_2` / `centroid_y_ratio` | `none` | -0.126 | -0.233 |
| `major_axis_length` / `centroid_y_ratio` | `none` | -0.219 | -0.232 |
| `circularity` / `convex_area` | `none` | 0.227 | 0.203 |
| `hu_moment_1` / `end_vertical_occupancy_max` | `none` | -0.218 | -0.214 |
| `convex_area` / `centroid_y_ratio` | `none` | -0.196 | -0.213 |
| `solidity` / `circularity` | `none` | 0.212 | 0.134 |
| `bbox_height` / `centroid_x_offset` | `none` | -0.211 | -0.196 |
| `area_major_axis_product` / `centroid_y_ratio` | `none` | -0.172 | -0.210 |
| `area` / `centroid_x_offset` | `none` | -0.208 | -0.207 |
| `equivalent_diameter` / `centroid_x_offset` | `none` | -0.207 | -0.207 |
| `area_power_1_5` / `centroid_x_offset` | `none` | -0.204 | -0.207 |
| `perimeter` / `centroid_x_offset` | `none` | -0.175 | -0.207 |
| `circularity` / `bbox_height` | `none` | 0.207 | 0.177 |
| `minor_axis_length` / `roundness` | `none` | -0.108 | -0.207 |
| `end_vertical_occupancy_min` / `center_to_end_occupancy_ratio` | `vertical_occupancy_ratio` | 0.206 | 0.042 |
| `roundness` / `end_vertical_occupancy_max` | `none` | -0.125 | -0.204 |
| `circularity` / `hu_moment_2` | `none` | 0.045 | 0.204 |
| `minor_axis_length` / `centroid_x_offset` | `none` | -0.195 | -0.204 |
| `area` / `centroid_y_ratio` | `none` | -0.180 | -0.204 |
| `equivalent_diameter` / `centroid_y_ratio` | `none` | -0.196 | -0.204 |
| `area_power_1_5` / `centroid_y_ratio` | `none` | -0.163 | -0.204 |
| `area_major_axis_product` / `centroid_x_offset` | `none` | -0.191 | -0.201 |
| `minor_axis_length` / `centroid_y_ratio` | `none` | -0.200 | -0.180 |
| `convexity` / `end_vertical_occupancy_min` | `none` | -0.200 | 0.034 |
| `minor_axis_length` / `hu_moment_1` | `none` | -0.200 | -0.200 |
| `circularity` / `minor_axis_length` | `none` | 0.197 | 0.163 |
| `perimeter` / `centroid_y_ratio` | `none` | -0.196 | -0.178 |
| `bbox_area` / `centroid_y_ratio` | `none` | -0.184 | -0.196 |
| `convex_area` / `centroid_x_offset` | `none` | -0.135 | -0.195 |
| `circularity` / `bbox_area` | `none` | 0.194 | 0.162 |
| `circularity` / `feret_diameter` | `none` | 0.191 | 0.156 |
| `roundness` / `center_vertical_occupancy` | `none` | 0.009 | -0.188 |
| `major_axis_length` / `centroid_x_offset` | `none` | -0.132 | -0.184 |
| `bbox_area` / `centroid_x_offset` | `none` | -0.099 | -0.184 |
| `perimeter` / `center_to_end_occupancy_ratio` | `none` | 0.181 | 0.184 |
| `circularity` / `centroid_x_offset` | `none` | -0.182 | -0.075 |
| `circularity` / `roundness` | `none` | -0.155 | -0.180 |
| `circularity` / `center_to_end_occupancy_ratio` | `none` | 0.026 | -0.178 |
| `equivalent_diameter` / `center_to_end_occupancy_ratio` | `none` | 0.177 | 0.129 |
| `perimeter` / `hu_moment_1` | `none` | -0.152 | -0.175 |
| `circularity` / `bbox_width` | `none` | 0.173 | 0.145 |
| `bbox_height` / `hu_moment_1` | `none` | -0.171 | -0.140 |
| `area` / `center_to_end_occupancy_ratio` | `none` | 0.169 | 0.129 |
| `area` / `hu_moment_1` | `none` | -0.168 | -0.155 |
| `equivalent_diameter` / `hu_moment_1` | `none` | -0.167 | -0.155 |
| `convexity` / `feret_diameter` | `none` | 0.166 | 0.082 |
| `hu_moment_1` / `area_power_1_5` | `none` | -0.165 | -0.155 |
| `feret_diameter` / `centroid_x_offset` | `none` | 0.025 | -0.165 |
| `feret_diameter` / `centroid_y_ratio` | `none` | -0.163 | -0.161 |
| `area_power_1_5` / `center_to_end_occupancy_ratio` | `none` | 0.160 | 0.129 |
| `minor_axis_length` / `center_to_end_occupancy_ratio` | `none` | 0.157 | 0.108 |
| `bbox_width` / `center_to_end_occupancy_ratio` | `none` | 0.014 | 0.150 |
| `area_major_axis_product` / `center_to_end_occupancy_ratio` | `none` | 0.149 | 0.125 |
| `center_vertical_occupancy` / `centroid_y_ratio` | `none` | -0.147 | -0.135 |
| `center_to_end_occupancy_ratio` / `centroid_y_ratio` | `none` | -0.144 | -0.079 |
| `bbox_width` / `centroid_x_offset` | `none` | 0.045 | -0.143 |
| `end_vertical_occupancy_max` / `centroid_y_ratio` | `none` | -0.133 | -0.132 |
| `hu_moment_1` / `area_major_axis_product` | `none` | -0.132 | -0.112 |
| `circularity` / `aspect_ratio` | `none` | 0.022 | 0.129 |
| `bbox_width` / `convexity` | `none` | 0.129 | 0.030 |
| `feret_diameter` / `hu_moment_1` | `none` | 0.128 | 0.037 |
| `bbox_width` / `hu_moment_1` | `none` | 0.122 | 0.017 |
| `bbox_area` / `center_to_end_occupancy_ratio` | `none` | 0.085 | 0.121 |
| `bbox_height` / `center_to_end_occupancy_ratio` | `none` | 0.118 | 0.047 |
| `major_axis_length` / `center_to_end_occupancy_ratio` | `none` | 0.110 | 0.115 |
| `roundness` / `end_vertical_occupancy_min` | `none` | 0.048 | -0.113 |
| `convex_area` / `center_to_end_occupancy_ratio` | `none` | 0.103 | 0.113 |
| `bbox_height` / `centroid_y_ratio` | `none` | -0.108 | -0.100 |
| `centroid_x_offset` / `centroid_y_ratio` | `none` | -0.019 | -0.107 |
| `convexity` / `minor_axis_length` | `none` | -0.101 | -0.080 |
| `circularity` / `hu_moment_1` | `none` | -0.098 | 0.073 |
| `end_vertical_occupancy_max` / `center_to_end_occupancy_ratio` | `vertical_occupancy_ratio` | -0.002 | -0.098 |
| `convex_area` / `hu_moment_1` | `none` | -0.072 | -0.086 |
| `feret_diameter` / `center_to_end_occupancy_ratio` | `none` | -0.037 | 0.085 |
| `convexity` / `area_power_1_5` | `none` | -0.082 | -0.036 |
| `extent` / `centroid_y_ratio` | `none` | -0.078 | -0.082 |
| `area` / `convexity` | `none` | -0.081 | -0.036 |
| `convexity` / `end_vertical_occupancy_max` | `none` | -0.079 | -0.033 |
| `end_vertical_occupancy_max` / `centroid_x_offset` | `none` | -0.077 | -0.016 |
| `equivalent_diameter` / `convexity` | `none` | -0.076 | -0.036 |
| `bbox_area` / `hu_moment_1` | `none` | -0.037 | -0.070 |
| `convexity` / `area_major_axis_product` | `none` | -0.061 | -0.014 |
| `bbox_height` / `convexity` | `none` | -0.055 | 0.004 |
| `circularity` / `centroid_y_ratio` | `none` | -0.041 | -0.048 |
| `solidity` / `centroid_y_ratio` | `none` | 0.024 | 0.044 |
| `convexity` / `centroid_y_ratio` | `none` | 0.043 | 0.032 |
| `end_vertical_occupancy_min` / `centroid_y_ratio` | `none` | -0.009 | -0.026 |
| `convexity` / `major_axis_length` | `none` | 0.019 | 0.026 |
| `major_axis_length` / `hu_moment_1` | `none` | -0.023 | -0.015 |
| `perimeter` / `circularity` | `area_contour_circularity` | 0.012 | -0.022 |
| `bbox_area` / `convexity` | `none` | 0.018 | -0.009 |
| `convex_area` / `convexity` | `none` | -0.003 | 0.003 |

## Efeitos de permutação

Cada média resume dez permutações determinísticas OOF; repetições e folds estão em `feature_predictive_evidence.csv` e `permutation_effects.png`.

| Feature | Δ MAE RF (kg) | Δ MAE rede densa (kg) |
|---|---:|---:|
| `area` | 3.51 | 1.77 |
| `area_major_axis_product` | 3.19 | -0.12 |
| `area_power_1_5` | 4.21 | 0.29 |
| `aspect_ratio` | 2.84 | 3.75 |
| `bbox_area` | 0.63 | 1.58 |
| `bbox_height` | 0.41 | 2.04 |
| `bbox_width` | 0.52 | 6.23 |
| `center_to_end_occupancy_ratio` | 0.38 | 1.38 |
| `center_vertical_occupancy` | 7.94 | 8.08 |
| `centroid_x_offset` | -0.00 | -0.28 |
| `centroid_y_ratio` | 1.19 | 2.96 |
| `circularity` | 0.19 | 2.91 |
| `convex_area` | 1.03 | 1.04 |
| `convexity` | -0.12 | -0.21 |
| `end_vertical_occupancy_max` | 1.95 | 8.83 |
| `end_vertical_occupancy_min` | 0.12 | 6.44 |
| `equivalent_diameter` | 4.45 | -0.73 |
| `extent` | 0.78 | 5.37 |
| `feret_diameter` | 0.68 | 10.15 |
| `hu_moment_1` | 0.98 | 3.72 |
| `hu_moment_2` | 0.53 | 9.07 |
| `major_axis_length` | 4.60 | 8.22 |
| `minor_axis_length` | 0.13 | -1.80 |
| `perimeter` | 0.86 | 3.30 |
| `roundness` | 0.05 | 0.87 |
| `solidity` | 8.05 | 7.32 |

## Testes de retirada

O mapa completo está em `removal_heatmap.png`.

| Feature ou grupo | Δ MAE RF (kg) | Δ MAE rede densa (kg) | Recomendação provisória |
|---|---:|---:|---|
| `area` | 0.18 | 2.61 | `retain_harm_veto` |
| `perimeter` | -0.27 | 3.77 | `retain_harm_veto` |
| `solidity` | 3.91 | 5.53 | `retain_harm_veto` |
| `circularity` | 0.14 | 4.48 | `retain_harm_veto` |
| `equivalent_diameter` | 0.04 | 4.63 | `retain_harm_veto` |
| `bbox_width` | -0.01 | 6.86 | `retain_harm_veto` |
| `bbox_height` | -0.14 | 8.21 | `retain_harm_veto` |
| `bbox_area` | -0.03 | 7.71 | `retain_harm_veto` |
| `aspect_ratio` | -0.12 | 5.86 | `retain_harm_veto` |
| `extent` | 0.31 | 10.71 | `retain_harm_veto` |
| `convex_area` | 0.24 | 2.06 | `retain_harm_veto` |
| `convexity` | 0.14 | 2.07 | `retain_harm_veto` |
| `major_axis_length` | 0.11 | 5.35 | `retain_harm_veto` |
| `minor_axis_length` | 0.29 | 2.38 | `retain_harm_veto` |
| `roundness` | 0.21 | 3.46 | `retain_harm_veto` |
| `feret_diameter` | 0.18 | 5.67 | `retain_harm_veto` |
| `hu_moment_1` | 0.66 | 1.50 | `retain_harm_veto` |
| `hu_moment_2` | 0.37 | -0.05 | `retain_double_neutral` |
| `area_power_1_5` | 0.55 | -1.52 | `recommend_removal` |
| `area_major_axis_product` | 0.60 | 1.28 | `retain_harm_veto` |
| `center_vertical_occupancy` | 0.48 | -0.03 | `retain_double_neutral` |
| `end_vertical_occupancy_min` | 0.63 | 3.16 | `retain_harm_veto` |
| `end_vertical_occupancy_max` | -0.09 | 9.69 | `retain_harm_veto` |
| `center_to_end_occupancy_ratio` | 0.52 | 4.40 | `retain_harm_veto` |
| `centroid_x_offset` | 0.20 | 3.80 | `retain_harm_veto` |
| `centroid_y_ratio` | 0.80 | 2.21 | `retain_harm_veto` |
| `area_transformations` | 0.28 | -0.74 | `retain_double_neutral` |
| `bounding_rectangle_relations` | 0.55 | -0.46 | `retain_double_neutral` |
| `equivalent_ellipse_relation` | 0.21 | 3.46 | `retain_harm_veto` |
| `vertical_occupancy_relation` | 0.52 | 4.40 | `retain_harm_veto` |
| `convex_hull_relations` | 3.49 | -0.03 | `retain_harm_veto` |
| `area_contour_relation` | 0.14 | 4.48 | `retain_harm_veto` |

## Limitações

As mesmas máscaras orientam esta seleção; os valores são evidência de desenvolvimento, não validação independente em animais novos.

## Registro de revisão humana

- Status: revisado
- Interpretações aceitas, corrigidas ou rejeitadas: foram aceitas as recomendações provisórias produzidas pela Regra Conservadora de Remoção. `area_power_1_5` foi retirada por ser estruturalmente redundante com `area` e porque sua retirada melhorou o MAE da Rede Densa em 1,52 kg sem prejudicar o Random Forest além da margem prática de 1 kg. As outras 25 features foram mantidas por veto de dano ou neutralidade dupla.
- Conjunto Compartilhado de Features confirmado: `area`, `perimeter`, `solidity`, `circularity`, `equivalent_diameter`, `bbox_width`, `bbox_height`, `bbox_area`, `aspect_ratio`, `extent`, `convex_area`, `convexity`, `major_axis_length`, `minor_axis_length`, `roundness`, `feret_diameter`, `hu_moment_1`, `hu_moment_2`, `area_major_axis_product`, `center_vertical_occupancy`, `end_vertical_occupancy_min`, `end_vertical_occupancy_max`, `center_to_end_occupancy_ratio`, `centroid_x_offset` e `centroid_y_ratio`.
- Regra de padronização: `fit within each permitted training partition`.
- Limitação aceita: esta decisão utiliza evidência OOF da Divisão Estratificada Canônica para desenvolvimento; não demonstra estabilidade entre outras divisões nem validação independente em animais novos.
- Revisor: Victor Alexandre Saraiva Pimentel
- Revisado em: 2026-08-04
- Decisão auditável: https://github.com/Victor-Saraiva-P/buffalo-weight-pred/issues/16#issuecomment-5185467162
