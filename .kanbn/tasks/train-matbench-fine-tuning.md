---
created: 2021-12-13T21:07:50.278Z
updated: 2022-03-02T12:49:40.635Z
assigned: ""
progress: 0
tags: []
completed: 2022-03-02T12:49:40.635Z
---

# Train matbench: fine-tuning

Unfreeze all layers and perform a few training iterations to fine-tune.
Must be done on local PC: clusters don't have compatible CUDA version to use newer versions of TFP. Older TFP versions cannot freeze VGP layers.

## Relations

- [blocker train-matbench-variation-point-locations](train-matbench-variation-point-locations.md)
