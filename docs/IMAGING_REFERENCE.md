# Imaging reference components

These components document confirmed method choices and provide reusable implementations. They were written for this release and must not be represented as the unavailable original training program.

## Inputs and feature interfaces

- Sequences and concatenation order: T2WI, DWI, ADC, delayed 3D-VIBE.
- Each normalized single-channel 224 × 224 ROI crop is repeated identically into three channels. ImageNet RGB intensity normalization is not added.
- Branch features: 2048; concatenated features: 8192. The fusion head outputs one logit.
- Radiomics aggregation: tumor-area-weighted slice feature mean. DL aggregation: tumor-area-weighted slice probability mean, then logit. Weighting areas refer to the T2WI reference mask before crop resizing.
- Training feature standardization is fitted on original observations before Borderline-SMOTE1. Held-out observations are transformed only. Sampling strategy 0.5, k_neighbors=5, m_neighbors=10, random_state=42.
- Equal aggregate patient weighting for slice training and area-weighted prediction aggregation address different operations.

`normalized_crop()` uses bilinear resizing as a reference implementation choice; exact original crop margins and interpolation settings were not available. `pcc_keep()` provides deterministic correlation filtering; it is not a recovered original implementation. There is no unverified reconstruction of the original nested LASSO, optimization/early-stopping or augmentation logs.

N4 correction and rigid registration are documented in the manuscript, but their original parameter files were not supplied. This repository does not silently invent these settings. MRI signature inference additionally requires the complete feature order, matching extracted feature units, actual model objects and original patient-signature transformation parameters. Array-level helpers do not establish that linkage.

## Checkpoint

The local checkpoint can be used by `network.load_checkpoint(path)` when provided independently. Its required dictionaries are T2WI, DWI, ADC, 3D-VIBE and fusion_head. Strict loading rejects missing/unexpected tensors. The optional ImageNet V1 initialization in the constructor is a reference option and is not a statement about the exact initialization of the historical training run.

## Grad-CAM

`positive_fusion_gradcam()` targets the same positive-class pre-sigmoid fusion logit for every patient, regardless of pathological outcome or the predicted class. The target layer is layer4[-1].conv3 in each branch. Spatially averaged gradients weight feature-map channels, followed by weighted summation, ReLU, per-map min–max normalization to 0–1 and bilinear interpolation. A zero map remains zero. `overlay()` blends display MRI at 60% with JET heatmap at 40%.

The display MRI must already be scaled into [0,1]. MRI display windowing was not recovered and is not inferred from colored overlays. Existing manuscript maps were not regenerated for this release. Independent normalization prevents absolute activation comparison across maps. The numerical CNN sigmoid output, relative CAM activation and logistic nomogram probability are separate quantities.
