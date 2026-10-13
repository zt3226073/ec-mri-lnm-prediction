# Model assets

The model assets are released separately from Git source to keep the repository small. Download the files from the [2026-10-13 model release](https://github.com/zt3226073/ec-mri-lnm-prediction/releases/tag/model-assets-20261013).

- `four_branch_resnet50_checkpoint.pth`: unchanged four-branch ResNet-50 state tensors and single-output fusion head.
- `ec_lnm_model_components.zip`: unchanged scalers, selected-input classifiers and feature metadata, plus clearly identified legacy patient-level mappings; includes current paper equations and portable numeric parameter exports.
- `SHA256SUMS.txt`: SHA-256 values for both binary assets.

Use `src/ec_lnm/assets/model_specifications.json` or the identical `current_model_specifications.json` in the archive for the current six patient-level models. Do not substitute the historical nomogram surrogate or reapply feature/signature standardizers to stored scores. Original joblib objects were serialized with scikit-learn 1.6.1; portable JSON arrays avoid that version-dependent deserialization.

The checkpoint was inspected to contain only network state tensors, without patient tables or images. Selected-feature matrices, full extraction headers, fold records and original generating code are not provided by this asset release. It does not establish complete historical MRI-to-risk reproduction or a new patient MRI validation. Code and the author-authorized asset release use the repository’s MIT license; dependencies retain their own licenses.

GitHub recommends releases for distributing binaries that should not be tracked directly in Git ([GitHub documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)).
