# EC MRI LNM prediction

Open-source research code for a clinical, radiomics and deep-learning nomogram for pathological lymph node metastasis in MRI-visible endometrial cancer.

This repository provides **full-precision patient-level logistic inference and performance evaluation**. It also includes separately identified **reference implementations** for aligned MRI crops, feature scaling before Borderline-SMOTE, area-weighted aggregation, the four-branch ResNet-50 architecture, binary focal loss, patient-equal slice weighting and positive-fusion-logit Grad-CAM.

The reference imaging components were assembled for reproducibility and documentation. They are **not the original historical training program**, and no new MRI-to-signature model development or fully nested cross-fitting has been executed with this release. Neither their availability nor checkpoint compatibility demonstrates that they generated the study's stored signatures.

## Install

Python 3.10 or later:

```bash
python -m pip install .
```

Inference uses the Python standard library. Statistical analysis and plotting require optional dependencies:

```bash
python -m pip install '.[analysis]'
```

For the optional PyTorch/reference imaging components:

```bash
python -m pip install '.[analysis,imaging]'
```

## Predict from clinical variables and stored signatures

```bash
ec-lnm predict examples/toy_inputs.csv new_predictions.csv --model Nomogram
```

The bundled six models are Clinical, Radiomics, DL, Clinical+Radiomics, Clinical+DL and Nomogram. The combined models use HE4, MR_reported_DMI, MR_reported_diameter (cm), MR_reported_LNM, radiomics and/or DL. Binary indicators use 0/1. The input radiomics and DL scores must use the study's stored signature units. **Do not apply an additional scaler, logit or probability transformation to these scores.**

All commands require a new output path and preserve inputs. `examples/toy_inputs.csv` is an artificial software example, not a patient dataset or a claim of model performance.

## Evaluate private patient tables

```bash
ec-lnm evaluate --train private/train.xlsx --internal private/internal.xlsx --external private/external.xlsx --output results/main --calibration-curves
ec-lnm plot results/main results/figures
ec-lnm nomogram results/nomogram
```

By default this evaluates the bundled frozen equations. It does not fit any model to validation patients. To reproduce the final training-only fixed-predictor logistic fitting, add `--refit-training`; this does not rerun feature selection or train ResNet. Training-table radiomics/DL inputs must already be the downstream OOF signatures described in the paper. Training probabilities from the final combined regression are **apparent fitted probabilities**, not complete-pipeline OOF predictions.

The evaluation includes paired two-sided DeLong comparisons, bounded normal-approximation AUC intervals, average precision with patient bootstrap intervals, Wilson binomial intervals, Brier scores, calibration-in-the-large, jointly fitted calibration intercept/slope, and exploratory DCA with paired pointwise bootstrap intervals. Flexible calibration curves use fixed cohort-specific cubic B-spline knots and patient bootstrap bands. The default analysis seed is 978 and the main-cohort bootstrap count is 1000.

## Pooled clinically early subgroup

```bash
ec-lnm early-subgroup --internal private/internal.xlsx --external private/external.xlsx --output results/early
```

Only internal and external patients are pooled. Eligibility requires all seven preoperative indicators listed in `docs/DATA_SCHEMA.md` to equal zero. DMI and postoperative FIGO stage are not eligibility criteria. All six models remain fixed. AUC/AP uncertainty is bootstrapped by cohort and outcome; Brier/DCA uncertainty is bootstrapped by cohort, with 20,000 paired samples. The primary bootstrap seed is 978. A pooled subgroup is not an independent external validation dataset. Sparse events limit inference; this command does not fit subgroup calibration regressions.

## Imaging and Grad-CAM reference code

See `docs/IMAGING_REFERENCE.md` before using `imaging.py`, `signatures.py`, `network.py` or `gradcam.py`. Target class: LNM-positive pre-sigmoid fusion logit. Target layers: `layer4[-1].conv3` in each branch. Maps use spatial-mean gradient weights, weighted activation sum, ReLU, independent 0–1 normalization and bilinear enlargement. The display overlay is 60% MRI and 40% JET heatmap. Within-map activation is not a nodal-metastasis probability and cannot directly explain the downstream logistic nomogram.

The author-authorized checkpoint, scalers, selected-input classifiers and feature metadata are available in the [model-asset release](https://github.com/zt3226073/ec-mri-lnm-prediction/releases/tag/model-assets-20261013). Clinical tables, source MRI, ROIs and CAM arrays are **not distributed**. Historical patient-level pipelines in that archive are clearly marked as legacy; use the repository’s current full-precision equations for the paper models. A checkpoint is loaded only from a caller-provided local path using `torch.load(..., weights_only=True)`. Missing N4/registration settings, full ordered radiomics headers, raw feature matrices, fold records and pre-standardization patient-signature parameters preclude an independently verified complete MRI-to-risk reproduction.

## Scope and threshold

The nomogram cutoff of 0.161 and comparator cutoffs are inherited **exploratory classification thresholds**. No clinical omission threshold has been validated. The software does not justify omitting pathological nodal assessment. Model outputs remain conditional on the input signatures and evaluated population; no clinical deployment claim is made.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use artificial arrays only. Private data are excluded from the Git repository. A companion GitHub Actions workflow runs core tests; optional imaging dependencies are tested separately when available. See `docs/REPRODUCIBILITY.md` for the distinction between numerical patient-level reproduction and historical imaging provenance.

## License and sources

Code and the author-authorized model-asset release are distributed under the MIT license. Clinical data redistribution is not granted. Dependencies retain their own licenses.

Method/API sources: [Torchvision ResNet-50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html), [BorderlineSMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.BorderlineSMOTE.html), [Grad-CAM paper](https://arxiv.org/abs/1610.02391).

## Model downloads

See [models/MODEL_ASSETS.md](models/MODEL_ASSETS.md) for download links, checksums and the distinction between imaging assets, legacy mappings and the current paper equations. The plotting commands produce Fig.3 (nomogram), Fig.4 (ROC/PR) and Fig.5 (calibration/DCA), with wider square plot boxes and single-row probability ticks.
