# Reproducibility scope

## Executable patient-level model

`assets/model_specifications.json` supplies the exact full-precision intercept, coefficients, feature order and exploratory thresholds for six patient-level models. The four clinical/combined models were fitted to 268 training patients with 41 LNM events using unpenalized binomial logistic maximum likelihood and fixed predictors. The standalone radiomics/DL mappings are fixed comparator mappings. The internal cohort has 117 patients/22 events and the external cohort 135 patients/21 events.

Private verification compares all 520 rows and all six probability columns with the analysis exports without disclosing them here. Published software tests deliberately use artificial inputs. Reproducing patient-level probabilities is conditional on correctly supplied clinical inputs and stored signatures.

## Imaging provenance

Table S3's selected-feature coefficients/scalers refer to a different interface from the six-input nomogram. The selected-input interface needs verified feature order and units; it must not be applied to stored patient signatures. The supplied classifier arrays and selected-feature names are included for documentation, with their linkage limitations stated in `IMAGING_REFERENCE.md`.

The four-branch checkpoint has independent ResNet-50 state dictionaries keyed T2WI, DWI, ADC and 3D-VIBE and a single-output fusion_head. Three-channel input replication and the binary focal-loss configuration were confirmed by the authors. Complete original MRI-to-signature code, trained-fold records and raw feature examples were not provided. Consequently the new reference modules must not be called the historical generating code.

## Validation boundaries

ResNet development was restricted to training patients; internal/external cohorts were excluded from fine-tuning, tuning and early stopping. ResNet representation learning preceded the downstream OOF partitions and was not repeated within them. Downstream OOF signatures therefore do not constitute fully cross-fitted representation learning. The final combined regression fit on OOF signatures still has apparent training performance. No complete nested cross-validation or cross-fitted stacking sensitivity analysis has been performed by this software release.

The inherited 0.161 cutoff is exploratory for the final model. No recalibration or threshold search is applied to held-out cohorts. Calibration bands and bootstrap intervals condition on the fitted predictions and do not include the uncertainty of end-to-end model development. The early subgroup is pooled and exploratory, with five events; it is not a new external cohort.
