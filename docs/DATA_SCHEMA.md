# Private input schema

Each row is one patient. `ID` must be nonmissing and unique within its cohort. Cohort membership must be fixed before fitting. `label` is 0/1 pathological LNM status; if `Pathological_LNM` is supplied, it must match `label`. Binary predictors HE4, MR_reported_DMI and MR_reported_LNM are 0/1. MR_reported_LNM covers both pelvic and para-aortic regions. MR_reported_diameter is positive, in centimetres. `radiomics` and `DL` are stored numeric patient-signature values, not raw features.

## Preoperatively apparent uterine corpus-confined subgroup

All seven fields must equal zero:

- MR_reported_LNM
- Preop_cervical_stromal_invasion
- Preop_adnexal_involvement
- Preop_uterine_serosal_involvement
- Preop_extrauterine_local_extension
- Preop_peritoneal_metastasis
- Preop_distant_metastasis

Code 0 = assessed without a suspicious finding; 1 = a positive/suspicious finding; 2 = indeterminate. Any positive finding excludes membership. Otherwise missing/indeterminate findings make membership unclassifiable. Deep myometrial invasion does not exclude a patient. Postoperative pathology and FIGO are not used for eligibility.

## Restricted nodal pathways

`nodal_pathway_masks()` implements the paper's post hoc pelvic-plus-para-aortic lymphadenectomy and bilateral SLNB definitions. It requires Nodal_procedure, SLN_mapping_attempted, Pelvic_nodes_assessed, Para_aortic_nodes_assessed, SLN_initial_HE_status, Bilateral_SLN_mapping and Ultrastaging_performed with the literal categories documented in the function.

CSV and XLSX tables remain local. Outputs contain patient-level predictions and should also remain local unless an appropriate separate data-sharing process authorizes disclosure. No patient table, medical image, actual case example or identifier is included in this repository.
