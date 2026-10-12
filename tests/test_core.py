import unittest
import numpy as np
import pandas as pd
from ec_lnm.predict import predict,load_specifications,nomogram_points
from ec_lnm.statistics import delong,paired_delong,wilson,classification,net_benefit
from ec_lnm.imaging import equal_patient_slice_weights,dl_patient_logit,mask_change
from ec_lnm.subgroups import EARLY_FIELDS,early_status

class CoreTests(unittest.TestCase):
    def test_point_probability_reconciliation_on_artificial_inputs(self):
        spec=load_specifications()["Nomogram"]
        # Toy values do not reproduce or publish patient records or manuscript case examples.
        for values in [[0,0,1,0,0,0],[1,1,3,1,.5,.5],[0,1,8,0,-2,-3]]:
            row=dict(zip(spec["input_features"],values));eta=spec["intercept"]+np.dot(spec["coefficients"],values)
            expected=float(1/(1+np.exp(-eta)));self.assertAlmostEqual(predict(row,spec),expected,places=14);self.assertAlmostEqual(nomogram_points(row)["probability"],expected,places=14)
    def test_ties_and_degenerate_paired_comparison(self):
        y=[0,0,1,1];p=[.2,.5,.5,.8];auc,cov=delong(y,p);self.assertEqual(auc[0],.875);self.assertEqual(paired_delong(y,p,p)["P"],1)
    def test_exploratory_dca_can_be_negative(self):
        y=[0,0,1,1];nb,_=net_benefit(y,[[1,1,1,1],[0,0,0,0]],[.8]);self.assertAlmostEqual(nb[0,0],-1.5);self.assertEqual(nb[1,0],0)
    def test_no_positive_predictions_is_not_ppv_zero(self):
        result=classification([0,0,1,1],[.01,.02,.03,.04],.161);self.assertEqual(result["FN"],2);self.assertIsNone(result["PPV"]["estimate"])
    def test_sparse_event_wilson_reference(self):
        result=wilson(4,9);self.assertAlmostEqual(result["estimate"],4/9);self.assertAlmostEqual(result["low"],.1887785211,places=8)
    def test_invalid_inputs_fail(self):
        spec=load_specifications()["Nomogram"];row=dict(zip(spec["input_features"],[0,0,2,0,.1,.2]));row["HE4"]=2
        with self.assertRaises(ValueError):predict(row,spec)
        row["HE4"]=0;row["DL"]=float("nan")
        with self.assertRaises(ValueError):predict(row,spec)
    def test_group_selection_does_not_use_pathology_or_dmi(self):
        d=pd.DataFrame({k:[0,0,0] for k in EARLY_FIELDS});d["label"]=[1,0,1];d["MR_reported_DMI"]=[1,0,1]
        d.loc[1,"Preop_distant_metastasis"]=2;d.loc[2,"MR_reported_LNM"]=1
        self.assertEqual(early_status(d).tolist(),["eligible","unclassifiable","excluded"])
    def test_equal_patient_training_is_distinct_from_area_voting(self):
        ids=["a","a","b"];w=equal_patient_slice_weights(ids);self.assertEqual(w[:2].sum(),w[2]);r=dl_patient_logit([.2,.8],[1,3]);self.assertAlmostEqual(r["probability"],.65)
    def test_mask_symmetric_difference_uses_final_denominator(self):
        self.assertEqual(mask_change([1,0,0],[1,1,0]),50)

if __name__=="__main__":unittest.main()
