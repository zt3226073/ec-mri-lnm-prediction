"""Optional reference-component checks on artificial inputs only."""
import unittest
import numpy as np

try:
    import torch
    from ec_lnm.network import FourBranchResNet50,SEQUENCES,binary_focal_loss,patient_equal_loss
    from ec_lnm.gradcam import positive_fusion_gradcam
except ImportError:
    torch=None

@unittest.skipIf(torch is None,"Optional imaging dependencies unavailable")
class ImagingTests(unittest.TestCase):
    def test_focal_and_patient_equal_weight(self):
        z=torch.zeros(3);y=torch.tensor([1.,1.,0.])
        loss=binary_focal_loss(z,y)
        self.assertAlmostEqual(float(loss[0]),.75*.25*np.log(2),places=7)
        weighted=patient_equal_loss(z,y,["A","A","B"],{"A":2,"B":1})
        self.assertAlmostEqual(float(weighted),float((loss[0]+loss[2])/2),places=7)

    def test_fixed_positive_cam_and_hook_cleanup(self):
        torch.set_num_threads(2);torch.manual_seed(18)
        model=FourBranchResNet50().eval()
        for p in model.parameters():p.requires_grad_(False)
        inputs={k:torch.randn(1,3,224,224) for k in SEQUENCES}
        result=positive_fusion_gradcam(model,inputs)
        self.assertGreaterEqual(result["cnn_probability"],0)
        self.assertLessEqual(result["cnn_probability"],1)
        for k,cam in result["maps"].items():
            self.assertEqual(cam.shape,(224,224));self.assertTrue(np.isfinite(cam).all())
            self.assertGreaterEqual(cam.min(),0);self.assertLessEqual(cam.max(),1.000001)
            self.assertEqual(len(model.branches[k].layer4[-1].conv3._forward_hooks),0)

class ScalingTests(unittest.TestCase):
    def test_scaler_fit_before_oversampling_and_no_heldout_fit(self):
        try:
            import imblearn
        except ImportError:self.skipTest("Optional imbalanced-learn unavailable")
        from ec_lnm.signatures import standardized_smote
        rng=np.random.RandomState(23);x=rng.normal(size=(40,3));y=np.r_[np.zeros(32),np.ones(8)]
        held=np.array([[100.,200.,300.]])
        result=standardized_smote(x,y,held)
        np.testing.assert_allclose(result["scaler"].mean_,x.mean(axis=0))
        np.testing.assert_allclose(result["heldout"],(held-x.mean(axis=0))/x.std(axis=0))
        self.assertEqual(result["scaler"].n_samples_seen_,40)

if __name__=="__main__":unittest.main()
