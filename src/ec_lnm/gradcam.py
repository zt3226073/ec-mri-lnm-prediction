"""Positive pre-sigmoid fusion-logit Grad-CAM; separate from the logistic nomogram."""
import torch
from torch.nn import functional as F
from .network import SEQUENCES

def positive_fusion_gradcam(model,inputs):
    """Exactly one aligned slice set (N=1). Return per-branch independently normalized maps."""
    if any(x.shape[0]!=1 for x in inputs.values()):raise ValueError("Compute CAM for one slice set at a time")
    activations={};handles=[]
    def hook(name):
        def capture(module,args,output):activations[name]=output
        return capture
    try:
        model.eval()
        for name in SEQUENCES:handles.append(model.branches[name].layer4[-1].conv3.register_forward_hook(hook(name)))
        with torch.enable_grad():
            prepared={k:v.detach().requires_grad_(True) for k,v in inputs.items()}
            logits=model(prepared);maps={}
            gradients=torch.autograd.grad(logits[0],[activations[k] for k in SEQUENCES])
            for name,grad in zip(SEQUENCES,gradients):
                weight=grad.mean(dim=(2,3),keepdim=True);cam=torch.relu((weight*activations[name]).sum(dim=1,keepdim=True))
                cam=cam-cam.amin();maximum=cam.amax();cam=cam/maximum if maximum>0 else torch.zeros_like(cam)
                cam=F.interpolate(cam,size=inputs[name].shape[-2:],mode="bilinear",align_corners=False)
                maps[name]=cam[0,0].detach().cpu().numpy()
            return {"fusion_logit":float(logits[0].detach()),"cnn_probability":float(torch.sigmoid(logits[0]).detach()),"maps":maps}
    finally:
        for h in handles:h.remove()

def overlay(display_mri,cam):
    import numpy as np
    from matplotlib import colormaps
    mri=np.asarray(display_mri,dtype=float);cam=np.asarray(cam,dtype=float)
    if mri.shape!=cam.shape or mri.ndim!=2 or not np.isfinite(mri).all() or not np.isfinite(cam).all():raise ValueError("Aligned finite 2D display image and CAM required")
    if np.any((mri<0)|(mri>1)) or np.any((cam<0)|(cam>1)):raise ValueError("Display MRI and normalized CAM must be in [0,1]")
    heatmap=colormaps["jet"](cam)[...,:3]
    return .60*np.repeat(mri[...,None],3,axis=-1)+.40*heatmap
