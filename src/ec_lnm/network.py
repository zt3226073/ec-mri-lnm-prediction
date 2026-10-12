"""Reference architecture matching the supplied four-branch checkpoint layout."""
import torch
from torch import nn
from torchvision.models import resnet50,ResNet50_Weights

SEQUENCES=("T2WI","DWI","ADC","3D-VIBE")

class FourBranchResNet50(nn.Module):
    def __init__(self,pretrained=False):
        super().__init__();self.branches=nn.ModuleDict()
        for name in SEQUENCES:
            b=resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None);b.fc=nn.Identity();self.branches[name]=b
        self.fusion_head=nn.Linear(8192,1)
    def features(self,inputs):
        if set(inputs)!=set(SEQUENCES):raise ValueError("All four aligned sequence tensors required")
        for x in inputs.values():
            if x.ndim!=4 or tuple(x.shape[1:])!=(3,224,224):raise ValueError("Each branch expects N x 3 x 224 x 224")
        return torch.cat([self.branches[k](inputs[k]) for k in SEQUENCES],dim=1)
    def forward(self,inputs):return self.fusion_head(self.features(inputs)).squeeze(-1)

def load_checkpoint(path,device="cpu"):
    # weights_only avoids executing arbitrary pickle code. Do not use weights_only=False.
    payload=torch.load(path,map_location=device,weights_only=True)
    if set(payload)!=set(SEQUENCES)|{"fusion_head"}:raise ValueError("Unexpected checkpoint dictionaries")
    model=FourBranchResNet50().to(device)
    for name in SEQUENCES:model.branches[name].load_state_dict(payload[name],strict=True)
    model.fusion_head.load_state_dict(payload["fusion_head"],strict=True);model.eval();return model

def binary_focal_loss(logits,targets,gamma=2.,positive_alpha=.75):
    targets=targets.to(logits.dtype);bce=nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction="none")
    probability=torch.sigmoid(logits);pt=torch.where(targets==1,probability,1-probability);alpha=torch.where(targets==1,positive_alpha,1-positive_alpha)
    return alpha*(1-pt).pow(gamma)*bce

def patient_equal_loss(logits,targets,patient_ids,total_slices_by_patient):
    """Per-slice inverse patient-count weights; denominator counts all epoch patients.

    total_slices_by_patient must refer to the full training epoch, not only this batch.
    Batch losses form a weighted empirical objective when all slices appear once per epoch.
    """
    weights=torch.as_tensor([1/total_slices_by_patient[i] for i in patient_ids],dtype=logits.dtype,device=logits.device)
    return (binary_focal_loss(logits,targets)*weights).sum()/len(total_slices_by_patient)
