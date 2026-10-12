"""Array-level reference utilities. Scanner/N4/registration defaults are not inferred."""
import numpy as np

def normalized_crop(image,mask,size=224):
    """One aligned axial slice; bounding box from the T2WI mask, per-crop z score."""
    from PIL import Image
    image=np.asarray(image,dtype=np.float32);mask=np.asarray(mask,dtype=bool)
    if image.shape!=mask.shape or image.ndim!=2 or not mask.any():raise ValueError("Nonempty aligned 2D mask required")
    if not np.isfinite(image).all():raise ValueError("Nonfinite MRI intensities")
    ys,xs=np.where(mask);crop=image[ys.min():ys.max()+1,xs.min():xs.max()+1]
    crop=np.asarray(Image.fromarray(crop).resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)
    sd=float(crop.std(ddof=0));crop=(crop-float(crop.mean()))/(sd if sd>0 else 1.)
    return np.repeat(crop[None],3,axis=0)

def area_weighted_mean(values,areas):
    values=np.asarray(values,dtype=float);areas=np.asarray(areas,dtype=float)
    if values.shape[0]!=len(areas) or not np.isfinite(values).all() or not np.isfinite(areas).all() or np.any(areas<=0):raise ValueError("Finite values and positive pre-resize tumor areas required")
    return np.tensordot(areas/areas.sum(),values,axes=(0,0))

def dl_patient_logit(slice_probabilities,areas):
    p=np.asarray(slice_probabilities,dtype=float)
    if p.ndim!=1 or np.any((p<0)|(p>1)):raise ValueError("One probability per aligned four-sequence slice set required")
    patient_p=float(area_weighted_mean(p,areas));clipped=np.clip(patient_p,1e-12,1-1e-12)
    return {"probability":patient_p,"logit":float(np.log(clipped/(1-clipped)))}

def equal_patient_slice_weights(patient_ids):
    ids=np.asarray(patient_ids)
    _,inverse,count=np.unique(ids,return_inverse=True,return_counts=True)
    if not len(ids):raise ValueError("Empty slice batch/index")
    return 1/count[inverse]

def mask_change(initial,final):
    a,b=np.asarray(initial,dtype=bool),np.asarray(final,dtype=bool)
    if a.shape!=b.shape or not b.any():raise ValueError("Aligned masks and nonempty final mask required")
    return float(np.logical_xor(a,b).sum()/b.sum()*100)
