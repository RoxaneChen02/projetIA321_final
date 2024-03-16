import numpy as np
import torch
from numpy import ndarray
from torchvision import transforms as T
from torch import Tensor, FloatTensor
from pathlib import Path
from typing import Union, Callable, Sequence, Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_observation(obs: ndarray) -> FloatTensor:
    """
    Converts a single CarRacing observation into a tensor ready for autoencoder consumption.
    The transformation pipeline applied is [ToPILImage, CenterCrop(64, 64), ToTensor].     

    ## Parameters:  

    - **obs** *(ndarray)*: Observation from the CarRacing enironvment.     
    
    ## Input:  

    - **ndarray** *(N, 3:# of channels, H: height, W: width)*:  
        Gym CarRacer-v2 observation.  

    ## Output:  

    - **FloatTensor** *(1: # samples, 3: # of channels, 64: height, 64: width)*:  
        Reshaped image tensor ready for model consumption.
    """
    
    cropper = T.Compose([T.ToPILImage(),
                            T.CenterCrop((64,64)),
                            T.ToTensor()])
    converted = torch.from_numpy(obs.copy())
    converted = torch.einsum("hwc -> chw", converted)
    tList = [cropper(m) for m in torch.unbind(converted, dim=0) ]
    res = torch.stack(tList, dim=0)
    result = torch.einsum("cnhw -> nchw", result)
    return result.to(DEVICE)