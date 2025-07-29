# Third-party libraries 
import numpy as np 
from collections import OrderedDict
from matplotlib import pyplot as plt
#import onnxruntime as ort 
import torch 
import torch.nn as nn 
from hydra.utils import instantiate 
from omegaconf import DictConfig, OmegaConf
import rasterio 
from torchvision.transforms import functional as TF
from pathlib import Path 
from typing import Any, Dict, Optional, Sequence 
from rasterio.windows import Window
from rasterio.windows import Window
import rasterio 
import numpy as np 
from matplotlib import pyplot as plt
import skimage
from abc import ABC
from picanteo.utils.logger import logger

class InferenceModel(nn.Module):
    """Basic inference model."""

    def __init__(self, inference_config_path: str) -> None:
        """Init wrapper."""
        super().__init__()

        """Initialize the model weights."""
        if not Path(inference_config_path).exists():
            raise ValueError(f'The provided  config file `{inference_config_path}` does not exist.')
        
        self.device = self.get_device()
    
        self.inference_config = OmegaConf.load(inference_config_path)
        self.setup_inference()
        self.nr_samples = 1
        self.nr_classes = self.inference_config.model.classes
   
        
    def get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    
    def setup_inference(self): 
    
        self.model = self.initialize_model(self.inference_config.model, self.inference_config.pretrained_weights)

    def get_state_dict(self,best_checkpoint):
        
        old_state_dict = torch.load(best_checkpoint, map_location="cpu", weights_only=True)#
        new_state_dict = OrderedDict()
        
        for key, value in old_state_dict.items():
            new_state_dict[key.replace('model.', '')] = value
        
        return new_state_dict 

    def initialize_model(self, model_to_instantiate, weights_path):
        """Initialize the model weights."""
      
        pretrained_weights = Path(weights_path)
        if not pretrained_weights.exists():
            raise ValueError(f'The given pretrained weights do not exist `{inference_config.pretrained_weights}`')
    
        model = instantiate(model_to_instantiate)
        model.load_state_dict(self.get_state_dict(pretrained_weights), strict=False)  # strict=True)
        model.eval()
        model.to(self.device)
        logger.info(f"Model: {model_to_instantiate}, instantiated with weights: {weights_path}")
        return model

    def get_mutual_information(self, probas):

        n_mc_index = 0
        n_classes_index = 2
        n_mc_samples = probas.shape[n_mc_index]

        n_classes = probas.shape[n_classes_index]
     
        #print("n_mc:" + str(n_mc_samples))
        #print("classes:" + str(n_classes))

        expected_p = torch.mean(probas, dim=n_mc_index)
        #print("expected p (mean proba), should be BS C W H", expected_p.shape)

        predictive_entropy = -torch.sum(expected_p*torch.log(expected_p), dim=1)
        #print("predictive entropy, should be BS W H", predictive_entropy.shape)
        del expected_p
        MC_entropy = torch.sum(probas*torch.log(probas), dim=n_classes_index)
        #print("MC entropy, should be NMC BS W H", MC_entropy.shape)
        #del probas
        expected_entropy = -torch.mean(MC_entropy, dim=0)
        #print("expected entropy, should be  BS W H", expected_entropy.shape)
        normed_predictive_entropy = predictive_entropy/np.log(n_classes)
        del MC_entropy
       
        mi = predictive_entropy - expected_entropy #normed_predictive_entropy - expected_entropy
   
        del expected_entropy

        return normed_predictive_entropy, mi 



    def logits_to_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.size(1) == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)
    
    @torch.no_grad()
    def forward(self, patches: torch.Tensor):
        """Torch forward call."""
        raise Exception("Forward not implemented!")

class BaseModel(InferenceModel):
    def __init__(self, config_path: str ) -> None:
        """Init wrapper."""
        super().__init__(config_path)
        self.nr_samples = 1

    
    @torch.no_grad()
    def forward(self, patches: torch.Tensor):
        probas = torch.zeros((self.nr_samples, patches.shape[0], self.nr_classes, patches.shape[2], patches.shape[3]),device=self.device)

        print("probas tensor", probas.shape)
        for i in range(0,self.nr_samples):
            logits = self.model(patches)
            print(logits.shape)
            probs = self.logits_to_probabilities(logits)
       
            probas[i]= self.logits_to_probabilities(logits)
            print("probas for one run", probas[i].shape)
        
        return probas 

class MCDOModel(InferenceModel):
    def __init__(self, config_path: str ) -> None:
        """Init wrapper."""
        super().__init__(config_path)
        self.nr_samples = 10

        for m in self.model.modules():
            if "dropout" in m.__class__.__name__.lower():
                print("ACTIVATE DROPOUT")
                m.train()

    
    @torch.no_grad()
    def forward(self, patches: torch.Tensor):
        probas = torch.zeros((self.nr_samples, patches.shape[0], self.nr_classes, patches.shape[2], patches.shape[3]),device=self.device)

        for i in range(0,self.nr_samples):
            logits = self.model(patches)
            print(logits.shape)
            probs = self.logits_to_probabilities(logits)
       
            probas[i]= self.logits_to_probabilities(logits)

        
        return probas 
    
class TTAModel(InferenceModel):
    # @TO-DO: refactor TTA Model to work with new inference structure, re-add MCDO + Ensemble
    def __init__(self, config_path: str ) -> None:
        """Init wrapper."""
        super().__init__(config_path)
        self.nr_samples = 6
    
    @torch.no_grad()
    def forward(self, patches: torch.Tensor):

        pre_mod_tta = [l for l in self.apply_tta(patches)] 
        probs = []
        for index, patch in enumerate(pre_mod_tta):
        
            logits = self.model(patch)
  
            probas = self.logits_to_probabilities(logits)
            probs.append(probas)
        reversed_probas = self.reverse_tta(probs)
        probas = np.asarray([l.cpu().numpy() for l in reversed_probas])
    
        
        return torch.from_numpy(probas) 
    
    def apply_tta(self, image):
        """Apply TTA: horizontal flip, vertical flip, and 90-degree rotation."""
        tta_images = [image]
        tta_images.append(TF.hflip(image))
        tta_images.append(TF.vflip(image))
        tta_images.append(TF.rotate(image, 90))
        tta_images.append(TF.rotate(image, 180))
        tta_images.append(TF.rotate(image, 270))
        return tta_images

    # Define the inverse augmentations
    def reverse_tta(self, predictions):
        """Reverse the TTA transformations."""
        reversed_predictions = [predictions[0]]
        reversed_predictions.append(TF.hflip(predictions[1]))
        reversed_predictions.append(TF.vflip(predictions[2]))
        reversed_predictions.append(TF.rotate(predictions[3], -90))
        reversed_predictions.append(TF.rotate(predictions[4], -180))
        reversed_predictions.append(TF.rotate(predictions[5], -270))
        return reversed_predictions