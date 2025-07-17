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


class Merger(ABC):
    def __init__(self, input_path, output_folder, patch_size, num_classes, save_labels=True, save_pe=True, save_mi=False):
        """Init wrapper."""
        super().__init__()

        """Initialize model outputs"""
        # Check model weights file exists
        if not Path(input_path).exists():
            raise ValueError(f'The provided  input path `{input_path}` does not exist.')

        self.input_path = Path(input_path)
        self.output_folder = Path(output_folder)
        self.patch_size = patch_size

        
        
        self.num_classes = num_classes
        self.save_labels = save_labels
        self.save_pe = save_pe
        self.save_mi = save_mi
  
        self.win = self.get_merge_window()
       
        
        self.initialize_outputs()
        logger.info("Initialized output arrays")
        
        
    def initialize_outputs(self):
        input_raster = rasterio.open(self.input_path)

        # create aggreation weights for patch merging
        self.out_profile_weights = input_raster.profile.copy()
        self.out_profile_weights["dtype"] = np.float32
        self.out_profile_weights["count"] = self.num_classes
        

        # create output probabilities per class
        self.out_profile_probas = input_raster.profile.copy()
        self.out_profile_probas["dtype"] = np.float32
        self.out_profile_probas["count"] = self.num_classes

        # create final segmentation mask
        if self.save_labels:
            self.out_profile_labels = input_raster.profile.copy()
            self.out_profile_labels["dtype"] = np.uint8
            self.out_profile_labels["count"] = 1

        # create predictive entropy (uncertainty estimation)
        if self.save_pe:
            self.out_profile_pe = input_raster.profile.copy()
            self.out_profile_pe["dtype"] = np.float32
            self.out_profile_pe["count"] = 1

        # create mutual information (uncertainty estimation)
        if self.save_mi:
            self.out_profile_mi = input_raster.profile.copy()
            self.out_profile_mi["dtype"] = np.float32
            self.out_profile_mi["count"] = 1


        out_weights = rasterio.open(self.output_folder / "weights.tif", 'w', **self.out_profile_weights)
        out_probas = rasterio.open(self.output_folder / "probas.tif", 'w', **self.out_profile_probas)
        out_probas_fin = rasterio.open(self.output_folder / "probas_fin.tif", 'w', **self.out_profile_probas)
        
        if self.save_labels:
            out_labels = rasterio.open(self.output_folder / "labels.tif", 'w', **self.out_profile_labels)

        if self.save_pe:
            out_pe = rasterio.open(self.output_folder / "predictive_entropy.tif", 'w', **self.out_profile_pe)

        if self.save_mi:
            out_mi = rasterio.open(self.output_folder / "mutual_information.tif", 'w', **self.out_profile_mi)


        img_ds = rasterio.open(self.input_path)

        for block_index, window in img_ds.block_windows(1):
            img = input_raster.read(1, window=window)
            out_weights.write(np.zeros((self.num_classes, img.shape[0], img.shape[1])), window=window)
            out_probas.write(np.zeros((self.num_classes, img.shape[0], img.shape[1])), window=window)
            out_probas_fin.write(np.zeros((self.num_classes, img.shape[0], img.shape[1])), window=window)

            if self.save_labels:
                out_labels.write(np.zeros_like(img), 1, window=window)

            if self.save_pe:
                out_pe.write(np.zeros_like(img), 1, window=window)

            if self.save_mi:
                out_mi.write(np.zeros_like(img), 1,window=window)

        out_weights.close()
        out_probas.close()
        out_probas_fin.close()

        if self.save_labels:
            out_labels.close()

        if self.save_pe:
            out_pe.close()

        if self.save_mi:
            out_mi.close()

    def aggregate_outputs(self):
        logger.info("Start aggregating outputs")
        out_weights = rasterio.open(self.output_folder / "weights.tif", 'r+', **self.out_profile_weights)
        out_probas = rasterio.open(self.output_folder / "probas.tif", 'r+', **self.out_profile_probas)
        out_probas_fin = rasterio.open(self.output_folder / "probas_fin.tif", 'r+')
        if self.save_labels:
            out_labels = rasterio.open(self.output_folder / "labels.tif", 'r+', **self.out_profile_labels)

        if self.save_pe:
            out_pe = rasterio.open(self.output_folder / "predictive_entropy.tif", 'r+', **self.out_profile_pe)

        if self.save_mi:
            out_mi = rasterio.open(self.output_folder / "mutual_information.tif", 'r+', **self.out_profile_mi)


        for block_index, window in out_weights.block_windows(1):
            weights = out_weights.read(window=window)
            probas = out_probas.read(window=window)
            merged_probas = probas/weights
            out_probas_fin.write(merged_probas, window=window)
            

            if self.save_labels:
                out_labels.write(np.argmax(merged_probas, axis=0), 1, window=window)

            if self.save_pe:
                pe = out_pe.read(window=window)
                out_pe.write(pe/weights[0,:,:],window=window)

            if self.save_mi:
                mi = out_mi.read(window=window)
                out_mi.write(mi/weights[0,:,:],window=window)



        out_weights.close()
        out_probas.close()

        if self.save_labels:
            out_labels.close()

        if self.save_pe:
            out_pe.close()

        if self.save_mi:
            out_mi.close()
        logger.info("Inference complete")
    def get_merge_window(self):
        pass

class MeanMerger(Merger):
    def __init__(self, input_path, output_folder, patch_size, num_classes, save_labels=True, save_pe=False, save_mi=False) -> None:
        """Init wrapper."""
        super().__init__(input_path, output_folder, patch_size, num_classes, save_labels, save_pe, save_mi)
    
    def get_merge_window(self):
    
        return np.ones((self.num_classes, self.patch_size, self.patch_size))
    
class WindowMerger(Merger):
    def __init__(self, input_path, output_folder, patch_size, num_classes, save_labels=True, save_pe=False, save_mi=False) -> None:
        """Init wrapper."""
        super().__init__(input_path, output_folder, patch_size, num_classes, save_labels, save_pe, save_mi)
    
    def get_merge_window(self):
        
        wins =  [skimage.filters.window("hann", (self.patch_size, self.patch_size)) + 1e-8 for _ in range(self.num_classes)] # + 1e-8
    
        return np.stack(wins, axis=0)