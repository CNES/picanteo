#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of Picanteo
# (see https://github.com/CNES/picanteo).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    This module aims to extract a Digital Terrain Model (DTM) using Bulldozer.
"""
from pathlib import Path
from picanteo.utils.logger import logger
from picanteo.utils.utils import remove_file_or_folder
from picanteo.toolbox.picanteo_step import PicanteoStep
from bulldozer.pipeline.bulldozer_pipeline import dsm_to_dtm
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader

import os 
from pathlib import Path 
from typing import Any, Dict, Optional, Sequence 

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

from tqdm import tqdm
from picanteo.toolbox.data_extraction._2D.semantic_segmentation.inference_model import InferenceModel, BaseModel, TTAModel
from picanteo.toolbox.data_extraction._2D.semantic_segmentation.inference_merger import Merger, MeanMerger, WindowMerger
from picanteo.toolbox.data_extraction._2D.semantic_segmentation.inference_dataset import InferenceDataset, FullInferenceDataset, TiledInferenceDataset
from rasterio.windows import Window

from rasterio import features
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd

from shareloc.geofunctions import localization
from shareloc.image import Image
from shareloc.geomodels.geomodel import GeoModel
import contextlib 

class SemanticSegmentation(PicanteoStep):
    """
        This step is used to extract a DTM from a DSM.
    """
    def __init__(self, input_config: str|Path) -> None:
        """
            Picanteo Semantic SEgmentation step constructor.
            
            Args:
                input_config (str|Path): path to the input yaml configuration file containing following keys in addition to the other required keys:
                                            - "input_img_path" (str|Path): path to the input image.
                                            - "step_output_dir" (float): path to the output directory.
                                         
                                            
            Raises:
                ValueError: If parameters from config file don't match the expected type.
        """
        super().__init__(input_config)
        self.conf_path = input_config

        required_keys = {"model", "pretrained_weights", "input_img_path", "num_classes", "step_output_dir"}
        if not all(key in self.config.keys() for key in required_keys):
            logger.error(f"Configuration file is missing required keys: {required_keys}. Provided keys: {self.config.keys()}")
            raise ValueError(f"Configuration file is missing required keys: {required_keys}")
         # check for standard dataset configuration
        if "dataset" in self.config.keys():   
            patch_size = self.config["dataset"]["patch_size"]
            overlap = self.config["dataset"]["overlap"]
            padding = self.config["dataset"]["padding"]
            shifted_border = self.config["dataset"]["shifted_border"]

        else:
            self.config["dataset"] = {"patch_size": 512, "overlap": 256, "shifted_border": True, "padding":False}
            logger.debug(f"No dataset information provided, use default values: {self.config['dataset']}")



        if "dataloader" in self.config.keys():   
            batch_size = self.config["dataloader"]["batch_size"]
            num_workers = self.config["dataloader"]["num_workers"]
        else:
            self.config["dataloader"] = {"batch_size": 8, "num_workers":1}
            logger.debug(f"No dataloader information provided, use default values: {self.config['dataloader']}")

    def get_device(self) -> torch.device:
        """Get available device. @TO-DO remove duplicate, add optional manual CPU selection"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def get_state_dict(self, best_checkpoint):
        
        old_state_dict = torch.load(best_checkpoint, map_location="cpu", weights_only=True)#
        new_state_dict = OrderedDict()
        
        for key, value in old_state_dict.items():
            new_state_dict[key.replace('model.', '')] = value
        
        return new_state_dict 

    def normalize_image_percentiles(self, image, lower_percentile=2, upper_percentile=98):
        # @TO-DO: move to pre-processing
        lower_bound = np.percentile(image, lower_percentile)
        upper_bound = np.percentile(image, upper_percentile)
        
        image_clipped = np.clip(image, lower_bound, upper_bound)
        
        image_normalized = (image_clipped - lower_bound) / (upper_bound - lower_bound)
        
        image_scaled = (image_normalized * 255).astype(np.uint8)
        
        return image_normalized


    def run(self) -> None:
        """
            Main function to run the step. 
        """
       
        logger.info("Starting 2D semantic segmentation part...")
        logger.debug(f"Input conf: 'config_path': {self.config}")

    
        inference_config = OmegaConf.load(self.conf_path)
        input_img_path = self.config["input_img_path"]
     
        device = self.get_device()

        # get tiled dataset #Tiled
        dataset = TiledInferenceDataset(self.config["input_img_path"], self.config["dataset"]["patch_size"], self.config["dataset"]["overlap"], self.config["dataset"]["padding"], self.config["dataset"]["shifted_border"])
        
        # get windowed and batched dataloader
        dataloader = DataLoader(dataset, batch_size=self.config["dataloader"]["batch_size"], shuffle=False, num_workers=self.config["dataloader"]["num_workers"])
        save_pe = True
        save_mi = False
        # get model instance
        inf_model = TTAModel(self.conf_path)
        merger = WindowMerger(input_img_path, output_folder=self.config["step_output_dir"], patch_size=self.config["dataset"]["patch_size"], num_classes=self.config["num_classes"], save_labels=True, save_pe=save_pe, save_mi=True) #WindowMerger(input_img_path, output_folder=self.config["step_output_dir"], patch_size=self.config["dataset"]["patch_size"], num_classes=self.config["num_classes"], save_labels=True, save_pe=save_pe, save_mi=False)

        # main loop for inference @TO-DO: add entropy/MI for uncertainty estimation and check for TTA/Ensemble adaptations + multi-threading
        with rasterio.open(input_img_path, "r") as src:
           #@TO-DO: add multi-threading  to main loop: https://rasterio.readthedocs.io/en/stable/topics/concurrency.html
           
            with tqdm(total=len(dataloader), desc="Infer patch-batches from tiled input image") as pbar:
                #with torch.no_grad():
                for data in dataloader:
                    # B C H W 

                    pre_mod_init = self.normalize_image_percentiles(data["image"].detach().cpu().numpy())[:,:3,:,:]


                    with torch.no_grad():
            
                        predicted_masks_out = inf_model(torch.from_numpy(pre_mod_init).float().to(device)) # N_MC B C H W 
            
                        if predicted_masks_out.shape[2]==1:
                            predicted_masks_out = torch.concatenate((1.0-predicted_masks_out, predicted_masks_out), axis=2)
                
                        predicted_masks_out += 1e-6
           

                        pred_entr, mi = inf_model.get_mutual_information(predicted_masks_out)
                        predicted_masks_out = torch.mean(predicted_masks_out, axis=0).cpu().numpy()
                  
                        with rasterio.open(Path(self.config["step_output_dir"]) / "probas.tif", "r+") as src_probas:
                            with rasterio.open(Path(self.config["step_output_dir"]) / "weights.tif", "r+") as src_weights:

                                with rasterio.open(Path(self.config["step_output_dir"]) / "predictive_entropy.tif", "r+") if save_pe else contextlib.nullcontext() as src_pe:
                                    with rasterio.open(Path(self.config["step_output_dir"]) / "mutual_information.tif", "r+") if save_pe else contextlib.nullcontext() as src_mi:
                                        for index in range(0, predicted_masks_out.shape[0]):
                                            m = predicted_masks_out[index,:,:,:]
                                           
                                            col = data["col"][index]
                                            row = data["row"][index]

                                            old_probas = src_probas.read(window=Window(col, row, self.config["dataset"]["patch_size"], self.config["dataset"]["patch_size"]))
                                            old_weights = src_weights.read(window=Window(col, row, self.config["dataset"]["patch_size"], self.config["dataset"]["patch_size"]))

                                            src_probas.write(old_probas + m*merger.win , window=Window(col, row, self.config["dataset"]["patch_size"], self.config["dataset"]["patch_size"]))  
                                            src_weights.write(old_weights + merger.win , window=Window(col, row, self.config["dataset"]["patch_size"], self.config["dataset"]["patch_size"]))  
                                            if save_pe:
                                                old_pe = src_pe.read(window=Window(col, row, self.config["dataset"]["patch_size"], self.config["dataset"]["patch_size"]))
                                                src_pe.write(old_pe + pred_entr[index,:,:].cpu().numpy()*merger.win[0,:,:] , window=Window(col, row, self.config["dataset"]["patch_size"], self.config["dataset"]["patch_size"]))  
                                            if save_mi:
                                                old_mi = src_mi.read(window=Window(col, row, self.config["dataset"]["patch_size"], self.config["dataset"]["patch_size"]))
                                                src_mi.write(old_mi + mi[index,:,:].cpu().numpy()*merger.win[0,:,:] , window=Window(col, row, self.config["dataset"]["patch_size"], self.config["dataset"]["patch_size"]))  
                                            

                                    
                    pbar.update(1)
                                    
        src_probas.close()
        src_weights.close() 
        src.close()
        merger.aggregate_outputs()


    
    def sensor_to_ground_mask_projection(self, mask_path, rastmsk, binarization_thresh = None):
       
        dsm_path = Path(self.config["reprojection_dsm"])
        img_path = Path(self.config["input_img_path"])

    
        vectmsk = str(mask_path).replace(".tif", ".shp")
        buildings = self.polygonize(mask_path, binarization_thresh)
        buildings.to_file(vectmsk.replace(".shp", "_before_altitude.shp"))
        loc, crs, height_map = self.dsm_to_height_map(dsm_path, img_path)
        height_map.to_file(vectmsk.replace(".shp", "_heightmap_altitude.shp"))
        buildings_with_alt = self.set_altitude_to_polygons(buildings, height_map, loc, crs)
        buildings_with_alt.to_file(vectmsk)
        self.burn_polygons(dsm_path, buildings_with_alt, rastmsk)

    def polygonize(self, img, binarization_thresh=None):
        with rasterio.open(img) as src:
            mask = src.read(1)
            if binarization_thresh is not None:
                mask_binarized = np.zeros_like(mask, dtype = np.uint8)
                mask_binarized[mask>=binarization_thresh] = 1
                shapes = features.shapes(mask_binarized, mask=mask_binarized)
            else:
                shapes = features.shapes(mask, mask=mask)
        fc = ({"geometry": shape, "properties": {"value": value}} for shape, value in shapes)
        return gpd.GeoDataFrame.from_features(fc)

    def dsm_to_height_map(self, dsm, img):
        model = os.path.splitext(img)[0]+".geom"
        simage = Image(img, vertical_direction="north")
        smodel = GeoModel(model, "RPC")

        with rasterio.open(dsm) as src:
            crs = src.crs
            loc = localization.Localization(smodel, image=simage, epsg=crs.to_epsg())
            heights = src.read(1)
            cols, rows = np.meshgrid(np.arange(heights.shape[1]), np.arange(heights.shape[0]))
            xs, ys = rasterio.transform.xy(src.transform, rows, cols)
            pc = np.stack((np.array(xs), np.array(ys), np.array(heights).flatten()))
            pc = pc.reshape(3, -1).T
            pc = pc[pc[:, 2] != src.nodata]

        xyz = loc.inverse(pc[:, 0], pc[:, 1], pc[:, 2], using_geotransform=True)
        height_map = gpd.GeoDataFrame(pd.DataFrame({"height": xyz[2]}),
                                    geometry=gpd.points_from_xy(x=xyz[1], y=xyz[0]))
        return loc, crs, height_map

    def set_altitude_to_polygons(self, polygons, height_map, loc, crs):
        height_map_join = gpd.sjoin(height_map, polygons, how="left")
        polygons_stats = height_map_join.groupby('index_right')['height'].agg(['mean','std','max','min'])
        polygons_with_alt = pd.merge(polygons, polygons_stats, left_index=True, right_index=True,how='outer')
        polygons_with_alt = polygons_with_alt.dropna()
        def remove_interiors(poly):
            if poly.interiors:
                
                ext_poly = poly.exterior
                rings = [i for i in poly.interiors]
                for r in rings:
                    inner_poly = Polygon(r)
                    
                    ext_poly = ext_poly - inner_poly
                    print("ext_poly coords", ext_poly.coords)
                return ext_poly
            else:
                print("exterior", poly.exterior.coords.xy)
                return poly

        def project_polygon(poly, height):
          
            xx, yy = poly.exterior.coords.xy 
            xyz = loc.direct(np.array(yy), np.array(xx), height, using_geotransform=True)
            xx = xyz[:, 0]
            yy = xyz[:, 1]
            return Polygon(zip(xx,yy))

        polygons_with_alt["geometry"] = polygons_with_alt.apply(lambda row : project_polygon(row["geometry"], row["mean"]), axis=1)
        polygons_with_alt.set_crs(crs)
        return polygons_with_alt

    def burn_polygons(self, dsm, polygons, out):
        with rasterio.open(dsm) as src:
            profile = src.profile
            profile["dtype"] = rasterio.uint8
            profile["nbits"] = 1
            del profile["nodata"]
            with rasterio.open(out, 'w', **profile) as dst:
                image = features.rasterize(((g, 255) for g in polygons["geometry"]), transform=profile["transform"], out_shape=src.shape)
                dst.write(image, indexes=1)



    def store_log(self) -> None:
        """
            Move the logs generated by the step application(s) in the step log directory. 
        """
        pass
    

    def clean(self) -> None:
        """
            Remove step temporary files. 
        """
        if not self.save_intermediate_data and Path(self.step_output_dir).exists():
            elements_to_rem : list[Path] = []
            elements_to_rem.append(Path(self.step_output_dir)/"developer")
            elements_to_rem.append(Path(self.step_output_dir)/"masks")

            for element in elements_to_rem:
                remove_file_or_folder(element, missing_ok=True)
    