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
    This module aims to filter out the points from a raster where their elevation is lower (or greater) than a threshold.
"""
import rasterio
import numpy as np
from pathlib import Path
from picanteo.utils.logger import logger
from picanteo.toolbox.data_filtering.picanteo_filtering_step import PicanteoFilteringStep
from picanteo.utils.utils import check_file

class ThresholdFilter(PicanteoFilteringStep):
    """
        Filter out points where the input value is lower (or greater) than a threshold.
    """

    def __init__(self, input_config: str|Path) -> None:
        """
            Picanteo threshold filter step constructor. 
            
            Args:
                input_config (str|Path): path to the input yaml configuration file containing following keys in addition to the other required keys:
                                            - "filter" (str|Path): path to the raster that is used to filter (e.g., NDVI map).
                                            - "threshold" (float):  threshold value.
                                         And optionnaly:
                                            - "greater" (bool, optional): if True, keep only the points that are greater than the threshold. Defaults to False.
                                            - "binary_input" (bool, optional): if True, write an output filtered raster as binary raster. Defaults to False.
                                            - "out_name" (str, optional): output file name with extension. Defaults to: "<input_path>_refined.tif".
            
            Raises:
                ValueError: If parameters from config file don't match the expected format.
                TypeError: If parameters from config file don't match the expected type.
        """
        super().__init__(input_config)
        
        required_keys = {"filter", "threshold"}
        if not all(key in self.config.keys() for key in required_keys):
            logger.error(f"Configuration file is missing required keys: {required_keys}. Provided keys: {self.config.keys()}")
            raise ValueError(f"Configuration file is missing required keys: {required_keys}")

        # Check filter
        check_file(self.config["filter"], valid_extensions=[".tif", ".tiff"])
        self.filter = self.config["filter"]

        # Check threshold type
        if not isinstance(self.config["threshold"], float):
            logger.error(f"'threshold' must be a float (here: {type(self.config['threshold'])})")
            raise TypeError("'threshold' must be a float")
        self.threshold = self.config["threshold"]
        
        # If True, keep only the points that are greater than the threshold. Otherwise keep only the points lower than the threshold
        if "greater" in self.config.keys():
            if not isinstance(self.config["greater"], bool):
                logger.error(f"'greater' param must be a bool (here: {type(self.config['greater'])})")
                raise TypeError("'greater' param must be a bool")
            self.greater = self.config["greater"]
        else:
            self.greater = False

        # If True, write an output filtered raster as binary raster
        if "binary_input" in self.config.keys():
            if not isinstance(self.config["binary_input"], bool):
                logger.error(f"'binary_input' param must be a bool (here: {type(self.config['binary_input'])})")
                raise TypeError("'binary_input' param must be a bool")
            self.binary_input = self.config["binary_input"]
        else:
            self.binary_input = False

        # output file name with extension. If None: "<input_path>_refined.tif".
        if "out_name" in self.config.keys():
            if not isinstance(self.config["out_name"], str):
                logger.error(f"'out_name' param must be a string (here: {type(self.config['out_name'])})")
                raise TypeError("'out_name' param must be a string")        
            self.out_name = self.config["out_name"]
        else:
            self.out_name = f"{Path(self.raster_to_filter).stem}_refined{Path(self.raster_to_filter).suffix}"
            logger.debug(f"No 'out_name' provided, use default value: {self.out_name}")


    def run(self) -> Path:
        """
            Main function to run the step. 
            
            Returns:
                Path: path of the refined change map.
        """
        logger.debug("Starting threshold filtering step...")

        self.raster_to_filter = Path(self.raster_to_filter)
        self.filter = Path(self.filter)
        with rasterio.open(self.raster_to_filter) as rtf_ds:
            with rasterio.open(self.filter) as f_ds:
                filter = f_ds.read(1)
                refined_raster = rtf_ds.read(1)

                if self.greater:
                    # Only keep values greater than the threshold
                    refined_raster[filter < self.threshold] = 0
                else:
                    # Only keep lower than the threshold
                    refined_raster[filter > self.threshold] = 0
                
                out_path = Path(self.step_output_dir, self.out_name)

                if self.binary_input:
                    with rasterio.open(out_path, "w", nbits=1,**rtf_ds.profile) as out_ds:
                        out_ds.write(refined_raster, 1)
                else:
                    binary_profile = rtf_ds.profile.copy()
                    binary_profile['dtype'] = np.uint8
                    binary_profile['nodata'] = None
                    with rasterio.open(out_path, "w", **binary_profile) as out_ds:
                        out_ds.write(refined_raster, 1)

                self.clean()

        logger.info("Threshold filter: Done !")
        return out_path
    

    def store_log(self) -> None:
        """
            No logfile to store. 
        """
        pass


    def clean(self) -> None:
        """
            Nothing to remove. 
        """
        pass
