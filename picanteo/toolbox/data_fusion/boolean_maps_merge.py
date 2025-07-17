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
    This module aims to merge two boolean layer of change detection using logical methods.
"""
import rasterio
import numpy as np
from pathlib import Path
from picanteo.utils.logger import logger
from picanteo.utils.utils import check_file
from picanteo.toolbox.picanteo_step import PicanteoStep

class BooleanMapsMerge(PicanteoStep):
    """
        This step is used to merge two boolean change maps.
    """

    def __init__(self, input_config: str|Path) -> None:
        """
            Merges two boolean change maps element-wise based on the specified logical operator.
            
            Args:
                input_config (str|Path): path to the input yaml configuration file containing following keys in addition to the other required keys:
                                            - "layer1" (str|Path): Path to the first change detection map.
                                            - "layer2" (str|Path): Path to the first change detection map.
                                            - "operator" (str): logical operator ('AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR').              
                                            - "out_name" (str, optional): output file name with extension.

            Raises:
                ValueError: If parameters from config file don't match the expected format.
                TypeError: If parameters from config file don't match the expected type.
                PermissionError: If permission is denied to read the file.
        """
        super().__init__(input_config)

        required_keys = {"layer1", "layer2", "operator"}
        if not all(key in self.config.keys() for key in required_keys):
            logger.error(f"Configuration file is missing required keys: {required_keys}. Provided keys: {self.config.keys()}")
            raise ValueError(f"Configuration file is missing required keys: {required_keys}")
        
        if not isinstance(self.config['operator'], str):
            logger.error(f"'operator' must be a str (here: {type(self.config['operator'])})")
            raise TypeError("'operator' must be a str")

        self.operator = self.config['operator'].upper()
        if self.operator not in ['AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR']:
            logger.error(f"'operator' must be a valid logic operator (here: {self.operator})")
            raise ValueError("'operator' must be a valid logic operator ('AND', 'OR', 'XOR', 'NAND', 'NOR', or 'XNOR')")

        check_file(self.config["layer1"], valid_extensions=[".tif", ".tiff"])
        check_file(self.config["layer2"], valid_extensions=[".tif", ".tiff"])
        with rasterio.open(self.config["layer1"]) as layer1_ds:
            self.layer1 = layer1_ds.read(1).astype(bool)
        with rasterio.open(self.config["layer2"]) as layer2_ds:
            self.layer2 = layer2_ds.read(1).astype(bool)
    
        if self.layer1.shape != self.layer2.shape:
            logger.error(f"Input layers must have the same shape (here: layer1 - {self.layer1.shape}, layer2 - {self.layer2.shape})")
            raise ValueError("Input layers must have the same shape")
        
        # output file name with extension. If None: "<input_path>_refined.tif".
        if "out_name" in self.config.keys():
            if not isinstance(self.config["out_name"], str):
                logger.error(f"'out_name' param must be a string (here: {type(self.config['out_name'])})")
                raise TypeError("'out_name' param must be a string")        
            self.out_name = self.config["out_name"]
        else:
            self.out_name = "binary_change_map_merged.tif"


    def run(self) -> Path:
        """
            Main function to run the step. 
            
            Returns:
                Path: path of the merged boolean 3D change map.

            Raises:
                ValueError: if an invalid operator is provided.
        """
        logger.debug("Starting change maps merge step...")

        if self.operator == 'AND':
            merge = np.logical_and(self.layer1, self.layer2)
        elif self.operator == 'OR':
            merge = np.logical_or(self.layer1, self.layer2)
        elif self.operator == 'XOR':
            merge = np.logical_xor(self.layer1, self.layer2)
        elif self.operator == 'NAND':
            merge = np.logical_not(np.logical_and(self.layer1, self.layer2))
        elif self.operator == 'NOR':
            merge = np.logical_not(np.logical_or(self.layer1, self.layer2))
        elif self.operator == 'XNOR':
            merge = np.logical_not(np.logical_xor(self.layer1, self.layer2))
        else:
            logger.error(f"'operator' must be a valid logic operator (here: {self.operator})")   
            raise ValueError("Invalid operator. Use 'AND', 'OR', 'XOR', 'NAND', 'NOR', or 'XNOR'.")
        
        out_path = Path(self.step_output_dir, self.out_name)
        with rasterio.open(self.config["layer1"]) as layer1_ds:
            with rasterio.open(out_path, "w", nbits=1,**layer1_ds.profile) as out_ds:
                out_ds.write(merge, 1)
        logger.info("Change maps merging: Done !")

        return out_path


    def store_log(self) -> None:
        """
            No logfile to store. 
        """
        pass


    def clean(self) -> None:
        """
            No file to clean. 
        """
        pass

