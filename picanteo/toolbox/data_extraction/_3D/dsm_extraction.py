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
    This module aims to extract a Digital Surface Model (DSM) using CARS.
"""
import json
import logging
from pathlib import Path
from picanteo.utils.logger import logger
from picanteo.utils.utils import remove_file_or_folder, check_file
from picanteo.toolbox.picanteo_step import PicanteoStep
from cars.pipelines.pipeline import Pipeline
from cars.core import cars_logging

class DsmExtraction(PicanteoStep):
    """
        This step is used to extract a DSM from a set of stereo satellite images.
    """

    def __init__(self, input_config: str|Path) -> None:
        """
            Picanteo DsmExtraction step constructor.
            
            Args:
                input_config (str|Path): path to the input yaml configuration file containing following keys in addition to the other required keys:
                                            - "inputs" (dict): dict containing "sensors" dict that contains subdicts with pairs of image and geomodel.
                                            - "output" (dict): dict containing "directory" key.
                                         And optionnaly other CARS parameters.
                                            
            Raises:
                ValueError: If parameters from config file don't match the expected format.
                TypeError: If parameters from config file don't match the expected type.

        """
        super().__init__(input_config)

        required_keys = {"inputs", "output"}
        if not all(key in self.config.keys() for key in required_keys):
            logger.error(f"Configuration file is missing required keys: {required_keys}. Provided keys: {self.config.keys()}")
            raise ValueError(f"Configuration file is missing required keys: {required_keys}")
        
        # Check inputs
        if not isinstance(self.config["inputs"], dict):
                logger.error(f"'inputs' element must be a dict (here: {type(self.config['inputs'])})")
                raise TypeError("'inputs' must be a dict")
        if "sensors" not in self.config["inputs"].keys():
            logger.error(f"Input config file must contains 'inputs' key with 'sensors' subkey. Here {self.config['inputs'].keys()}")
            raise ValueError(f"'inputs' dict missing required key 'sensors'")
        if len(self.config["inputs"]["sensors"]) < 2:
            logger.error(f"DsmExtraction except at least a pair or stereo image. Here {len(self.config['inputs']['sensors'])} images provided.")
            raise ValueError(f"Not enough images provided (expected at least 2): {len(self.config['inputs']['sensors'])}")
        if len(self.config["inputs"]["sensors"]) > 2 and "pairing" not in self.config["inputs"].keys():
            logger.error(f"MMissing key: DsmExtraction except 'pairing' key if you provide more than 2 images.")
            raise ValueError(f"'inputs' dict missing required key 'pairing' when there is more than 2 input images")

        for img in self.config["inputs"]["sensors"]:
            check_file(self.config["inputs"]["sensors"][img]["image"], valid_extensions=[".tif", ".tiff", ".xml"])
        for model in self.config["inputs"]["sensors"]:
            check_file(self.config["inputs"]["sensors"][model]["geomodel"], valid_extensions=[".geom", ".xml"])
       
        # Check ouputs
        if not isinstance(self.config["output"], dict):
                logger.error(f"'output' element must be a dict (here: {type(self.config['output'])})")
                raise TypeError("'output' must be a dict")
        if "directory" not in self.config["output"].keys():
            logger.error(f"Input config file must contains 'output' key with 'directory' subkey. Here {self.config['output'].keys()}")
            raise ValueError(f"'inputs' dict missing required key 'sensors'")

        self.cars_config = {key: self.config[key] for key in self.config 
                            if key not in {"step_output_dir", "save_intermediate_data", "create_logdir"}}
        logger.debug(f"CARS Input data: {self.cars_config}")        


    def run(self) -> None:
        """
            Main function to run the step. 
        """
        logger.debug("Starting DSM extraction step...")
        cars_logging.setup_logging(
            "PROGRESS",
            out_dir=Path(self.step_output_dir, "logs"),
            pipeline="default",
        )
        pipeline = Pipeline("default", self.cars_config, self.step_output_dir)
        pipeline.run()
        self.clean()
        logger.info("DSM extraction: Done !")


    def store_log(self) -> None:
        """
            CARS already write the logs in the target log_dir. 
        """
        pass


    def clean(self) -> None:
        """
            Remove temporary files and directories generated during the step.
            Only removes files if `save_intermediate_data` is False and the output directory exists.

            Raises:
                RuntimeError: If cleanup fails.
        """
        if self.save_intermediate_data:
            logger.debug("Skipping cleanup: save_intermediate_data is True")
        else:
            try:
                elements_to_remove: list[Path] = [
                    Path(self.step_output_dir, "logs", "profiling"),
                    Path(self.step_output_dir, "logs", "workers_log"),
                    Path(self.step_output_dir, "metadata.json"),
                    Path(self.step_output_dir, "dsm", "index.json"),  
                    Path(self.step_output_dir, "used_conf.json")  
                ]
                for element in elements_to_remove:
                        remove_file_or_folder(element, missing_ok=True)
            except Exception as error:
                logger.error(f"Failed to clean temporary files: {str(error)}")
                raise RuntimeError(f"Cleanup failed: {str(error)}")
        
        # Move the results in the step_output_dir
        for result in Path(self.step_output_dir, "dsm").iterdir():
            result.rename(Path(result.parent.parent, result.name))
            
        remove_file_or_folder(Path(self.step_output_dir, "dsm"), missing_ok=True)    