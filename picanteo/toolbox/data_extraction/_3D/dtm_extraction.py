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
import yaml
from pathlib import Path
from picanteo.utils.logger import logger
from picanteo.utils.utils import remove_file_or_folder, check_file
from picanteo.toolbox.picanteo_step import PicanteoStep
from bulldozer.pipeline.bulldozer_pipeline import dsm_to_dtm

class DtmExtraction(PicanteoStep):
    """
        This step is used to extract a DTM from a DSM.
    """

    def __init__(self, input_config: str|Path) -> None:
        """
            Picanteo DtmExtraction step constructor.
            
            Args:
                input_config (str|Path): path to the input yaml configuration file containing following keys in addition to the other required keys:
                                            - "dsm_path" (str|Path): path to the input DSM.
                                            - "output_dir" (float): path to the output directory.
                                         And optionnaly:
                                            - "generate_dhm" (bool, optional): if True, generates the DHM.
                                            
            Raises:
                ValueError: If parameters from config file don't match the expected type.
                YAMLError: If an error occurs while parsing the YAML file.
                PermissionError: If permission is denied to read the file.
        """
        super().__init__(input_config)

        required_keys = {"dsm_path", "output_dir"}
        if not all(key in self.config.keys() for key in required_keys):
            logger.error(f"Configuration file is missing required keys: {required_keys}. Provided keys: {self.config.keys()}")
            raise ValueError(f"Configuration file is missing required keys: {required_keys}")

        # Check dsm
        check_file(self.config["dsm_path"], valid_extensions=[".tif", ".tiff"])
        
        self.bulldozer_config = Path(self.config["step_output_dir"], "bulldozer_config.yaml")
        try:
            with open(self.bulldozer_config, "w") as file:
                bulldozer_params = {key: self.config[key] for key in self.config 
                                    if key not in {"step_output_dir", "save_intermediate_data", "create_logdir"}}
                # Write input config after remove picanteo keys 
                yaml.dump(bulldozer_params, file)
        except YAMLError as error:
            logger.error(f"Exception occured while writting bulldozer conf file '{self.bulldozer_config}': {error}")
            raise YAMLError(f"Failed to update YAML file: {error}")
        except PermissionError as error:
            logger.error(f"Permission denied accessing file '{self.bulldozer_config}': {error}")
            raise PermissionError(f"Failed to access YAML file:{error}")
        logger.debug(f"Bulldozer Input data: {bulldozer_params}")


    def run(self) -> None:
        """
            Main function to run the step. 
        """
        logger.debug("Starting DTM extraction step...")
        dsm_to_dtm(config_path=str(self.bulldozer_config))
        self.store_log()
        self.clean()
        logger.info("DTM extraction: Done !")


    def store_log(self) -> None:
        """
            Move the logs generated by the step application(s) in the step log directory. 
        """
        bulldozer_log_files = [f for f in Path(self.step_output_dir).iterdir() if f.match("bulldozer*.log")]
        for f in bulldozer_log_files:
            f.rename(Path(self.log_dir, f.name))


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
                    Path(self.step_output_dir) / "developer",
                    Path(self.step_output_dir) / "masks",
                    self.bulldozer_config
                ]
                for element in elements_to_remove:
                        remove_file_or_folder(element, missing_ok=True)
            except Exception as error:
                logger.error(f"Failed to clean temporary files: {str(error)}")
                raise RuntimeError(f"Cleanup failed: {str(error)}")
    