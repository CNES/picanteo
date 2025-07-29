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
    This module is used to define the structure of a pipeline step in Picanteo.
"""
from pathlib import Path
from abc import ABC, abstractmethod
from picanteo.utils.utils import check_dir, check_file
from picanteo.utils.logger import logger
from picanteo.utils.config_parser import ConfigParser

class PicanteoStep(ABC):
    """
        Abstract base class for a Picanteo pipeline step.

        Subclasses must implement `run` to execute the step, `store_log` to move step-specific logs to the log directory,
        and `clean` to remove temporary files generated during execution.
    """

    def __init__(self, input_config: str|Path) -> None: 
        """
            Picanteo step constructor. 
            
            Args:
                input_config (str|Path): path to the input yaml configuration file containing following keys:
                                            - "step_output_dir" (str|Path): path to the output directory.
                                         And optionnaly:
                                            - "save_intermediate_data" (bool, optional): if True, retains intermediate files generated
                                                                                         during the step execution. Defaults to False.
                                            - "create_logdir" (bool, optional): if True, creates a "logs" directory for the step 
                                                                                (required if the step produces more logs than the 
                                                                                regular picanteo logger). Defaults to False.
                            
            Raises:
                TypeError: If "step_output_dir" or "input_config" is not a string or Path, or if "save_intermediate_data" or "create_logdir" is not a boolean.
                ValueError: If "step_output_dir" is missing in the input configuration file.
                PermissionError: If the log directory cannot be created.        
        """
        # Check input configuration
        if not isinstance(input_config, (str, Path)):
            logger.error(f"input_config must be a string or Path, got: {type(input_config)}")
            raise TypeError("input_config must be a string or Path")
        check_file(input_config, valid_extensions=[".yml", ".yaml"])
        parser = ConfigParser()
        self.config = parser.read(input_config)
        
        # Check output directory
        if "step_output_dir" not in self.config.keys():
            logger.error(f"Missing 'step_output_dir' key in input configuration file. Provided keys: {self.config.keys()}")
            raise ValueError(f"Missing 'step_output_dir' key in input configuration file")
        Path(self.config["step_output_dir"]).mkdir(exist_ok=True)
        self.step_output_dir = self.config["step_output_dir"]

        # Check save_intermediate_data parameter
        if "save_intermediate_data" in self.config.keys():
            if not isinstance(self.config["save_intermediate_data"], bool):
                logger.error(f"save_intermediate_data param must be a bool (here: {type(self.config['save_intermediate_data'])})")
                raise TypeError("save_intermediate_data param must be a bool")
            self.save_intermediate_data = self.config["save_intermediate_data"]
        else:
            self.save_intermediate_data = False

        # Check create_logdir parameter
        if "create_logdir" in self.config.keys():
            if not isinstance(self.config["create_logdir"], bool):
                logger.error(f"create_logdir param must be a bool (here: {type(self.config['create_logdir'])})")
                raise TypeError("create_logdir param must be a bool")
            if self.config["create_logdir"]:
                # Create step log directory
                self.log_dir = Path(self.step_output_dir, "logs")
                try:
                    self.log_dir.mkdir(exist_ok=True)
                    logger.debug(f"Created log directory: {self.log_dir}")
                except PermissionError as e:
                    logger.error(f"Permission denied creating log directory '{self.log_dir}': {e}")
                    raise


    @abstractmethod
    def run(self) -> None:
        """
            Execute the step's workflow.

            This method must be implemented by subclasses to define the specific
            actions of the step. It should handle all necessary processing and
            store results in the `step_output_dir`.

            Raises:
                NotImplementedError: If not overridden by a subclass.
        """
        logger.error("The PicanteoStep sublasses must implement 'run' method")
        raise NotImplementedError("Subclasses must implement the 'run' method") 

    @abstractmethod
    def store_log(self) -> None:
        """
            Move step-specific logs to the log directory. 
        """
        pass

    @abstractmethod
    def clean(self) -> None:
        """
            Remove step temporary files generated during execution. 
        """
        pass
