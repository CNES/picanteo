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
    This module defines the abstract base class for pipelines in the Picanteo framework.

    It provides a structure for implementing specific pipeline workflows, ensuring consistent
    handling of output directories and intermediate data.
"""
import yaml
from pathlib import Path
from abc import ABC, abstractmethod
from picanteo.utils.utils import check_dir, check_file
from picanteo.utils.config_parser import ConfigParser
from picanteo.utils.logger import logger, setup_logger, init_logfile

class PicanteoPipeline(ABC):
    """
        Abstract base class for defining a Picanteo pipeline.

        Subclasses must implement the `run` method to define the pipeline's execution logic
        and `clean` to remove temporary files generated during execution.
    """

    def __init__(self, input_config: str|Path) -> None:
        """
            Picanteo pipeline constructor. 
            
            Args:
                input_config (str|Path): path to the input yaml configuration file containing following keys:
                                            - "pipeline_output_dir" (str|Path): path to the output directory.
                                         And optionnaly:
                                            - "save_intermediate_data" (bool, optional): if True, retains intermediate files generated
                                                                                         during the pipeline execution. Defaults to False.
                            
            Raises:
                TypeError: If "pipeline_output_dir" is not a string or Path, or if "save_intermediate_data" is not a boolean.
                ValueError: If "pipeline_output_dir" is missing in the input configuration file.
        """
        # Check input configuration
        if not isinstance(input_config, (str, Path)):
            raise TypeError("input_config must be a string or Path")
        check_file(input_config, valid_extensions=[".yml", ".yaml"])
        parser = ConfigParser()
        self.config = parser.read(input_config)
        
        # Check output directory
        if "pipeline_output_dir" not in self.config.keys():
            raise ValueError(f"Missing 'pipeline_output_dir' key in input configuration file")
        Path(self.config["pipeline_output_dir"]).mkdir(exist_ok=True)
        self.pipeline_output_dir = self.config["pipeline_output_dir"]
        
        log_dir = Path(self.pipeline_output_dir, "logs")
        if not log_dir.exists():
            log_dir.mkdir()
        logger = setup_logger(log_dir)
        init_logfile()

        # Check save_intermediate_data parameter
        if "save_intermediate_data" in self.config.keys():
            if not isinstance(self.config["save_intermediate_data"], bool):
                logger.error(f"save_intermediate_data param must be a bool (here: {type(self.config['save_intermediate_data'])})")
                raise TypeError("save_intermediate_data param must be a bool")
            self.save_intermediate_data = self.config["save_intermediate_data"]
        else:
            self.save_intermediate_data = False


    def prepare_step_config(self, 
                            template_config: str|Path, 
                            updated_keys: dict, 
                            step_output_dir: str|Path) -> Path:
        """
            Create a copy of the provided template_config file in the step_output_dir with updated keys. 
            
            Args:
                template_config (str|Path): path to the input yaml config file used as template_config.
                updated_keys (dict): keys and associated values to add or edit in the config file.
                step_output_dir (str|Path): location where the new config file is write.
                           
            Returns:
                Path: path of the new config file. Format: step_output_dir/<template_config>_used_conf.yaml 

            Raises:
                TypeError: If updated_keys is not a list of dir or step_output_dir is not a Path.
                YAMLError: If an error occurs while parsing the YAML file.
                PermissionError: If permission is denied to read the file.
        """
        if not isinstance(updated_keys, dict):
            logger.error(f"'updated_keys' must be a dict (here: {type(updated_keys)})")
            raise TypeError("'updated_keys' must be a dict")

        check_file(template_config)
        check_dir(step_output_dir)

        parser = ConfigParser()
        input_config = parser.read(template_config, verbose=False)

        for key, value in updated_keys.items():
            input_config[key] = value
        if "_conf" in template_config:
            used_conf_path = Path(step_output_dir, f"{Path(template_config).stem.replace('_conf', '_used_conf')}{Path(template_config).suffix}")
        else:    
            used_conf_path = Path(step_output_dir, f"{Path(template_config).stem}_used_conf{Path(template_config).suffix}")
        try:
            with open(used_conf_path, "w") as used_conf:
                yaml.dump(input_config, used_conf)
        except YAMLError as error:
            logger.error(f"Exception occured while writting conf file '{used_conf_path}': {error}")
            raise YAMLError(f"Failed to update YAML file: {error}")
        except PermissionError as error:
            logger.error(f"Permission denied accessing file '{used_conf_path}': {error}")
            raise PermissionError(f"Failed to access YAML file:{error}")
        return used_conf_path


    @abstractmethod
    def run(self) -> None:
        """
            Execute the pipeline's workflow.

            This method must be implemented by subclasses to define the specific steps
            of the pipeline. It should handle all necessary processing and store
            results in the `pipeline_output_dir`.

            Raises:
                NotImplementedError: If not overridden by a subclass.
        """
        logger.error("The PicanteoPipeline sublasses must implement 'run' method")
        raise NotImplementedError("Subclasses must implement the 'run' method") 

    @abstractmethod
    def clean(self) -> None:
        """
            Remove pipeline temporary files generated during execution. 
        """
        pass