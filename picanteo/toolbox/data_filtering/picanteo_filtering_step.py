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
    This module is used to define the structure of a pipeline filtering step in Picanteo.
"""

from pathlib import Path
from picanteo.toolbox.picanteo_step import PicanteoStep
from picanteo.utils.utils import check_file
from picanteo.utils.logger import logger
from picanteo.utils.config_parser import ConfigParser

class PicanteoFilteringStep(PicanteoStep):
    """
        Abstract base class for a Picanteo pipeline filtering step.

        Subclasses must implement `run` to filter the raster specified in the configuration,
        `store_log` to handle step-specific logs, and `clean` to remove temporary files.    
    """

    def __init__(self, input_config: str|Path) -> None:
        """
            Picanteo filtering step constructor. 
            
            Args:
                input_config (str|Path): path to the input yaml configuration file containing following key in addition to the other required keys:
                                             - "raster_to_filter" (str|Path): path to the input raster to filter.

            Raises:
                ValueError: If "raster_to_filter" is missing in the input configuration file.
        """
        super().__init__(input_config)
        # Check input raster to filter
        if "raster_to_filter" not in self.config.keys():
            logger.error(f"Missing 'raster_to_filter' key in input configuration file. Provided keys: {self.config.keys()}")
            raise ValueError(f"Missing 'raster_to_filter' key in input configuration file")
        check_file(self.config["raster_to_filter"], valid_extensions=[".tif", ".tiff"])
        self.raster_to_filter = self.config["raster_to_filter"]
