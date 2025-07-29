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
import os
from pathlib import Path
from picanteo.utils.logger import logger
from picanteo.utils.utils import check_file
from picanteo.toolbox.picanteo_step import PicanteoStep


class Visualization(PicanteoStep):
    """
        This step is used to merge two boolean change maps.
    """

    def __init__(self, input_config: str | Path) -> None:
        """
            Merges two boolean change maps element-wise based on the specified logical operator.

            Args:
                input_config (str|Path): path to the input yaml configuration file containing following keys in addition to the other required keys:


            Raises:
                ValueError: If parameters from config file don't match the expected format.
                TypeError: If parameters from config file don't match the expected type.
                PermissionError: If permission is denied to read the file.
        """
        super().__init__(input_config)
        self.conf_path = input_config




    def run(self) -> None:
        """
            Main function to run the step.

        """
        logger.debug("Starting picanteo visualization step..")
        os.system(f"panel serve picanteo/toolbox/data_visualization/dashboard.py --args {self.conf_path} --dev")


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

