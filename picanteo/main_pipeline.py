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
    This module contains the main Picanteo pipeline launch.
"""
import argparse
from pathlib import Path
from picanteo._version import __version__
from picanteo.utils.config_parser import ConfigParser
from picanteo.utils.logger import logger
from picanteo.pipelines.picanteo_pipeline import PicanteoPipeline
from picanteo.pipelines.bitemporal_2d_3d import Bitemporal2d3dPipeline   
from picanteo.pipelines.bitemporal_2d import Bitemporal2dPipeline   
from picanteo.toolbox.data_visualization.visualize_map import Visualization

class PipelineFactory:
    """
        Picanteo pipeline factory. Returns pipeline instance based on provided pipeline name.
    """

    @staticmethod
    def create_pipeline(pipeline_type: str, config_file: str|Path) -> PicanteoPipeline:
        """
        Factory method to create Picanteo pipeline objects based on input string and associated config.
        
        Args:
            pipeline_type (str): String indicating the pipeline to create ('bitemporal_2d3d', 'bitemporal_2d', etc.).
            config_file (str|Path): pipeline config file.

        Returns:
            PicanteoPipeline object of the specified type
            
        Raises:
            ValueError: If pipeline_type is not recognized
        """
        pipeline_types = {
            'bitemporal_2d3d': Bitemporal2d3dPipeline,
            'bitemporal_2d': Bitemporal2dPipeline
        }
        try:
            return pipeline_types[pipeline_type.lower()](config_file)
        except KeyError:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}. Except valid pipeline type: {list(pipeline_types.keys())}")


def run_pipeline(config_file: str|Path) -> None:
    """
        Picanteo main function. Run the pipeline defined in config_file. 

        Args:
            config_file (str|Path): pipeline config file (examples available in conf/pipelines).
        
        Raises:
            ValueError: If 'pipeline' key is missing in the config file.
    """
    parser = ConfigParser(verbose=False)
    cfg = parser.read(config_file)
    try:
        pipeline = PipelineFactory.create_pipeline(pipeline_type=cfg["pipeline"], config_file=config_file)
        logger.debug(f"Pipeline used by user: {pipeline}")
        pipeline.run()
    except KeyError:
            raise ValueError(f"Missing required key: 'pipeline' in input configuration file.")


def picanteo_cli() -> None:
    """
        Picanteo Command Line Interface (CLI). Run the pipeline defined in config file.
    """
    parser = argparse.ArgumentParser(
        description="Picanteo: CNES change detection framework for natural disaster response",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to the YAML configuration file'
    )
    parser.add_argument(
        '-v', '--version',
        action='version',
        version=f'picanteo {__version__}',
        help='Show the software version and exit'
    )
    args = parser.parse_args()
    run_pipeline(args.config_file)
    
def picanteo_visualize() -> None:
    parser = argparse.ArgumentParser(
        description="Picanteo: CNES change detection framework for natural disaster response",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'config',
        type=str,
        help='Path to the visualization config'
    )

    args = parser.parse_args()
    visu_step = Visualization(args.config)
    visu_step.run()    

if __name__ == "__main__":
    picanteo_cli()