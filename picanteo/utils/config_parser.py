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
    This module is used to retrieve the Picanteo parameters from a YAML configuration file.
"""

import logging
from yaml import safe_load, YAMLError, dump
from pathlib import Path
from picanteo.utils.logger import logger
from picanteo.utils.utils import check_file


VALID_EXTENSIONS = [".yaml", ".yml"]


class ConfigParser:
    """
        Configuration file parser for reading Bulldozer parameters from YAML files.
        Implements the Singleton pattern to ensure only one instance exists.
    """

    _instance = None

    def __new__(cls, verbose: bool = False):
        """
        Controls the creation of a single instance of ConfigParser.

        Args:
            verbose (bool, optional): If True, increase output verbosity to DEBUG level. Defaults to False.

        Returns:
            ConfigParser: The single instance of the ConfigParser class.

        Raises:
            TypeError: If verbose is not a boolean.
        """
        if cls._instance is None:
            cls._instance = super(ConfigParser, cls).__new__(cls)
            # Initialize the instance only once
            if not isinstance(verbose, bool):
                raise TypeError("verbose must be a boolean")
            cls._instance.verbose = verbose
        return cls._instance


    def read(self, path: str|Path, verbose: bool = None) -> dict:
        """
            Read a YAML configuration file and return corresponding dict.

            Args:
                path (str|Path): Path to the YAML configuration file.
                verbose (bool, optional): If True, logs the retrieved data. Defaults to False.

            Returns:
                dict: Configuration parameters parsed from the YAML file.

            Raises:
                ValueError: If the path is not a string or does not point to a YAML file.
                FileNotFoundError: If the configuration file doesn't exist.
                YAMLError: If an error occurs while parsing the YAML file.
                PermissionError: If permission is denied to read the file.
        """
        if verbose is None:
            verbose = self.verbose
        check_file(path, verbose=verbose, valid_extensions=VALID_EXTENSIONS)     
        config_path = Path(path)
        try:
            with config_path.open("r") as stream:
                cfg = safe_load(stream)
                if cfg is None:
                    logger.warning(f"YAML file '{path}' is empty or contains no valid data.")
                    return {}
                if verbose:
                    logger.debug(f"Retrieved data: {cfg}")
                return cfg
        except YAMLError as error:
            logger.error(f"Exception occured while parsing the input configuration file '{path}': {error}")
            raise YAMLError(f"Failed to parse YAML file:{error}")
        except PermissionError as error:
            logger.error(f"Permission denied accessing file '{path}': {error}")
            raise PermissionError(f"Failed to access YAML file:{error}")
        except FileNotFoundError as error:
            logger.error(f"Config file not found ('{path}'). {error}")
            raise FileNotFoundError(f"Config file not found. {error}")

    def add_or_edit(self, path: str|Path, key: str, value: object) -> None:
        """
            Add or edit key of the of the provided YAML configuration file.

            Args:
                path (str|Path): Path to the YAML configuration file.
                key (str): Key to add or edit int the YAML file.
                value (object): Value associated to the key.

            Raises:
                ValueError: If the path is not a string or does not point to a YAML file.
                YAMLError: If an error occurs while parsing the YAML file.
                PermissionError: If permission is denied to read the file.
        """
        if not isinstance(key, str):
            logger.error(f"key param must be a str (here: {type(key)})")
            raise TypeError("key param must be a str")
        config = self.read(path, verbose=False)
        if key in config.keys():
            logger.debug(f"Config file updated with the new value: {str(value)} for the key: {key}")
        else:
            logger.debug(f"Config file updated. Added new key: {key}")
        config[key] = value
        try:
            with open(path, "w") as stream:
                dump(config, stream)
        except YAMLError as error:
            logger.error(f"Exception occured while updating the input configuration file '{path}': {error}")
            raise YAMLError(f"Failed to update YAML file: {error}")
        except PermissionError as error:
            logger.error(f"Permission denied accessing file '{path}': {error}")
            raise PermissionError(f"Failed to access YAML file: {error}")