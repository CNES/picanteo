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
    This module regroups bunch of convenience functions.
"""

from pathlib import Path
from picanteo.utils.logger import logger

from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import  cog_profiles

def create_cog(src_path, dst_path, profile="deflate", web_optimized=False, profile_options={}, **options):
    """Convert image to COG.
    Code example taken from rio-cogeo example: https://cogeotiff.github.io/rio-cogeo/API/
    """

    # Format creation option (see gdalwarp `-co` option)
    output_profile = cog_profiles.get(profile)
    output_profile.update(dict(BIGTIFF="IF_SAFER", BLOCKXSIZE=128, BLOCKYSIZE=128))
    output_profile.update(profile_options)

    # Dataset Open option (see gdalwarp `-oo` option)
    config = dict(
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="128",
    )

    cog_translate(
        src_path,
        dst_path,
        output_profile,
        config=config,
        in_memory=False,
        quiet=True,
        web_optimized = web_optimized,
        overview_level=4,
        allow_intermediate_compression=True,
        **options,
    )
    return True

def check_dir(path: str|Path, verbose: bool = False) -> None:
    """
        Asserts that the target directory exists. 
        
        Args:
            path (str|Path): Path to the file or directory to remove.
            verbose (bool, optional): If True, log that the directory exist. Defaults to False.
        
        Raises:
            ValueError: If the path is not a string or doesn't fit the expected formats.
            FileNotFoundError: If the file or directory doesn't exist.
    """
    # Validate path type
    if not isinstance(path, str) and not isinstance(path, Path):
        logger.error(f"Path must be a string or a Path (here: {type(path)})")
        raise ValueError("Path must be a string or a Path")
    
    # Check file format
    target_directory = Path(path).resolve()

    # Check file existence
    if not target_directory.is_dir():
        logger.error(f"Directory '{path}' doesn't exist")
        raise FileNotFoundError(f"Directory not found: {path}")

    if verbose:
        logger.debug(f"Directory '{path}' checked")


def check_file(path: str|Path, valid_extensions: list[str]|None = None, verbose: bool = False) -> None:
    """
        Asserts that the target file exists and fit with the provided extensions. 
        
        Args:
            path (str|Path): Path to the file or directory to remove.
            valid_extensions ([str], optional): List of supported file format.
            verbose (bool, optional): If True, log that the file exist. Defaults to False.
        
        Raises:
            TypeError: If the path or the is not a string or valid_extensions is not a list of string.
            ValueError: If the path doesn't fit the expected formats.
            FileNotFoundError: If the file or directory doesn't exist.
    """
    # Validate path type
    if not isinstance(path, str) and not isinstance(path, Path):
        logger.error(f"Path must be a string or a Path (here: {type(path)})")
        raise TypeError("Path must be a string or a Path")
    
    if valid_extensions and not all(isinstance(ext, str) for ext in valid_extensions):
        logger.error(f"valid_extensions must be a list of strings, got: {valid_extensions}")
        raise TypeError("valid_extensions must be a list of strings")

    # Check file format
    target_file = Path(path).resolve()
    if valid_extensions and target_file.suffix.lower() not in [ext.lower() for ext in valid_extensions]:
        logger.error(f"'path' argument must point to a {valid_extensions} file (here: {path})")
        raise ValueError(f"Invalid file extension: {target_file.suffix}")

    # Check file existence
    if not target_file.is_file():
        logger.error(f"File '{path}' doesn't exist")
        raise FileNotFoundError(f"File not found: {path}")

    if verbose:
        logger.debug(f"Input file ({path}) checked")


def remove_file_or_folder(path: Path, missing_ok: bool=False) -> None:
    """
        Utility function that remove target file or folder. 
        
        Args:
            path (Path): path to the file or directory to remove.
            missing_ok (bool, optional): if missing_ok is false, FileNotFoundError is raised if the path does not exist.
        
        Raises:
            TypeError: If the path is not a string.
            FileNotFoundError: If the file or directory doesn't exist.
    """
    # Validate path type
    if not isinstance(path, Path):
        logger.error(f"Path must be a valid Path (here: {type(path)})")
        raise TypeError("Provided path must be a Path")
    
    target_path = Path(path)
    try:
        # Check if the path points a file or a directory
        if target_path.is_file():
            target_path.unlink(missing_ok=missing_ok)
            logger.debug(f"Successfully deleted file: {path}")
        elif path.is_dir():
            for sub in target_path.iterdir():
                if sub.is_dir():
                    remove_file_or_folder(sub)
                else:
                    sub.unlink()
            target_path.rmdir()
            logger.debug(f"Successfully deleted directory (and subdirectories): {path}")
        else:
            if not missing_ok:
                logger.error(f"The provided file or directory '{path}' doesn't exist")
                raise FileNotFoundError(f"File or directory not found: {path}")
            else:
                logger.debug(f"No file or directory: {path} found.")
    except PermissionError as error:
        logger.error(f"Permission denied deleting '{path}': {error}")
        raise PermissionError(f"Permission denied deleting '{path}': {error}")