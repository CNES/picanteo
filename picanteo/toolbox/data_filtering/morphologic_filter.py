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
    This module aims to filter out the points from a raster based on morphologic approach.
"""
import rasterio
import cv2
import numpy as np
from skimage import morphology
from scipy.ndimage import label
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from pathlib import Path
from picanteo.utils.logger import logger
from picanteo.toolbox.data_filtering.picanteo_filtering_step import PicanteoFilteringStep
from picanteo.utils.utils import check_file

class MorphologicFilter(PicanteoFilteringStep):
    """
        Apply an opening step and filter out areas where the surface is lower than a threshold.
    """

    def __init__(self, input_config: str|Path) -> None:
        """
            Picanteo threshold filter step constructor. 
            
            Args:
                input_config (str|Path): path to the input yaml configuration file containing following key in addition to the other required keys:
                                            - "min_surface" (float): Minimum valid surface in square meter. If changes surface are lower than this value, remove it from the change map.
                                         And optionnaly:
                                            - "out_name" (str, optional): output file name with extension. Defaults to: "<input_path>_filtered.tif".            
            Raises:
                ValueError: If the 'min_surface' key is missing in the config file.
                TypeError: If 'min_surface' is not a float.
        """
        super().__init__(input_config)
        
        if not "min_surface" in self.config.keys():
            logger.error(f"Configuration file is missing required keys 'min_surface'. Provided keys: {self.config.keys()}")
            raise ValueError(f"Configuration file is missing required 'min_surface' key.")

        # Check threshold type
        if not isinstance(self.config["min_surface"], float):
            logger.error(f"'min_surface' must be a float (here: {type(self.config['min_surface'])})")
            raise TypeError("'min_surface' must be a float")
        self.min_surface = self.config["min_surface"]

        # convex_hull option. If None: False.
        if "convex_hull" in self.config.keys():
            if not isinstance(self.config["convex_hull"], bool):
                logger.error(f"'convex_hull' param must be a bool (here: {type(self.config['convex_hull'])})")
                raise TypeError("'convex_hull' param must be a bool")        
            self.convex_hull = self.config["convex_hull"]
        else:
            self.convex_hull = False
            logger.debug(f"No 'convex_hull' param provided, use default value: False")

        # output file name with extension. If None: "<input_path>_filtered.tif".
        if "out_name" in self.config.keys():
            if not isinstance(self.config["out_name"], str):
                logger.error(f"'out_name' param must be a string (here: {type(self.config['out_name'])})")
                raise TypeError("'out_name' param must be a string")        
            self.out_name = self.config["out_name"]
        else:
            self.out_name = f"{Path(self.raster_to_filter).stem}_filtered{Path(self.raster_to_filter).suffix}"
            logger.debug(f"No 'out_name' provided, use default value: {self.out_name}")


    def run(self) -> Path:
        """
            Main function to run the step. 
            
            Returns:
                Path: path of the filterd binary 3D change map.
        """
        logger.debug("Starting morphologic filtering step...")

        with rasterio.open(Path(self.raster_to_filter)) as rtf_ds:
            binary_profile = rtf_ds.profile.copy()
            binary_profile['dtype'] = np.uint8
            binary_profile['nodata'] = None

            filtered_raster = rtf_ds.read(1).astype(bool)
            
            kernel_size=15
            kernel_type='square'
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            filtered_raster  = self.opening_by_reconstruction(filtered_raster.astype(np.uint8), kernel).astype(bool)
            pixel_surface = np.floor(((self.min_surface)**2) / (rtf_ds.profile["transform"][0]**2)).astype(int)
            filtered_raster = morphology.remove_small_objects(filtered_raster, pixel_surface)

            if self.save_intermediate_data and self.convex_hull:
                with rasterio.open(Path(self.step_output_dir, 'raster_before_convex_hull.tif'), "w", nbits=1,**binary_profile) as out_ds:
                    out_ds.write(filtered_raster, 1)

            if self.convex_hull:
                filtered_raster = self.label_and_convex_hull(filtered_raster)

            out_path = Path(self.step_output_dir, self.out_name)
            with rasterio.open(out_path, "w", nbits=1,**binary_profile) as out_ds:
                out_ds.write(filtered_raster, 1)

            self.clean()

        logger.info("Morphologic filter: Done !")
        return out_path
    

    def opening_by_reconstruction(self, in_arr: np.array, kernel: np.array):
        """
            Apply an opening step on input array. 
               
            Args:
                in_arr (np.array): input boolean array.
                kernel (np.array): kernel used by the opening.
              
            Returns:
                np.array: array after opening.
        """
        reconstruction = cv2.morphologyEx(in_arr, cv2.MORPH_OPEN, kernel)
        while True:
            dilated = np.minimum(in_arr, cv2.dilate(reconstruction, kernel))
            if (dilated == reconstruction).all():
                return reconstruction
            reconstruction = dilated


    def label_and_convex_hull(self, in_arr: np.array) -> np.array:
        """
            Labels regions, applies convex hulls and returns the array with True for pixels within each convex hull.
        
            Args:
                in_arr (np.array): : input boolean array.
        
            Returns:
                np.array: array with True for pixels inside convex hulls of labeled regions.
        """
        labels, num_features = label(in_arr)
        result = np.zeros_like(in_arr, dtype=bool)
        
        for i in range(1, num_features + 1):
            coords = np.where(labels == i)
            points = np.column_stack((coords[1], coords[0]))
            
            # Need at least 3 points for a convex hull
            if len(points) >= 3: 
                hull = ConvexHull(points)
                
                # Check which points are inside the hull
                delaunay = Delaunay(points[hull.vertices])
                height, width = in_arr.shape
                x, y = np.meshgrid(np.arange(width), np.arange(height))
                grid_points = np.column_stack((x.ravel(), y.ravel()))
                
                inside_hull = delaunay.find_simplex(grid_points) >= 0
                inside_hull = inside_hull.reshape(height, width)
                
                result = np.logical_or(result, inside_hull)
            else:
                result[coords[0], coords[1]] = True
                
        return result


    def store_log(self) -> None:
        """
            No logfile to store. 
        """
        pass


    def clean(self) -> None:
        """
            Nothing to remove. 
        """
        pass
