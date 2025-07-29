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
import rasterio
import numpy as np
from pathlib import Path
from picanteo.utils.utils import check_file, remove_file_or_folder
from picanteo.utils.config_parser import ConfigParser
from picanteo.toolbox.data_extraction._2D.semantic_segmentation.semantic_segmentation import SemanticSegmentation
from picanteo.toolbox.data_extraction._3D.dsm_extraction import DsmExtraction
from picanteo.toolbox.data_filtering.threshold_filter import ThresholdFilter
from picanteo.toolbox.data_filtering.morphologic_filter import MorphologicFilter
from picanteo.toolbox.data_fusion.boolean_maps_merge import BooleanMapsMerge
from picanteo.pipelines.picanteo_pipeline import PicanteoPipeline

class BasicBitemporal2d3dPipeline(PicanteoPipeline):

    def __init__(self, input_config: str|Path) -> None:
               super().__init__(input_config)
           
    def run(self) -> None:
        # DSM extraction
        DsmExtraction(input_config=self.config['dsm_extraction_pre_conf']).run()
        DsmExtraction(input_config=self.config['dsm_extraction_post_conf']).run()
        for suffix in ['_pre', '_post']:
            # Building extraction 
            step = SemanticSegmentation(input_config=self.config[f'semantic_segmentation{suffix}_conf'])
            step.run()
            # Reproject building mask in ground geometry 
            step.sensor_to_ground_mask_projection(f"tutorial_bitemporal_2D3D_pipeline/semantic_segmentation{suffix}/predictive_entropy.tif", 
                f"tutorial_bitemporal_2D3D_pipeline/semantic_segmentation{suffix}/binarized_entropy_reprojected.tif", binarization_thresh=0.6)
            step.sensor_to_ground_mask_projection(f"tutorial_bitemporal_2D3D_pipeline/semantic_segmentation{suffix}/labels.tif", 
                f"tutorial_bitemporal_2D3D_pipeline/semantic_segmentation{suffix}/labels_reprojected.tif")
        # 2D diff
        with rasterio.open("tutorial_bitemporal_2D3D_pipeline/semantic_segmentation_pre/labels_reprojected.tif") as buildings_pre_ds:
            with rasterio.open("tutorial_bitemporal_2D3D_pipeline/semantic_segmentation_post/labels_reprojected.tif") as buildings_post_ds:
                buildings_pre = buildings_pre_ds.read(1).astype(bool)
                buildings_post = buildings_post_ds.read(1).astype(bool)
                diff = np.zeros_like(buildings_pre, dtype=bool)
                diff[np.logical_and(buildings_pre, np.logical_not(buildings_post))] = 1               
                binary_profile = buildings_pre_ds.profile.copy()
                binary_profile['dtype'] = np.uint8
                binary_profile['nodata'] = None
                with rasterio.open(Path(self.pipeline_output_dir, "2D_binary_raw_diff.tif"), "w", nbits=1, **binary_profile) as out_ds:
                    out_ds.write(diff, 1)
        # 3D diff
        with rasterio.open("tutorial_bitemporal_2D3D_pipeline/dsm_extraction_pre/dsm.tif") as dhm_pre_ds:
            with rasterio.open("tutorial_bitemporal_2D3D_pipeline/dsm_extraction_post/dsm.tif") as dhm_post_ds:
                dhm_pre = dhm_pre_ds.read(1)
                dhm_post = dhm_post_ds.read(1)
                raw_diff = dhm_post - dhm_pre
                raw_diff[np.logical_or(dhm_pre == dhm_pre_ds.profile['nodata'], dhm_post == dhm_post_ds.profile['nodata'])] = dhm_pre_ds.profile['nodata']
                with rasterio.open(Path(self.pipeline_output_dir, "3D_raw_diff.tif"), "w", **dhm_pre_ds.profile) as out_ds:
                    out_ds.write(raw_diff, 1)
                # Binary diff writing
                binary_diff = np.zeros_like(raw_diff, dtype=bool)
                binary_diff[raw_diff!=0] = 1
                binary_diff[raw_diff == dhm_pre_ds.profile['nodata']] = 0
                binary_profile = dhm_pre_ds.profile.copy()
                binary_profile['dtype'] = np.uint8
                binary_profile['nodata'] = None
                with rasterio.open(Path(self.pipeline_output_dir, "3D_binary_raw_diff.tif"), "w", nbits=1, **binary_profile) as out_ds:
                    out_ds.write(binary_diff, 1)
        # Elevation filter
        ThresholdFilter(input_config=self.config['elevation_filter_conf']).run()
        # Ambiguity filter
        ThresholdFilter(input_config=self.config['ambiguity_filter_pre_conf']).run()
        ThresholdFilter(input_config=self.config['ambiguity_filter_post_conf']).run()
        # Morphologic filter
        MorphologicFilter(input_config=self.config['2d_morphologic_filter_conf']).run()
        MorphologicFilter(input_config=self.config['3d_morphologic_filter_conf']).run()
        # Merges results
        BooleanMapsMerge(input_config=self.config['change_map_merging_conf']).run()

    def clean(self) -> None:
        pass
        
if __name__ == "__main__":
    pipeline = BasicBitemporal2d3dPipeline("docs/tutorials/data/configs/basic_bitemporal_2D3D_pipeline.yaml")
    pipeline.run()