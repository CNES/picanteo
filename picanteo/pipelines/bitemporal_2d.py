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
    This module describe the change detection pipeline based on 2D and 3D bitemporal VHR optical data.
"""
import rasterio
import numpy as np
from pathlib import Path
from shutil import copyfile
from picanteo.utils.logger import logger
from picanteo.utils.utils import check_file, remove_file_or_folder
from picanteo.utils.config_parser import ConfigParser
from picanteo.pipelines.picanteo_pipeline import PicanteoPipeline
from picanteo.toolbox.data_extraction._2D.semantic_segmentation.semantic_segmentation import SemanticSegmentation
from picanteo.toolbox.data_filtering.threshold_filter import ThresholdFilter
from picanteo.toolbox.data_filtering.morphologic_filter import MorphologicFilter
from picanteo.utils.utils import create_cog

class Bitemporal2dPipeline(PicanteoPipeline):
    """
        2D Bitemporal VHR optical data change detection pipeline.
    """
    def __init__(self, input_config: str|Path) -> None:
        """
            Picanteo Bitemporal 2D/3D change detection pipeline constructor. 
            
            Args:
                input_config (str|Path): path to the input yaml configuration file containing following keys in addition to the other required keys:
                                            - "semantic_segmentation_conf" (str|Path): Path to the inference step yaml configuration file template.
                                            - "uncertainty_filter_conf" (str|Path): Path to the uncertainty filter step yaml configuration file template.
                                            - "morphologic_filter_conf" (str|Path): Path to the morphologic filter step yaml configuration file template.
                                            - "pre_event" (dict): dict containing set of at least 2 images.
                                            - "post_event" (dict): dict containing set of at least 2 images.                                           
                                            - "roi" (dict, optional):  optional region of interest (expect GeoJSON).                                   
            Raises:
                ValueError: If parameters from config file don't match the expected format.
                TypeError: If parameters from config file don't match the expected type.
        """
        super().__init__(input_config)
        config_file_required = {"semantic_segmentation_conf", "uncertainty_filter_conf", "morphologic_filter_conf"}
        required_keys = config_file_required |{"pre_event", "post_event"}

        if not all(key in self.config.keys() for key in required_keys):
            logger.error(f"Configuration file is missing required keys: {required_keys}. Provided keys: {self.config.keys()}")
            raise ValueError(f"Configuration file is missing required keys: {required_keys}")

        for conf_file in config_file_required:
            check_file(self.config[conf_file], valid_extensions=[".yml", ".yaml"])
        
        if not isinstance(self.config["pre_event"], dict):
            logger.error(f"'pre_event' must be a dict (here: {type(self.config['pre_event'])})")
            raise TypeError("'pre_event' must be a dict")        
            
        if not isinstance(self.config["post_event"], dict):
            logger.error(f"'post_event' must be a dict (here: {type(self.config['post_event'])})")
            raise TypeError("'post_event' must be a dict")

        if "roi" in self.config.keys() and not isinstance(self.config["roi"], dict):
            logger.error(f"'roi' must be a dict (here: {type(self.config['roi'])})")
            raise TypeError("'roi' must be a dict")
        
        if "roi" not in self.config.keys():
            self.config["roi"] = None
        
        logger.debug(f"Bitemporal 2D pipeline input data: {self.config}")        


        self.step_conf = self.config["semantic_segmentation_conf"]
        
        step_output_dir = Path(self.pipeline_output_dir, f"inference") 
        if not step_output_dir.exists():
            step_output_dir.mkdir()
        self.step_output_dir = step_output_dir
    
    def run(self) -> None:
        """
            Main function describing the pipeline steps. 
        """
        buildings_pre, uncertainty_pre = self.segment_image(input_img=self.config['pre_event']['image1'], suffix="pre_event_img")
        buildings_post, uncertainty_post = self.segment_image(input_img=self.config['post_event']['image1'], suffix="post_event_img")

        # Uncertainty filtering
        building_diff_path = self.compute_building_diff(buildings_pre, buildings_post)
        uncertainty_refined_path = self.uncertainty_filter(self.config['uncertainty_filter_conf'], building_diff_path, uncertainty_pre, uncertainty_post)

        # Morphologic filtering
        change_map_2D = self.morphologic_filter(self.config['morphologic_filter_conf'], uncertainty_refined_path, "2D")
                
        self.prepare_visualization()
        
        self.clean()
        
  

    def segment_image(self, input_img, suffix=""):
        """
            Segment image using the initialized segmentation model and the defined merger. 
            
            Args:
                input_img (str|Path): path to the input image.
                suffix (str): suffix to append to output dir, e.g. pre or post

            Returns:
                Path, Path: path of the semantic segmentation map and the associated uncertainty map.

        """
        semantic_outdir = Path(self.pipeline_output_dir, f"semantic_segmentation_{suffix}") 
        if not semantic_outdir.exists():
            semantic_outdir.mkdir()
        updates = {
        "step_output_dir": str(semantic_outdir),
        "save_intermediate_data": self.save_intermediate_data,
        "create_logdir": False,
        "input_img_path": str(input_img)
        }
        semantic_used_conf = self.prepare_step_config(self.step_conf, updates, semantic_outdir)
        step = SemanticSegmentation(input_config=semantic_used_conf)
        step.run()   
        buildings_path = Path(semantic_outdir, "labels.tif")
        check_file(buildings_path)
       
        uncertainty_path = Path(semantic_outdir, "predictive_entropy.tif")
        check_file(uncertainty_path)

        return buildings_path, uncertainty_path

    def compute_building_diff(self, 
                             buildings_pre: str|Path, 
                             buildings_post: str|Path) -> Path:
        """
            Compute the binary difference between the two provided building maps. 
            
            Args:
                buildings_pre (str|Path): path to the input pre-event building segmentation map.
                buildings_post (str|Path): path to the input post-event building segmentation map.

            Returns:
                Path: path of the binary diff.

        """
        binary_building_diff_path = Path(self.pipeline_output_dir, "2D_binary_diff.tif")
        with rasterio.open(buildings_pre) as buildings_pre_ds:
            with rasterio.open(buildings_post) as buildings_post_ds:
                buildings_pre = buildings_pre_ds.read(1).astype(bool)
                buildings_post = buildings_post_ds.read(1).astype(bool)
                diff = np.zeros_like(buildings_pre, dtype=bool)
                diff[np.logical_and(buildings_pre, np.logical_not(buildings_post))] = 1               
                binary_profile = buildings_pre_ds.profile.copy()
                binary_profile['dtype'] = np.uint8
                binary_profile['nodata'] = None
                with rasterio.open(binary_building_diff_path, "w", nbits=1, **binary_profile) as out_ds:
                    out_ds.write(diff, 1)
        return binary_building_diff_path

    def uncertainty_filter(self, 
                           uncertainty_filter_conf: str|Path, 
                           raster_to_filter_path: str|Path, 
                           uncertainty_pre: str|Path,
                           uncertainty_post: str|Path) -> Path:
        """ 
            2D uncertainty filtering step of the Bitemporal 2D/3D change detection pipeline. 
            
            Args:
                uncertainty_filter_conf (str|Path): path to the input configuration file.
                raster_to_filter_path (str|Path): path to the input raster change map that will be filtered.
                uncertainty_pre (str|Path): Path to the  pre-event 2D uncertainty.
                uncertainty_post (str|Path): Path to the post-event 2D uncertainty.
            
            Returns:
                Path: path of the ambiguity refined change map.

        """
        logger.info("Starting uncertainty filtering step...")
        check_file(raster_to_filter_path)
        uncertainty_f_outdir = Path(self.pipeline_output_dir, "uncertainty_filtering")
        if not uncertainty_f_outdir.exists():
            uncertainty_f_outdir.mkdir()
        updates = {
        "step_output_dir": str(uncertainty_f_outdir),
        "save_intermediate_data": self.save_intermediate_data,
        "raster_to_filter": str(raster_to_filter_path),
        "greater": False
        }
        check_file(uncertainty_pre)
        updates["filter"] = str(uncertainty_pre)
        updates["out_name"] = f"{Path(raster_to_filter_path).stem}_uncertainty_pre_refined{Path(raster_to_filter_path).suffix}"
        uncertainty_pre_used_conf = self.prepare_step_config(uncertainty_filter_conf, updates, uncertainty_f_outdir)
        step = ThresholdFilter(input_config=uncertainty_pre_used_conf)
        pre_out_path = step.run()

        # Post-event filtering
        uncertainty_pre_used_conf.rename(str(uncertainty_pre_used_conf).replace('filter_used_conf', 'pre_filter_used_conf'))
        updates["raster_to_filter"] = str(pre_out_path)
        updates["filter"] = str(uncertainty_post)
        updates["out_name"] = updates["out_name"].replace('_uncertainty_pre_refined','_uncertainty_refined')
        uncertainty_post_used_conf = self.prepare_step_config(uncertainty_filter_conf, updates, uncertainty_f_outdir)
        step = ThresholdFilter(input_config=uncertainty_post_used_conf)
        out_path = step.run()
        uncertainty_post_used_conf.rename(str(uncertainty_post_used_conf).replace('filter_used_conf', 'post_filter_used_conf'))
        logger.info("Uncertainty filter: Done !")
        return out_path

    def morphologic_filter(self, 
                    morphologic_filter_conf: str|Path, 
                    raster_to_filter_path: str|Path,
                    prefix: str) -> Path:
        """
            Morphologic filtering step of the Bitemporal 2D/3D change detection pipeline. 
            
            Args:
                morphologic_filter_conf (str|Path): path to the input configuration file.
                raster_to_filter_path (str|Path): path to the input raster change map that will be filtered.
                prefix (str): except '2D' or '3D' indicating which change map is filtered.
            
            Returns:
                Path: path of the filtered change map.

        """
        if prefix not in ["2D", "3D"]:
            logger.error(f"'prefix' must be a valid str: '2D' or '3D' (here: {prefix})")
            raise ValueError("'prefix' must be a valid str: '2D' or '3D'")        
        
        logger.info(f"Starting {prefix} morphologic filtering step...")
        check_file(raster_to_filter_path)
        morphologic_f_outdir = Path(self.pipeline_output_dir, f"{prefix}_morphologic_filtering")
        if not morphologic_f_outdir.exists():
            morphologic_f_outdir.mkdir()
        updates = {
        "step_output_dir": str(morphologic_f_outdir),
        "save_intermediate_data": self.save_intermediate_data,
        "raster_to_filter": str(raster_to_filter_path),
        "min_surface": 12.0,
        "out_name": f"{Path(raster_to_filter_path).stem}_filtered_morphologic{Path(raster_to_filter_path).suffix}"
        }   
        morphologic_used_conf = self.prepare_step_config(morphologic_filter_conf, updates, morphologic_f_outdir)
        step = MorphologicFilter(input_config=morphologic_used_conf)
        out_path = step.run()
        logger.info(f"{prefix} Morphologic filter: Done !")
        return out_path


    def prepare_visualization(self) -> None:
        """
            Prepare a directory with all the files to display and pre-processed them for visualization.

        """
        visualization_outdir = Path(self.pipeline_output_dir, "data_visualization")
        pre_event_outdir = Path(visualization_outdir, "pre_event")
        post_event_outdir = Path(visualization_outdir, "post_event")
        for dir in [visualization_outdir, pre_event_outdir, post_event_outdir]:
            if not dir.exists():
                dir.mkdir()
        visu_filenames = {
            'pre_event': {
                'image':{
                    'src': Path(self.config['pre_event']['image1']),
                    'dst': Path(pre_event_outdir, 'image.tif')
                },
                'buildings':{
                    'src': Path(self.pipeline_output_dir, 'semantic_segmentation_pre_event_img', 'labels.tif'),
                    'dst': Path(pre_event_outdir, 'buildings.tif')
                },
                'uncertainty':{
                    'src': Path(self.pipeline_output_dir, 'semantic_segmentation_pre_event_img', 'predictive_entropy.tif'),
                    'dst': Path(pre_event_outdir, 'uncertainty.tif')
                },
                'diff':{
                    'src': Path(self.pipeline_output_dir, '2D_morphologic_filtering', '2D_binary_diff_uncertainty_refined_filtered_morphologic.tif'),
                    'dst': Path(pre_event_outdir, '2D_diff.tif')
                }
            },
            'post_event': {
                'image':{
                    'src': Path(self.config['post_event']['image1']),
                    'dst': Path(post_event_outdir, 'image.tif')
                },
                'buildings':{
                    'src': Path(self.pipeline_output_dir, 'semantic_segmentation_post_event_img', 'labels.tif'),
                    'dst': Path(post_event_outdir, 'buildings.tif')
                },
                'uncertainty':{
                    'src': Path(self.pipeline_output_dir, 'semantic_segmentation_post_event_img', 'predictive_entropy.tif'),
                    'dst': Path(post_event_outdir, 'uncertainty.tif')
                },
                'diff':{
                    'src': Path(self.pipeline_output_dir, '2D_morphologic_filtering', '2D_binary_diff_uncertainty_refined_filtered_morphologic.tif'),
                    'dst': Path(post_event_outdir, '2D_diff.tif')
                }
            }
        }
        for sub_dir in visu_filenames:
            for f in visu_filenames[sub_dir]:
                check_file(visu_filenames[sub_dir][f]['src'])
                copyfile(visu_filenames[sub_dir][f]['src'], visu_filenames[sub_dir][f]['dst'])
        # Normalization
        for image_path in [visu_filenames['pre_event']['image']['dst'], visu_filenames['post_event']['image']['dst']]:
            with rasterio.open(image_path) as image_ds:
                image = image_ds.read().astype(float)
                profile = image_ds.profile.copy()
                if 'nodata' in image_ds.profile.keys():
                    image = np.where(image == image_ds.profile['nodata'], np.nan, image)
                    del profile['nodata']
                for band in range(image.shape[0]):
                    image[band,:,:] = self.normalize(image[band,:,:])
                profile['dtype'] = np.uint8
                profile['count'] = 3
                with rasterio.open(image_path, "w", **profile) as out_ds:
                    out_ds.write(image[:3,:,:])

        for image_path in [visu_filenames['pre_event']['buildings']['dst'], visu_filenames['post_event']['buildings']['dst']]:
            with rasterio.open(image_path) as image_ds:
                image = image_ds.read(1).astype(int)*255
                profile = image_ds.profile.copy()
                if 'nodata' in image_ds.profile.keys():
                    del profile['nodata']
                profile['dtype'] = np.uint8

                with rasterio.open(image_path, "w", **profile) as out_ds:
                    out_ds.write(image, 1)
        # For uncertainty, we use the same color ramp
        uncert_pre_path = visu_filenames['pre_event']['uncertainty']['dst']
        uncert_post_path = visu_filenames['post_event']['uncertainty']['dst']
        with rasterio.open(uncert_pre_path) as uncert_pre_ds:
            with rasterio.open(uncert_post_path) as uncert_post_ds:
                profile = uncert_post_ds.profile.copy()
                uncertainty_pre = uncert_pre_ds.read(1).astype(float)
                uncertainty_post = uncert_post_ds.read(1).astype(float)
                if 'nodata' in uncert_pre_ds.profile.keys():
                    uncertainty_pre[uncertainty_pre == uncert_pre_ds.profile['nodata']] = np.nan
                if 'nodata' in uncert_post_ds.profile.keys():
                    uncertainty_post[uncertainty_post == uncert_post_ds.profile['nodata']] = np.nan
                    del profile['nodata']
                uncert_pre_min, uncert_pre_max = np.nanpercentile(uncertainty_pre, 2), np.nanpercentile(uncertainty_pre, 98)
                uncert_post_min, uncert_post_max = np.nanpercentile(uncertainty_post, 2), np.nanpercentile(uncertainty_post, 98)
                uncert_min = np.max([uncert_pre_min, uncert_post_min])
                uncert_max = np.min([uncert_pre_max, uncert_post_max])
                uncertainty_pre = self.normalize(uncertainty_pre, min=uncert_min,max=uncert_max)
                uncertainty_post = self.normalize(uncertainty_post, min=uncert_min,max=uncert_max)
                profile['dtype'] = np.uint8
                with rasterio.open(uncert_pre_path, "w", **profile) as out_ds:
                    out_ds.write(uncertainty_pre, 1)
                with rasterio.open(uncert_post_path, "w", **profile) as out_ds:
                    out_ds.write(uncertainty_post, 1)
        logger.info("Prepare Cloud Optimized GeoTIFFs for visualization")
        for sub_dir in visu_filenames:
            for f in visu_filenames[sub_dir]:
                check_file(visu_filenames[sub_dir][f]['dst'])
                create_cog(src_path=visu_filenames[sub_dir][f]['dst'], dst_path=str(visu_filenames[sub_dir][f]['dst']).replace(".tif", "_cog.tif"))
                remove_file_or_folder(visu_filenames[sub_dir][f]['dst'])

            
    def normalize(self, array: np.array, quantile: float = 2, min: float = None, max: float = None) -> None:
            """
                Normalizes bands into 0-255 scale after removing outliers.

                Args:
                    array (np.array): numpy array to normalize.
                    quantile (float, optional): quantile for ignoring outliers.
                    min (float, optional): fixed value for min in normalization.
                    max (float, optional): fixed value for max in normalization.

                Returns:
                    np.array: normalized array.

            """
            if min is not None and max is not None:
                array_min = min
                array_max = max
            else:
                array_min, array_max = np.nanpercentile(array, quantile), np.nanpercentile(array, 100-quantile)
            normalized = 255*((array - array_min)/(array_max - array_min))
            normalized[normalized>255] = 255
            normalized[normalized<0] = 0
            normalized = np.nan_to_num(normalized)
            return normalized.astype(np.uint8)
            

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
                elements_to_save = [Path(self.pipeline_output_dir, "logs"), Path(self.pipeline_output_dir, "data_visualization")]
                for child in Path(self.pipeline_output_dir).iterdir():
                    if child not in elements_to_save:
                        remove_file_or_folder(child)
            except Exception as error:
                logger.error(f"Failed to clean temporary files: {str(error)}")
                raise RuntimeError(f"Cleanup failed: {str(error)}")