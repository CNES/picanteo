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
from picanteo.toolbox.data_extraction._2D.semantic_segmentation.semantic_segmentation import SemanticSegmentation
from picanteo.toolbox.data_extraction._3D.dtm_extraction import DtmExtraction
from picanteo.toolbox.data_extraction._3D.dsm_extraction import DsmExtraction
from picanteo.toolbox.data_filtering.threshold_filter import ThresholdFilter
from picanteo.toolbox.data_filtering.morphologic_filter import MorphologicFilter
from picanteo.toolbox.data_fusion.boolean_maps_merge import BooleanMapsMerge
from picanteo.pipelines.picanteo_pipeline import PicanteoPipeline
from picanteo.utils.utils import create_cog

class Bitemporal2d3dPipeline(PicanteoPipeline):
    """
        2D/3D Bitemporal VHR optical data change detection pipeline.
    """
	

    def __init__(self, input_config: str|Path) -> None:
        """
            Picanteo Bitemporal 2D/3D change detection pipeline constructor. 
            
            Args:
                input_config (str|Path): path to the input yaml configuration file containing following keys in addition to the other required keys:
                                            - "semantic_segmentation_conf" (str|Path): Path to the inference step yaml configuration file template.
                                            - "dsm_extraction_conf" (str|Path): Path to the dsm extraction step yaml configuration file template.
                                            - "dtm_extraction_conf" (str|Path): Path to the dtm extraction step yaml configuration file template.
                                            - "elevation_filter_conf" (str|Path): Path to the elevation filter step yaml configuration file template.
                                            - "ambiguity_filter_conf" (str|Path): Path to the ambiguity filter step yaml configuration file template.
                                            - "ndvi_filter_conf" (str|Path): Path to the ndvi filter step yaml configuration file template.
                                            - "uncertainty_filter_conf" (str|Path): Path to the uncertainty filter step yaml configuration file template.
                                            - "morphologic_filter_conf" (str|Path): Path to the morphologic filter step yaml configuration file template.
                                            - "boolean_maps_merge_conf" (str|Path): Path to the boolean change maps merging step yaml configuration file template.
                                            - "pre_event" (dict): dict containing set of at least 2 images.
                                            - "post_event" (dict): dict containing set of at least 2 images.                                           
                                            - "roi" (dict, optional):  optional region of interest (expect GeoJSON).                                   
            Raises:
                ValueError: If parameters from config file don't match the expected format.
                TypeError: If parameters from config file don't match the expected type.
        """
        super().__init__(input_config)
        config_file_required = {"semantic_segmentation_conf", "dsm_extraction_conf",  "dtm_extraction_conf", 
                                "elevation_filter_conf", "ambiguity_filter_conf", "ndvi_filter_conf",
                                "boolean_maps_merge_conf", "uncertainty_filter_conf", "morphologic_filter_conf"}
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
        
        logger.debug(f"Bitemporal 2D/3D pipeline input data: {self.config}")        


    def run(self) -> None:
        """
            Main function describing the pipeline steps. 
        """
        logger.info("Starting 2D/3D bitemporal change detection pipeline")
        # @TO-DO: Move to config
        filter_individual_uncertainties = False
        ### DATA EXTRACTION ###
        # 3D Part
        logger.debug("Starting 3D data extraction part...")
        # DSM extraction 
        dsm_pre, ambiguity_pre, color_pre = self.dsm_extraction(self.config['dsm_extraction_conf'], self.config['pre_event'], pre_event=True, roi=self.config['roi'])
        dsm_post, ambiguity_post, color_post = self.dsm_extraction(self.config['dsm_extraction_conf'], self.config['post_event'], pre_event=False, roi=self.config['roi'])
        # DHM extraction 
        dhm_pre = self.dhm_extraction(self.config['dtm_extraction_conf'], dsm_pre, pre_event=True)
        dhm_post = self.dhm_extraction(self.config['dtm_extraction_conf'], dsm_post, pre_event=False)
        
        # 2D Part
        logger.debug("Starting 2D data extraction part...")
        buildings_pre_refined, uncertainty_pre = self.building_extraction(self.config['semantic_segmentation_conf'], 
                                                                  image=self.config['pre_event'][list(self.config['pre_event'].keys())[0]]['color'],
                                                                  pre_event=True,
                                                                  dsm=dsm_pre)
        buildings_post_refined, uncertainty_post = self.building_extraction(self.config['semantic_segmentation_conf'], 
                                                                  image=self.config['post_event'][list(self.config['post_event'].keys())[0]]['color'],
                                                                  pre_event=False,
                                                                  dsm=dsm_post)
        
        if filter_individual_uncertainties:
            # Directly calculate destroyed buildings from individual uncertainty-filtered predictions
            building_diff_path = self.compute_building_diff(buildings_pre_refined, buildings_post_refined)
        else:
            # filter destroyed reprojected buildings with re-projected binarized uncertainties
            raw_building_diff_path = self.compute_building_diff(buildings_pre_refined, buildings_post_refined)
            building_diff_path = self.uncertainty_filter_diff(self.config['uncertainty_filter_conf'], raw_building_diff_path, uncertainty_pre, uncertainty_post)
        
        # Morphologic filtering
        change_map_2D = self.morphologic_filter(self.config['morphologic_filter_conf'], building_diff_path, "2D")

        # 3D Part
        raw_diff_path, binary_raw_diff_path = self.compute_3D_diff(dhm_pre, dhm_post)
        # Elevation filtering
        alti_refined_path = self.elevation_filter(self.config['elevation_filter_conf'], binary_raw_diff_path, raw_diff_path)
        # Ambiguity filtering
        ambiguity_refined_path = self.ambiguity_filter(self.config['ambiguity_filter_conf'], alti_refined_path, ambiguity_pre, ambiguity_post)
        # NDVI filtering
        ndvi_refined_path = self.ndvi_filter(self.config['ndvi_filter_conf'], ambiguity_refined_path, color_pre, color_post)
        # Morphologic filtering
        change_map_3D = self.morphologic_filter(self.config['morphologic_filter_conf'], ndvi_refined_path, "3D")

        ### DATA FUSION ###
        merged_map = self.change_maps_merge(self.config['boolean_maps_merge_conf'], change_map_2D, change_map_3D)

        ### DATA VISUALIZATION ###
        self.prepare_visualization()
        
        self.clean()


    def dsm_extraction(self, 
                       dsm_extraction_conf: str|Path, 
                       images: dict[dict[Path]], 
                       pre_event: bool,
                       roi: dict=None) -> tuple[Path, Path, Path]:
        """
            DSM extraction step of the Bitemporal 2D/3D change detection pipeline. 
            
            Args:
                dsm_extraction_conf (str|Path): path to the input configuration file.
                images (dict(dict(Path))): dict containing pairs of Path to image and associated geomodel and color. 
                                           Expected format: {img1:{image: img1.tif, geomodel: geom1.geom, color: color1.tif}, img2: ...}
                pre_event (bool): flag indicating if the DSM input is the prevent or postevent one. 
                roi (dict, optionnal): Region of interest. 

            Returns:
                tuple: path of the output DSM, ambiguity and color reprojected in ground geometry.

        """
        logger.debug(f"Starting {'pre-event' if pre_event else 'post-event'} DSM extraction step...")
        suffix = "_pre" if pre_event else "_post"
        dsm_outdir = Path(self.pipeline_output_dir, f"dsm_extraction{suffix}") 
        if not dsm_outdir.exists():
            dsm_outdir.mkdir()
        updates = {
        "step_output_dir": str(dsm_outdir),
        "save_intermediate_data": self.save_intermediate_data,
        "create_logdir": True,
        "inputs": {"sensors" : images},
        "output": {"directory" : str(dsm_outdir), "auxiliary" : {"ambiguity" : True}}
        }
        if roi is not None and isinstance(roi, dict):
            updates["inputs"]["roi"] = roi 
        dsm_used_conf = self.prepare_step_config(dsm_extraction_conf, updates, dsm_outdir)
        step = DsmExtraction(input_config=dsm_used_conf)
        step.run()    
        dsm_path = Path(dsm_outdir, "dsm.tif")
        check_file(dsm_path)
        ambiguity_path = Path(dsm_outdir, "confidence_from_ambiguity_cars_1.tif")
        check_file(ambiguity_path)
        color_path = Path(dsm_outdir, "color.tif")
        check_file(color_path)
        logger.info(f"{'Pre-event' if pre_event else 'Post-event'} DSM extraction: Done !")
        return dsm_path, ambiguity_path, color_path


    def dhm_extraction(self, dtm_extraction_conf: str|Path, input_dsm: str|Path, pre_event: bool) -> Path:
        """
            DHM extraction step of the Bitemporal 2D/3D change detection pipeline. 
            
            Args:
                dtm_extraction_conf (str|Path): path to the input configuration file.
                input_dsm (str|Path): path to the input raster that will be filtered.
                pre_event (bool): flag indicating if the DSM input is the prevent or postevent one. 
            
            Returns:
                Path: path of the output DHM.

        """
        logger.debug(f"Starting {'pre-event' if pre_event else 'post-event'} DHM extraction step...")
        check_file(input_dsm)
        suffix = "_pre" if pre_event else "_post"
        dtm_outdir = Path(self.pipeline_output_dir, f"dtm_extraction{suffix}") 
        if not dtm_outdir.exists():
            dtm_outdir.mkdir()
        updates = {
        "step_output_dir": str(dtm_outdir),
        "save_intermediate_data": self.save_intermediate_data,
        "create_logdir": True,
        "dsm_path": str(input_dsm),
        "output_dir": str(dtm_outdir),
        "generate_dhm": True
        }
        dtm_used_conf = self.prepare_step_config(dtm_extraction_conf, updates, dtm_outdir)
        step = DtmExtraction(input_config=dtm_used_conf)
        step.run()
        dhm_path = Path(dtm_outdir, "dhm.tif")
        check_file(dhm_path)
        if not self.save_intermediate_data:
            remove_file_or_folder(Path(dtm_outdir, "dtm.tif"))
        logger.info(f"{'Pre-event' if pre_event else 'Post-event'} DHM extraction: Done !")
        return dhm_path


    def building_extraction(self, 
                        semantic_segmentation_conf: str|Path, 
                        image: str|Path, 
                        pre_event: bool,
                        dsm: str|Path, 
                        filter_individual_uncertainties: bool = False) -> tuple[Path, Path]:
        """
            Building extraction step of the Bitemporal 2D/3D change detection pipeline. 
            
            Args:
                semantic_segmentation_conf (str|Path): path to the input configuration file.
                image (str|Path): input image Path.
                pre_event (bool): flag indicating if the input image is the prevent or postevent one. 
                roi (dict, optionnal): Region of interest. 

            Returns:
                tuple: path of the buildings mask and the associated uncertainty.

        """
        logger.debug(f"Starting {'pre-event' if pre_event else 'post-event'} Building extraction step...")
        suffix = "_pre" if pre_event else "_post"
        semantic_outdir = Path(self.pipeline_output_dir, f"semantic_segmentation{suffix}") 
        if not semantic_outdir.exists():
            semantic_outdir.mkdir()
        updates = {
        "step_output_dir": str(semantic_outdir),
        "save_intermediate_data": self.save_intermediate_data,
        "create_logdir": False,
        "input_img_path": str(image),
        "reprojection_dsm": str(dsm)
        }
        semantic_used_conf = self.prepare_step_config(semantic_segmentation_conf, updates, semantic_outdir)
        step = SemanticSegmentation(input_config=semantic_used_conf)
        step.run()   
        buildings_path = Path(semantic_outdir, "labels.tif")
        uncertainty_path = Path(semantic_outdir, "predictive_entropy.tif")

        uncertainty_refined, unc_thresh = self.uncertainty_filter_seg(self.config['uncertainty_filter_conf'], buildings_path, uncertainty_path, pre_event)

        if filter_individual_uncertainties:
            # Filter uncertain buildings in sensor geometry individually
            #print("uncertainty refined path (sensor)", uncertainty_refined, buildings_path)
            uncertainty_filtered_reprojected_path =  uncertainty_refined.with_stem(uncertainty_refined.stem + "_reprojected")
            logger.debug("Reproject uncertainty filtered buildings to ground geometry...")
            step.sensor_to_ground_mask_projection(uncertainty_refined, uncertainty_filtered_reprojected_path)
            logger.debug("Reprojection done!")
            #print("uncertainty filtered reprojected", uncertainty_filtered_reprojected_path, uncertainty_refined.with_stem(uncertainty_refined.stem + "_unfiltered_reprojected"))
            #step.sensor_to_ground_mask_projection(buildings_path, uncertainty_refined.with_stem(uncertainty_refined.stem + "_unfiltered_reprojected"))

            check_file(uncertainty_filtered_reprojected_path)
            check_file(uncertainty_path)

            logger.info(f"{'Pre-event' if pre_event else 'Post-event'} Building extraction: Done !")
            return uncertainty_filtered_reprojected_path, uncertainty_path

        else:
            # Reproject binarized uncertainty in order to filter 
            uncertainty_filtered_reprojected_path = uncertainty_path.with_stem(uncertainty_path.stem + "_binarized_reprojected")
            buildings_reprojected_path = buildings_path.with_stem(buildings_path.stem + "_reprojected")
            logger.debug("Reproject binarized uncertainties to ground geometry...")
            step.sensor_to_ground_mask_projection(uncertainty_path, uncertainty_filtered_reprojected_path, binarization_thresh=unc_thresh)
            logger.debug("Reproject buildings to ground geometry...")
            step.sensor_to_ground_mask_projection(buildings_path, buildings_reprojected_path)
            logger.debug("Reprojection done!")
            check_file(uncertainty_filtered_reprojected_path)
            check_file(buildings_reprojected_path)
            logger.debug(f"{'Pre-event' if pre_event else 'Post-event'} Building extraction: Done !")

            return buildings_reprojected_path, uncertainty_filtered_reprojected_path


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
        binary_building_diff_path = Path(self.pipeline_output_dir, "2D_binary_raw_diff.tif")
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

    def uncertainty_filter_diff(self, 
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
        updates["out_name"] = f"{Path(raster_to_filter_path).stem}_diff_uncertainty_pre_refined{Path(raster_to_filter_path).suffix}"
        uncertainty_pre_used_conf = self.prepare_step_config(uncertainty_filter_conf, updates, uncertainty_f_outdir)
        step = ThresholdFilter(input_config=uncertainty_pre_used_conf)
        pre_out_path = step.run()

        # Post-event filtering
        uncertainty_pre_used_conf.rename(str(uncertainty_pre_used_conf).replace('filter_used_conf', 'pre_filter_used_conf'))
        updates["raster_to_filter"] = str(pre_out_path)
        updates["filter"] = str(uncertainty_post)
        updates["out_name"] = updates["out_name"].replace('_diff_uncertainty_pre_refined','_uncertainty_refined')
        uncertainty_post_used_conf = self.prepare_step_config(uncertainty_filter_conf, updates, uncertainty_f_outdir)
        step = ThresholdFilter(input_config=uncertainty_post_used_conf)
        out_path = step.run()
        uncertainty_post_used_conf.rename(str(uncertainty_post_used_conf).replace('filter_used_conf', 'post_filter_used_conf'))
        logger.info("Uncertainty filter: Done !")
        return out_path

    def uncertainty_filter_seg(self, 
                           uncertainty_filter_conf: str|Path, 
                           raster_to_filter_path: str|Path, 
                           uncertainty: str|Path, 
                           pre_event: bool) -> Path:
        """ 
            2D uncertainty filtering step of the Bitemporal 2D/3D change detection pipeline. 
            
            Args:
                uncertainty_filter_conf (str|Path): path to the input configuration file.
                raster_to_filter_path (str|Path): path to the input raster change map that will be filtered.
                uncertainty_pre (str|Path): Path to the 2D uncertainty.

            
            Returns:
                Path: path of the uncertainty refined change map.

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
        check_file(uncertainty)
        updates["filter"] = str(uncertainty)
        updates["out_name"] = f"{Path(raster_to_filter_path).stem}_uncertainty_refined_{'pre-event' if pre_event else 'post-event'}{Path(raster_to_filter_path).suffix}"
        uncertainty_used_conf = self.prepare_step_config(uncertainty_filter_conf, updates, uncertainty_f_outdir)
        step = ThresholdFilter(input_config=uncertainty_used_conf)
        out_path = step.run()
        unc_thresh = step.config["threshold"]
        logger.info("Uncertainty filter: Done !")
        return out_path, unc_thresh


    def compute_3D_diff(self, 
                       dhm_pre: str|Path, 
                       dhm_post: str|Path) -> tuple[Path, Path]:
        """
            Compute the raw (in meter) and binary 3D difference between the two provided DHM. 
            
            Args:
                dhm_pre (str|Path): path to the input pre-event DHM.
                dhm_post (str|Path): path to the input post-event DHM.

            Returns:
                tuple: path of the raw 3D diff and path to the corresponding binary 3D diff.

        """
        raw_diff_path = Path(self.pipeline_output_dir, "3D_raw_diff.tif")
        binary_raw_diff_path = Path(self.pipeline_output_dir, "3D_binary_raw_diff.tif")
        with rasterio.open(dhm_pre) as dhm_pre_ds:
            with rasterio.open(dhm_post) as dhm_post_ds:
                dhm_pre = dhm_pre_ds.read(1)
                dhm_post = dhm_post_ds.read(1)
                raw_diff = dhm_post - dhm_pre
                # Raw diff writing
                if 'nodata' in dhm_pre_ds.profile.keys() and 'nodata' in dhm_post_ds.profile.keys():
                    raw_diff[np.logical_or(dhm_pre == dhm_pre_ds.profile['nodata'], dhm_post == dhm_post_ds.profile['nodata'])] = dhm_pre_ds.profile['nodata']
                with rasterio.open(raw_diff_path, "w", **dhm_pre_ds.profile) as out_ds:
                    out_ds.write(raw_diff, 1)
                # Binary diff writing
                binary_diff = np.zeros_like(raw_diff, dtype=bool)
                binary_diff[raw_diff!=0] = 1
                if 'nodata' in dhm_pre_ds.profile.keys() and 'nodata' in dhm_post_ds.profile.keys():
                    binary_diff[np.logical_or(dhm_pre == dhm_pre_ds.profile['nodata'], dhm_post == dhm_post_ds.profile['nodata'])] = 0
                binary_profile = dhm_pre_ds.profile.copy()
                binary_profile['dtype'] = np.uint8
                binary_profile['nodata'] = None
                with rasterio.open(binary_raw_diff_path, "w", nbits=1, **binary_profile) as out_ds:
                    out_ds.write(binary_diff, 1)
        return raw_diff_path, binary_raw_diff_path


    def elevation_filter(self, 
                         elevation_filter_conf: str|Path, 
                         raster_to_filter_path: str|Path, 
                         raw_diff_path: str|Path) -> Path:
        """
            Elevation filtering step of the Bitemporal 2D/3D change detection pipeline. 
            
            Args:
                elevation_filter_conf (str|Path): path to the input configuration file.
                raster_to_filter_path (str|Path): path to the input raster that will be filtered.
                raw_diff_path (str|Path): path to the 3D raw difference map use as filter.
            
            Returns:
                Path: path of the elevation refined change map.

        """
        logger.info("Starting elevation filtering step...")
        check_file(raster_to_filter_path)
        check_file(raw_diff_path)
        alti_f_outdir = Path(self.pipeline_output_dir, "elevation_filtering")        
        if not alti_f_outdir.exists():
            alti_f_outdir.mkdir()
        if "_refined" in str(raster_to_filter_path):
            out_name = f"{Path(raster_to_filter_path).stem.replace('_refined','_alti_refined')}{Path(raster_to_filter_path).suffix}"
        else:    
            out_name = f"{Path(raster_to_filter_path).stem}_alti_refined{Path(raster_to_filter_path).suffix}"
        updates = {
        "step_output_dir": str(alti_f_outdir),
        "save_intermediate_data": self.save_intermediate_data,
        "raster_to_filter": str(raster_to_filter_path),
        "filter": str(raw_diff_path),
        "out_name": out_name
        }
        # If the elevation threshold is not defined, use the default value: 4*planimetric resolution
        parser = ConfigParser()
        config = parser.read(elevation_filter_conf)
        if "thresold" not in config.keys():
            with rasterio.open(raster_to_filter_path) as rtf_ds:
                # Approximate the altimetric resolution as 2*plannimetric_resolution
                altimetric_resolution = np.abs(2*rtf_ds.profile["transform"][0]) 
                elevation_threshold = 2 * altimetric_resolution
                # Since we want to keep only negative changes, we use a - sign
                elevation_threshold = -elevation_threshold
                logger.debug(f"No threshold value provided, use default value: {elevation_threshold}m (2*2*{rtf_ds.profile['transform'][0]})")
                updates["threshold"] = float(elevation_threshold)
        elevation_filter_used_conf = self.prepare_step_config(elevation_filter_conf, updates, alti_f_outdir)
        step = ThresholdFilter(input_config=elevation_filter_used_conf)
        out_path = step.run()
        logger.info("Elevation filter: Done !")
        return out_path


    def ambiguity_filter(self, 
                         ambiguity_filter_conf: str|Path, 
                         raster_to_filter_path: str|Path, 
                         ambiguity_pre: str|Path,
                         ambiguity_post: str|Path) -> Path:
        """
            3D ambiguity filtering step of the Bitemporal 2D/3D change detection pipeline. 
            
            Args:
                ambiguity_filter_conf (str|Path): path to the input configuration file.
                raster_to_filter_path (str|Path): path to the input raster change map that will be filtered.
                ambiguity_pre (str|Path): Path to the CARS pre-event DSM ambiguity.
                ambiguity_post (str|Path): Path to the CARS post-event DSM ambiguity.
            
            Returns:
                Path: path of the ambiguity refined change map.

        """
        logger.info("Starting ambiguity filtering step...")
        check_file(raster_to_filter_path)
        amb_f_outdir = Path(self.pipeline_output_dir, "ambiguity_filtering")
        if not amb_f_outdir.exists():
            amb_f_outdir.mkdir()
        updates = {
        "step_output_dir": str(amb_f_outdir),
        "save_intermediate_data": self.save_intermediate_data,
        "raster_to_filter": str(raster_to_filter_path),
        "greater": True
        }
        check_file(ambiguity_pre)
        updates["filter"] = str(ambiguity_pre)
        if "_refined" in str(raster_to_filter_path):
            out_name = f"{Path(raster_to_filter_path).stem.replace('_refined','_ambiguity_pre_refined')}{Path(raster_to_filter_path).suffix}"
        else:    
            out_name = f"{Path(raster_to_filter_path).stem}_ambiguity_pre_refined{Path(raster_to_filter_path).suffix}"
        updates["out_name"] = out_name
        ambiguity_pre_used_conf = self.prepare_step_config(ambiguity_filter_conf, updates, amb_f_outdir)
        step = ThresholdFilter(input_config=ambiguity_pre_used_conf)
        pre_out_path = step.run()

        # Post-event filtering
        ambiguity_pre_used_conf.rename(str(ambiguity_pre_used_conf).replace('filter_used_conf', 'pre_filter_used_conf'))
        updates["raster_to_filter"] = str(pre_out_path)
        updates["filter"] = str(ambiguity_post)
        updates["out_name"] = updates["out_name"].replace('_ambiguity_pre_refined','_ambiguity_refined')
        ambiguity_post_used_conf = self.prepare_step_config(ambiguity_filter_conf, updates, amb_f_outdir)
        step = ThresholdFilter(input_config=ambiguity_post_used_conf)
        out_path = step.run()
        ambiguity_post_used_conf.rename(str(ambiguity_pre_used_conf).replace('filter_used_conf', 'post_filter_used_conf'))
        logger.info("Ambiguity filter: Done !")
        return out_path


    def ndvi_filter(self, 
                    ndvi_filter_conf: str|Path, 
                    raster_to_filter_path: str|Path, 
                    color_pre: str|Path, 
                    color_post: str|Path) -> Path:
        """
            NDVI filtering step of the Bitemporal 2D/3D change detection pipeline. 
            
            Args:
                ndvi_filter_conf (str|Path): path to the input configuration file.
                raster_to_filter_path (str|Path): path to the input raster change map that will be filtered.
                color_pre (str|Path): Path to the CARS pre-event color image.
                color_post (str|Path): Path to the CARS post-event color image.
            
            Returns:
                Path: path of the NDVI refined change map.

        """
        logger.info("Starting NDVI filtering step...")
        check_file(raster_to_filter_path)
        check_file(color_pre)
        ndvi_f_outdir = Path(self.pipeline_output_dir, "ndvi_filtering")
        if not ndvi_f_outdir.exists():
            ndvi_f_outdir.mkdir()
        updates = {
        "step_output_dir": str(ndvi_f_outdir),
        "save_intermediate_data": self.save_intermediate_data,
        "raster_to_filter": str(raster_to_filter_path),
        "greater": False
        }   

        # Ignore divide warning in NDVI computation
        np.seterr(divide='ignore', invalid='ignore')

        # Pre-event filtering
        with rasterio.open(color_pre) as color_pre_ds:
            color = color_pre_ds.read()
            ndvi = (color[3,:,:] - color[2, :,:])/(color[3,:,:] + color[2,:,:])
            ndvi_profile = color_pre_ds.profile.copy()
            ndvi_profile["count"] = 1
            ndvi_profile["dtype"] = "float32"
            ndvi_pre_path = Path(ndvi_f_outdir, "ndvi_pre_event.tif")
            with rasterio.open(ndvi_pre_path, "w", **ndvi_profile) as out_ds:
                    out_ds.write(ndvi, 1)
            updates["filter"] = str(ndvi_pre_path)
            if "_refined" in str(raster_to_filter_path):
                out_name = f"{Path(raster_to_filter_path).stem.replace('_refined','_ndvi_pre_refined')}{Path(raster_to_filter_path).suffix}"
            else:    
                out_name = f"{Path(raster_to_filter_path).stem}_ndvi_pre_refined{Path(raster_to_filter_path).suffix}"
            updates["out_name"] = out_name
            ndvi_pre_used_conf = self.prepare_step_config(ndvi_filter_conf, updates, ndvi_f_outdir)
            step = ThresholdFilter(input_config=ndvi_pre_used_conf)
            pre_out_path = step.run()

        # Post-event filtering
        check_file(color_post)
        with rasterio.open(color_post) as color_post_ds:
            color = color_post_ds.read()
            ndvi = (color[3,:,:] - color[2, :,:])/(color[3,:,:] + color[2,:,:])
            ndvi_profile = color_post_ds.profile.copy()
            ndvi_profile["count"] = 1
            ndvi_profile["dtype"] = "float32"
            ndvi_post_path = Path(ndvi_f_outdir, "ndvi_post_event.tif")
            with rasterio.open(ndvi_post_path, "w", **ndvi_profile) as out_ds:
                    out_ds.write(ndvi, 1)
            ndvi_pre_used_conf.rename(str(ndvi_pre_used_conf).replace('filter_used_conf', 'pre_filter_used_conf'))
            updates["raster_to_filter"] = str(pre_out_path)
            updates["filter"] = str(ndvi_post_path)
            updates["out_name"] = updates["out_name"].replace('_ndvi_pre_refined','_ndvi_refined')
            ndvi_post_used_conf = self.prepare_step_config(ndvi_filter_conf, updates, ndvi_f_outdir)
            step = ThresholdFilter(input_config=ndvi_post_used_conf)
            out_path = step.run()
            ndvi_post_used_conf.rename(str(ndvi_post_used_conf).replace('filter_used_conf', 'post_filter_used_conf'))

        if not self.save_intermediate_data:
            remove_file_or_folder(ndvi_pre_path)
            remove_file_or_folder(ndvi_post_path)

        logger.info("NDVI filter: Done !")
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
        "convex_hull": True if prefix == "3D" else False,
        "out_name": f"{Path(raster_to_filter_path).stem}_filtered_morphologic{Path(raster_to_filter_path).suffix}"
        }   
        morphologic_used_conf = self.prepare_step_config(morphologic_filter_conf, updates, morphologic_f_outdir)
        step = MorphologicFilter(input_config=morphologic_used_conf)
        out_path = step.run()
        logger.info(f"{prefix} Morphologic filter: Done !")
        return out_path


    def change_maps_merge(self, 
                          boolean_maps_merge_conf: str|Path, 
                          change_map_2D: str|Path, 
                          change_map_3D: str|Path) -> Path:
        """
            Change maps merging step of the Bitemporal 2D/3D change detection pipeline. 
            
            Args:
                boolean_maps_merge_conf (str|Path): path to the input configuration file.
                change_map_2D (str|Path): path to 2D boolean change detection map.
                change_map_3D (str|Path): path to the 3D boolean change detection map.
            
            Returns:
                Path: path of merged change detection map.

        """
        logger.info("Starting change maps merge step...")
        check_file(change_map_2D)
        check_file(change_map_3D)
        merging_outdir = Path(self.pipeline_output_dir, "change_maps_merging")        
        if not merging_outdir.exists():
            merging_outdir.mkdir()
        updates = {
        "step_output_dir": str(merging_outdir),
        "save_intermediate_data": self.save_intermediate_data,
        "layer1": str(change_map_2D),
        "layer2": str(change_map_3D),
        "operator": "OR",
        "out_name": "bitemporal_2D_3D_final_change_map.tif"
        }
        elevation_filter_used_conf = self.prepare_step_config(boolean_maps_merge_conf, updates, merging_outdir)
        step = BooleanMapsMerge(input_config=elevation_filter_used_conf)
        out_path = step.run()
        logger.info("Merging step: Done !")
        return out_path
         

    def prepare_visualization(self) -> None:
        """
            Prepare a directory with all the files to display and pre-processed them for visualization.

        """
        visualization_outdir = Path(self.pipeline_output_dir, "data_visualization")
        pre_event_outdir = Path(visualization_outdir, "pre_event")
        post_event_outdir = Path(visualization_outdir, "post_event")
        _3d_outdir = Path(visualization_outdir, "3D")
        for dir in [visualization_outdir, pre_event_outdir, post_event_outdir, _3d_outdir]:
            if not dir.exists():
                dir.mkdir()
        visu_filenames = {
            'pre_event': {
                'image':{
                    'src': Path(self.pipeline_output_dir, 'dsm_extraction_pre', 'color.tif'),
                    'dst': Path(pre_event_outdir, 'image.tif')
                },
                'buildings':{
                    'src': Path(self.pipeline_output_dir, 'semantic_segmentation_pre', 'labels_reprojected.tif'),
                    'dst': Path(pre_event_outdir, 'buildings.tif')
                },
                'uncertainty':{
                    'src': Path(self.pipeline_output_dir, 'semantic_segmentation_pre', 'predictive_entropy_binarized_reprojected.tif'),
                    'dst': Path(pre_event_outdir, 'binary_uncertainty.tif')
                },
                '2D_diff':{
                    'src': Path(self.pipeline_output_dir, '2D_morphologic_filtering', '2D_binary_raw_diff_uncertainty_refined_filtered_morphologic.tif'),
                    'dst': Path(pre_event_outdir, '2D_diff.tif')
                },
                'combined_diff':{
                    'src': Path(self.pipeline_output_dir, 'change_maps_merging', 'bitemporal_2D_3D_final_change_map.tif'),
                    'dst': Path(pre_event_outdir, 'combined_diff.tif')
                }
            },
            'post_event': {
                'image':{
                    'src': Path(self.pipeline_output_dir, 'dsm_extraction_post', 'color.tif'),
                    'dst': Path(post_event_outdir, 'image.tif')
                },
                'buildings':{
                    'src': Path(self.pipeline_output_dir, 'semantic_segmentation_post', 'labels_reprojected.tif'),
                    'dst': Path(post_event_outdir, 'buildings.tif')
                },
                'uncertainty':{
                    'src': Path(self.pipeline_output_dir, 'semantic_segmentation_post', 'predictive_entropy_binarized_reprojected.tif'),
                    'dst': Path(post_event_outdir, 'binary_uncertainty.tif')
                },
                '2D_diff':{
                    'src': Path(self.pipeline_output_dir, '2D_morphologic_filtering', '2D_binary_raw_diff_uncertainty_refined_filtered_morphologic.tif'),
                    'dst': Path(post_event_outdir, '2D_diff.tif')
                },
                'combined_diff':{
                    'src': Path(self.pipeline_output_dir, 'change_maps_merging', 'bitemporal_2D_3D_final_change_map.tif'),
                    'dst': Path(post_event_outdir, 'combined_diff.tif')
                }
            },
            '3D': {
                'dhm_pre_event': {
                    'src': Path(self.pipeline_output_dir, 'dtm_extraction_pre', 'dhm.tif'),
                    'dst': Path(_3d_outdir, 'dhm_pre_event.tif')
                },
                'dhm_post_event': {
                    'src': Path(self.pipeline_output_dir, 'dtm_extraction_post', 'dhm.tif'),
                    'dst': Path(_3d_outdir, 'dhm_post_event.tif')
                },
                'ambiguity_pre_event': {
                    'src': Path(self.pipeline_output_dir, 'dsm_extraction_pre', 'confidence_from_ambiguity_cars_1.tif'),
                    'dst': Path(_3d_outdir, 'ambiguity_pre_event.tif')
                },
                'ambiguity_post_event': {
                    'src': Path(self.pipeline_output_dir, 'dsm_extraction_post', 'confidence_from_ambiguity_cars_1.tif'),
                    'dst': Path(_3d_outdir, 'ambiguity_post_event.tif')
                },
                '3D_diff': {
                    'src': Path(self.pipeline_output_dir, '3D_morphologic_filtering', '3D_binary_raw_diff_alti_ambiguity_ndvi_refined_filtered_morphologic.tif'),
                    'dst': Path(_3d_outdir, '3D_diff.tif')
                },
                'elevation_diff': {
                    'src': Path(self.pipeline_output_dir, '3D_raw_diff.tif'),
                    'dst': Path(_3d_outdir, 'elevation_diff.tif')
                },
                '2D_diff':{
                    'src': Path(self.pipeline_output_dir, '2D_morphologic_filtering', '2D_binary_raw_diff_uncertainty_refined_filtered_morphologic.tif'),
                    'dst': Path(_3d_outdir, '2D_diff.tif')
                },
                'combined_diff':{
                    'src': Path(self.pipeline_output_dir, 'change_maps_merging', 'bitemporal_2D_3D_final_change_map.tif'),
                    'dst': Path(_3d_outdir, 'combined_diff.tif')
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

        diff_path = visu_filenames['3D']['elevation_diff']['dst']
        with rasterio.open(diff_path) as diff_ds:
            image = diff_ds.read(1).astype(float)
            profile = diff_ds.profile.copy()
            if 'nodata' in diff_ds.profile.keys():
                image = np.where(image == diff_ds.profile['nodata'], 0, image)
                del profile['nodata']
            # Fixed elevation thresholds
            image = self.normalize(image, min=-10,max=10)
            profile['dtype'] = np.uint8
            with rasterio.open(diff_path, "w", **profile) as out_ds:
                out_ds.write(image, 1)

        # For ambiguity, we use the same color ramp
        amb_pre_path = visu_filenames['3D']['ambiguity_pre_event']['dst']
        amb_post_path = visu_filenames['3D']['ambiguity_post_event']['dst']
        with rasterio.open(amb_pre_path) as amb_pre_ds:
            with rasterio.open(amb_post_path) as amb_post_ds:
                profile = amb_post_ds.profile.copy()
                ambiguity_pre = amb_pre_ds.read(1).astype(float)
                ambiguity_post = amb_post_ds.read(1).astype(float)
                if 'nodata' in amb_pre_ds.profile.keys():
                    ambiguity_pre[ambiguity_pre == amb_pre_ds.profile['nodata']] = np.nan
                if 'nodata' in amb_post_ds.profile.keys():
                    ambiguity_post[ambiguity_post == amb_post_ds.profile['nodata']] = np.nan
                    del profile['nodata']
                amb_pre_min, amb_pre_max = np.nanpercentile(ambiguity_pre, 2), np.nanpercentile(ambiguity_pre, 98)
                amb_post_min, amb_post_max = np.nanpercentile(ambiguity_post, 2), np.nanpercentile(ambiguity_post, 98)
                amb_min = np.max([amb_pre_min, amb_post_min])
                amb_max = np.min([amb_pre_max, amb_post_max])
                ambiguity_pre = self.normalize(ambiguity_pre, min=amb_min,max=amb_max)
                ambiguity_post = self.normalize(ambiguity_post, min=amb_min,max=amb_max)
                profile['dtype'] = np.uint8
                with rasterio.open(amb_pre_path, "w", **profile) as out_ds:
                    out_ds.write(ambiguity_pre, 1)
                with rasterio.open(amb_post_path, "w", **profile) as out_ds:
                    out_ds.write(ambiguity_post, 1)
        
        # For DHM, we use the same color ramp
        dhm_pre_path = visu_filenames['3D']['dhm_pre_event']['dst']
        dhm_post_path = visu_filenames['3D']['dhm_post_event']['dst']
        with rasterio.open(dhm_pre_path) as dhm_pre_ds:
            with rasterio.open(dhm_post_path) as dhm_post_ds:
                profile = dhm_post_ds.profile.copy()
                dhm_pre = dhm_pre_ds.read(1).astype(float)
                dhm_post = dhm_post_ds.read(1).astype(float)
                if 'nodata' in dhm_pre_ds.profile.keys():
                    dhm_pre[dhm_pre == dhm_pre_ds.profile['nodata']] = np.nan
                if 'nodata' in dhm_post_ds.profile.keys():
                    dhm_post[dhm_post == dhm_post_ds.profile['nodata']] = np.nan
                    del profile['nodata']
                dhm_pre_min, dhm_pre_max = np.nanpercentile(dhm_pre, 2), np.nanpercentile(dhm_pre, 98)
                dhm_post_min, dhm_post_max = np.nanpercentile(dhm_post, 2), np.nanpercentile(dhm_post, 98)
                dhm_min = np.min([dhm_pre_min, dhm_post_min])
                dhm_max = np.max([dhm_pre_max, dhm_post_max])
                dhm_pre = self.normalize(dhm_pre, min=dhm_min,max=dhm_max)
                dhm_post = self.normalize(dhm_post, min=dhm_min,max=dhm_max)
                profile['dtype'] = np.uint8
                with rasterio.open(dhm_pre_path, "w", **profile) as out_ds:
                    out_ds.write(dhm_pre, 1)
                with rasterio.open(dhm_post_path, "w", **profile) as out_ds:
                    out_ds.write(dhm_post, 1)

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
        
