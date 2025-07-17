from abc import ABC, abstractmethod
import leafmap
from ipyleaflet import Map, WMSLayer, LocalTileLayer
import os
from ipywidgets import jslink, HBox, VBox, Label, Layout, Output

import geopandas as gpd


class BaseMap(ABC):
    def __init__(self, map_id, name, zoom):

        self.map_id = map_id
        self.zoom = zoom
        self.name = name

    @abstractmethod
    def get_map(self, data):

        pass

    @abstractmethod
    def add_layers(self, localhost: str, data_dir: str, localhost_repo: str):

        pass


class LeafletMap(BaseMap):
    def __init__(self, map_id, name, center, zoom, height: int = 600):

        super().__init__(map_id, name, zoom)
        self.center = center
        self.height = str(height) + "px"
        self.map = leafmap.Map(center=self.center, zoom=self.zoom, max_zoom=18, scroll_wheel_zoom=True,
                               height=self.height)
        self.map.add_layer_control()
        # Liste des couches
        self.layers = []

    def add_layers(self, localhost: str, data_dir: str, localhost_repo: str):


        local_dir = os.path.join(localhost_repo, data_dir)
        for data in os.listdir(local_dir):

            if data.endswith('.geojson'):
                print('missing func for geojson')
            else:
                print(os.path.join(localhost, data_dir, data))
                self.add_tiles_local(img_path=os.path.join(localhost_repo, data_dir, data), name=data)


    def add_Tiles(self, path: str, name: str):

        self.map.add(LocalTileLayer(path=os.path.join(path, '{z}/{x}/{y}.png'), tms=True, name=name))

    def add_tiles_local(self, img_path: str, name: str):

        if "image" in img_path:
            self.map.add_raster(img_path, bands=[1, 2, 3], layer_name=name, nodata=0)
        elif "dhm" in img_path:
            params_dict = {"colormap": "terrain", "layer_name": name}
            self.map.add_raster(img_path, colormap="terrain", vmin=0, vmax=255, layer_name=name)  # , **params_dict)
        elif "ambiguity" in img_path:
            self.map.add_raster(img_path, colormap="viridis", vmin=0, vmax=255, layer_name=name)
        else:
            self.map.add_raster(img_path,vmin=0, vmax=255, layer_name=name)
    def get_map(self):
        return self.map

