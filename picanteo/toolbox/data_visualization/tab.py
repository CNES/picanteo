import geopandas as gpd

import panel as pn
from ipywidgets import jslink, Label, IntSlider
from ipyleaflet import LocalTileLayer

from picanteo.toolbox.data_visualization.map import LeafletMap  # ,LonBoardLeafMap
from abc import ABC, abstractmethod

pn.extension("ipywidgets")


class TabMaps():
    def __init__(self, Tabinfo, settings, local_data_dir, map_height: int = 600):

        self.tab_title = Tabinfo['Name']
        self.list_2D = Tabinfo.get('2D', [])
        self.list_3D = Tabinfo.get('3D', [])
        self.settings = settings
        self.local_data_dir = local_data_dir
        self.map_height = self.settings['map_height']
        self.localhost = self.settings['localhost']
        self.map_2D_dict = {}
        self.map_3D_dict = {}
        self.add_maps()
        self.synch_all()

    def add_maps(self):

        if self.list_2D:
            for m2D in self.list_2D.values():
                map_2D = LeafletMap(map_id=m2D['id'], name=m2D['name'], center=[43, 3], zoom=8, height=self.map_height)
                map_2D.add_layers(localhost=self.localhost, data_dir=m2D['path'],
                                  localhost_repo=self.local_data_dir) #settings['local_data_dir'])

                self.map_2D_dict[m2D["name"]] = map_2D

    def create_tab(self):

        columns_2D = [pn.Column(name, map_obj.map) for name, map_obj in
                      self.map_2D_dict.items()]  # pn.Column('Date 1',m.map),pn.Column('Date 2',m2.map)
        columns_3D = [pn.Column(name, pn.pane.IPyWidget(map_obj.map, sizing_mode="stretch_both")) for name, map_obj in
                      self.map_3D_dict.items()]  # pn.pane.IPyWidget(list(map_3D_dict.values())[0].map, sizing_mode="stretch_both")
        res = pn.Row(*columns_2D, *columns_3D, sizing_mode="stretch_both", name=self.tab_title)
        return res

    # SYNCHRONISATION
    def synch_2D(self, value):
        maps_to_synch = [maps.map for maps in self.map_2D_dict.values() if maps != value.owner]
        if value['name'] == 'zoom':  # zoom synchronisation
            for maps in maps_to_synch:
                maps.zoom = value['new']
        else:  # center synchronisation
            for maps in maps_to_synch:
                maps.center = value['new'];

    def synch_all_2Ds(self):

        for maps in self.map_2D_dict.values():

            maps.map.observe(self.synch_2D, names=['center', 'zoom'])

    def synch_all(self):

        if self.map_2D_dict:
            # self.synch_all_2Ds_new()
            self.synch_all_2Ds()


class TabText():

    def __init__(self, Tabinfo):
        self.tab_title = Tabinfo['Name']
        self.text = Tabinfo['Texte']

    def create_tab(self):

        res = self.text
        return pn.Row(res, name=self.tab_title)

