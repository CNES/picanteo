import geopandas as gpd

import panel as pn
import yaml
from ipywidgets import jslink,Label,IntSlider
from ipyleaflet import LocalTileLayer

from picanteo.toolbox.data_visualization.map import LeafletMap
from picanteo.toolbox.data_visualization.tab import TabMaps,TabText
from picanteo.toolbox.data_visualization.page import Page
import numpy as np
import pandas as pd
import os
import sys

pn.extension("ipywidgets")

if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = "config_CDD_visu.yaml"

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)


#pn.extension(design='bootstrap', global_css=[':root { --design-primary-color: #303F9F; --design-secondary-color: #5C6BC0; --design-surface-color: #EEEEEE; --design-background-color: #E8EAF6;}'])


settings = config['settings']
title = config['title']
logo = "https://cnes.fr/sites/default/files/logos/logo-cnes-blanc-footer.png"

pages = {}
for m,c in config['pages'].items():
    tabs = Page(c,settings=settings)
    pages[tabs.title] = tabs

dropdown = pn.widgets.Select(
    name="Select application",
    options=list(pages.keys()),
    value=list(pages.keys())[0],
)

@pn.depends(dropdown)
def update_plot(selected_plot):

    if selected_plot in pages:
        obj = pages[selected_plot]
        return pn.Card(obj.fill_page(), title=obj.title)
    else:
        return pn.pane.Markdown(f"**Option '{selected_plot}' not specified**", style={'color': 'red'})



template = pn.template.FastListTemplate(
    title = title,
    logo = logo,
    sidebar=[dropdown,],
    header_background="#303F9F",
    theme="dark"
)

template.main.append(
    pn.Row(update_plot)
)


template.servable()
