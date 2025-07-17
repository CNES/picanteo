import panel as pn
from picanteo.toolbox.data_visualization.tab import TabMaps,TabText


class Page():
    def __init__(self, PageInfo, settings):
        self.tabs = pn.Tabs(dynamic=True)
        self.title = PageInfo['Title']
        self.PageInfo = PageInfo
        self.settings = settings
        self.local_data_dir = PageInfo['local_data_dir']
    def fill_page(self):

        for tab in self.PageInfo.values():
            if type(tab) == dict:
                if tab['Map']:
                    instance_tab = TabMaps(Tabinfo=tab,settings=self.settings, local_data_dir = self.local_data_dir)
                    res = instance_tab.create_tab()
                else:
                    instance_tab = TabText(Tabinfo=tab)
                    res = instance_tab.create_tab()
                self.tabs.append(res)
        return self.tabs
