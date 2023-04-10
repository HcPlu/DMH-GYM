# -*- coding:utf-8 _*-

import numpy as np

class Site(object):
    def __init__(self, name, type, coord):
        self.name = name
        self.type = type
        self.coord = coord

    @classmethod
    def toObj(cls,site):
        return cls(site["name"],site["type"],np.array(site["coods"]))

    def _toJson(self):
        site={}
        site["name"] = self.name
        site["type"] = self.type
        site["coods"] = [int(self.coord[0]), int(self.coord[1])]
        return site
    def get_coord(self):
        return self.coord