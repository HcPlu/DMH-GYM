# -*- coding:utf-8 _*-

import re
import numpy as np
import matplotlib.pyplot as plt

from .aStar import AStar
from .node import Site
from .task import Task
from .vehicle import Vehicle,RealVehicle
import json
import logging.config
import matplotlib.patches as mpathes

class Map(object):
    def __init__(self,file_path=None):
        self.id=''
        self.file_path=file_path
        self.site_num=0
        self.task_order = 0
        self.sites = []
        self.rechage_list = []
        self.topology = [] # topology of the map
        self.nodes = [] # all nodes in the map including road nodes
        self.parking_list = []
        self.pickup_list = []
        self.delivery_list = []
        self.raw_tasks = []
        self.task_num = 0
        self.history_generated_tasks = []
        self.node_map_name = {}# access the list by site's name
        self.current_tasks = []
        self.node_map = {} # get the coord
        self.running_map = {} # get route and distance
        self.vehicle_types = []
        self.vehicles = []
        self.charging_list = []
        self.vehicle_num=0



    def constuct_map(self):
        for node in self.nodes:
            self.node_map[node.name]=node.coord
            self.node_map_name[node.name]=node

        for site_x in self.nodes:
            self.running_map[site_x.name] = {}
            for site_y in self.nodes:
                self.running_map[site_x.name][site_y.name]={}

        self.a_star()

    def _Euclidean_distance(self,a,b):
        return np.linalg.norm(a - b, ord=2)

    def _get_route_distance(self,route):
        distance = 0
        for i in range(len(route)-1):
            distance += self._Euclidean_distance(self.node_map[route[i]],self.node_map[route[i+1]])
        return distance

    def a_star(self):
        for site_x in self.nodes:
            for site_y in self.nodes:
                if site_x==site_y:
                    self.running_map[site_x.name][site_y.name]["distance"] = 0
                    self.running_map[site_x.name][site_y.name]["route"] = [site_x.name]
                    continue
                astar = AStar(self.topology, self.node_map, site_x.name, site_y.name)
                route = astar.a_start()
                distance = self._get_route_distance(route)
                self.running_map[site_x.name][site_y.name]["route"] = route
                self.running_map[site_x.name][site_y.name]["distance"] = distance


    def draw_map(self):
        fig,ax = plt.subplots(tight_layout=True)
        all_node = np.array([np.array(item.get_coord()) for item in self.nodes if not re.search(r"p\d",item.name)])
        rect_all_node = np.array([np.array(item.get_coord()) for item in self.nodes if not re.search(r"p\d", item.name) and item.name!="carport" and item.name!="warehouse"])
        plt.scatter(all_node[:, 0], all_node[:, 1],c="b",alpha=0.5)
        l=10
        w=7

        for i in range(len(rect_all_node)):
            height_y=4
            xy=(rect_all_node[i, 0]-l/2, rect_all_node[i, 1]+height_y)
            rect = mpathes.Rectangle(xy, l, w, color='b',fill=False)
            ax.add_patch(rect)
            coords = np.array([[rect_all_node[i, 0],rect_all_node[i, 1]],[rect_all_node[i, 0],rect_all_node[i, 1]+height_y]])

            plt.plot(coords[:,0],coords[:,1],c="b",alpha=0.5)
        for key in self.topology:
            for next in self.topology[key]:
                line = []
                line.append(self.node_map[key])
                line.append(self.node_map[next])
                line=np.array(line)
                plt.plot(line[:, 0], line[:, 1], c="b", linewidth=1,alpha=1)
        for node in self.node_map:
            # print(re.search(r"p\d",node))
            if re.search(r"p\d",node):
                continue
            elif node=="carport" :
                ll=20
                lw=6
                height_y = lw
                plt.text(self.node_map[node][0] - ll/ 2, self.node_map[node][1] + lw+1, node, fontsize=12)
                xy = (self.node_map[node][0]- ll/ 2, self.node_map[node][1] + height_y)
                rect = mpathes.Rectangle(xy, ll, lw, color='b', fill=False)
                ax.add_patch(rect)
                coords = np.array([[self.node_map[node][0], self.node_map[node][1]], [self.node_map[node][0], self.node_map[node][1] + height_y]])
                plt.plot(coords[:, 0], coords[:, 1], c="b",alpha=0.5)
            elif  node=="warehouse":
                ll=27
                lw=6
                height_y = lw
                shift_x = ll/ 2-6
                plt.text(self.node_map[node][0] - shift_x, self.node_map[node][1] + lw+1, node, fontsize=12)
                xy = (self.node_map[node][0]- shift_x, self.node_map[node][1] + lw)
                rect = mpathes.Rectangle(xy, ll+1, lw, color='b', fill=False)
                ax.add_patch(rect)
                coords = np.array([[self.node_map[node][0], self.node_map[node][1]], [self.node_map[node][0], self.node_map[node][1] + height_y]])
                plt.plot(coords[:, 0], coords[:, 1], c="b",alpha=0.5)
            else:
                plt.text(self.node_map[node][0]-l/2+1, self.node_map[node][1]+5, node, fontsize=13)
        # plt.show()
        return fig,ax

    def init(self):
        for v in self.vehicles:
            v.map = self
            v.start_site="carport"
            v.current_coord = self.node_map["carport"]
            v.clear()



    def read_map(self,path,draw=0):
        with open(path,"r") as f:
            res=json.load(f)
        self.id = res["id"]
        self.vehicle_num=res["vehicle_num"]
        self.site_num=res["site_num"]
        self.site_num=res["site_num"]
        self.nodes = [Site.toObj(s) for s in res["map"]["nodes"]]
        self.topology = res["map"]["topology"]
        self.sites = [Site.toObj(s) for s in res["map"]["sites"]]
        self.parking_list = [Site.toObj(s) for s in res["map"]["pickup"]]
        self.delivery_list = [Site.toObj(s) for s in res["map"]["delivery"]]
        self.charging_list = [Site.toObj(s) for s in res["map"]["charging"]]
        self.vehicles = [RealVehicle(Vehicle.toObj(v)) for v in res["vehicle"]["vehicles"]]
        self.raw_tasks = [Task.toObj(task) for task in res["task"]["task_list"]]
        self.task_num = res["task"]["task_num"]
        for i,v in enumerate(self.vehicles):
            v.name="v%d"%i
        self.vehicle_types = [Vehicle.toObj(v) for v in res["vehicle"]["vehicle_types"]]
        self.constuct_map()
        self.init()
        if draw==1:
            self.draw_map()
        distance = 0
        for st in self.nodes:
            for se in self.nodes:
                if st.name is not se.name:
                    distance =max(distance,self.running_map[st.name][se.name]["distance"])
        self.max_distance = distance




if __name__=="__main__":
    test_map=Map()


