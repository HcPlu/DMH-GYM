# -*- coding:utf-8 _*-

import copy
import time

from .task import AssignedTask
import numpy as np
from .aStar import AStar
from .node import Site
class Vehicle(object):
    def __init__(self,c,vl,cr,dr,pl,G):
        self.c=c
        self.vl=vl
        self.cr=cr
        self.dr=dr
        self.pl=pl
        # self.wl=wl
        self.G=G
        self.status=0
        self.battery_level=100
        self.history_list = []
        self.waiting_list = []

    @classmethod
    def toObj(cls,vehicle):
        return cls(vehicle["c"],vehicle["vl"],vehicle["cr"],vehicle["dr"],vehicle["pl"],vehicle["G"])

    def _toJson(self):
        vehicle = {}
        vehicle["c"]=self.c
        vehicle["vl"] = self.vl
        vehicle["cr"] = self.cr
        vehicle["dr"] = self.dr
        vehicle["pl"] = self.pl
        vehicle["G"] = self.G
        return vehicle

class RealVehicle(Vehicle):
    def __init__(self,vehicle):
        super(RealVehicle, self).__init__(vehicle.c,vehicle.vl,vehicle.cr,vehicle.dr,vehicle.pl,vehicle.G)
        self.name = ''
        self.map = None
        self.tfrate = 1
        self.start_site = "carport"
        self.current_coord = (0, 0)
        self.current_site = ''
        self.remained_time = 0
        self.status=0
        self.running_route = []
        self.running_route.append(self.start_site)
        self.expected_coords = []
        self.expected_coords.append(self.current_coord)
        self.expected_sites = []
        self.expected_sites.append(self.current_site)
        self.fake_site = None


    def clear(self):
        self.current_site = self.start_site
        self.current_coord=self.map.node_map[self.start_site]
        self.expected_coords = []
        self.expected_coords.append(self.current_coord)
        self.expected_sites = []
        self.expected_sites.append(self.current_site)
        self.remained_time = 0
        self.status=0
        self.battery_level=100
        self.history_list = []
        self.waiting_list = []
        self.running_route = []
        self.running_route.append(self.start_site)

    def check_site(self):
        for site in self.map.node_map:
            if np.linalg.norm(np.array(self.current_coord)-self.map.node_map[site],ord=2)<0.0001:
                return site
        return self.current_site

    def update_coord(self):
        if len(self.expected_coords)>1:
            self.expected_coords.pop(0)
        self.current_coord = self.expected_coords[0]

        if self.expected_sites[0]!="fake_site":
            self.current_site = self.check_site()


    def set_breakdown_site(self):
        pass

    def update_time(self,frame,rate):

        if self.status==2:
            # the AVG is breakdown
            return -3

        if len(self.waiting_list)==0:
            self.is_free()
            return 0
        else:
            # update the coord in real time
            cost_time = self.waiting_list[0].update(frame,rate)
            self.update_coord()
            if cost_time==-1: #finised
                self.current_coord = self.waiting_list[0].e.coord
                self.current_site = self.waiting_list[0].e.name
                self.running_route.append(self.waiting_list[0].e.name)
                # self.waiting_list[0].delay = 0 if self.waiting_list[0].tao+self.waiting_list[0].due-frame>0 else abs(self.waiting_list[0].tao+self.waiting_list[0].due-frame)

                self.history_list.append(self.waiting_list.pop(0))
                self.is_free()

                return -1
            if cost_time==-2:
                # print("back")
                self.current_coord = self.waiting_list[0].e.coord
                self.current_site = self.waiting_list[0].e.name
                self.running_route.append(self.waiting_list[0].e.name)
                # todo: what is the delay ratio??

                self.history_list.append(self.waiting_list.pop(0))
                self.is_free()
                return -2
            else:
                self.is_working()
                return cost_time #time spent

    def get_next_node(self):
        current_next_nodes = self.map.topology[self.current_site]
        fake_coord = np.array(self.fake_site.coord)
        fake_x = fake_coord[0]
        fake_y = fake_coord[1]
        current_x = self.map.node_map[self.current_site][0]
        current_y = self.map.node_map[self.current_site][1]
        for site in current_next_nodes:
            x = self.map.node_map[site][0]
            y = self.map.node_map[site][1]
            if (fake_x-current_x)*(y-current_y)-(fake_y-current_y)*(x-current_x)<0.00001:   #cross product
                if fake_x<=max(x,current_x) and fake_x>=min(x,current_x) and fake_y <=max(y,current_y) and fake_y>=min(y,current_y):
                    return site
        return None

    def get_task_distance(self,task,running_map):
        if self.fake_site != None:
            fake_topology = copy.deepcopy(self.map.topology)
            fake_next_site = self.get_next_node()
            fake_topology[self.fake_site.name] = [self.current_site,fake_next_site]
            fake_topology[self.current_site].append(self.fake_site.name)
            fake_topology[fake_next_site].append(self.fake_site.name)
            fake_node_map = copy.deepcopy(self.map.node_map)
            fake_node_map[self.fake_site.name] = self.fake_site.coord
            astar = AStar(fake_topology, fake_node_map, self.fake_site.name, task.s.name)
            fake_route = astar.a_start() +task.get_minimum_route(running_map=running_map)

            distance = 0
            for i in range(len(fake_route) - 1):
                distance += np.linalg.norm((fake_node_map[fake_route[i]]- fake_node_map[fake_route[i + 1]]),ord=2)
            actual_time = distance/self.vl+task.handling

        else:
            actual_time = task.get_actual_distance(self,running_map)
        return actual_time

    def get_task(self,task,running_map):
        assignedTask = AssignedTask(task, self)
        assignedTask.set_time(running_map)
        if self.fake_site != None:
            fake_topology = copy.deepcopy(self.map.topology)
            fake_next_site = self.get_next_node()
            fake_topology[self.fake_site.name] = [self.current_site,fake_next_site]
            fake_topology[self.current_site].append(self.fake_site.name)
            fake_topology[fake_next_site].append(self.fake_site.name)
            fake_node_map = copy.deepcopy(self.map.node_map)
            fake_node_map[self.fake_site.name] = self.fake_site.coord
            astar = AStar(fake_topology, fake_node_map, self.fake_site.name, assignedTask.s.name)
            fake_route = astar.a_start() +assignedTask.get_minimum_route(running_map=running_map)
            distance = 0
            for i in range(len(fake_route) - 1):
                distance += np.linalg.norm((fake_node_map[fake_route[i]]- fake_node_map[fake_route[i + 1]]),ord=2)

            assignedTask.actual_time =distance/assignedTask.vehicle.vl+assignedTask.handling
            self.expected_coords = self.cal_position(fake_route,fake_node_map)
            self.expected_sites = fake_route
            self.fake_site = None
        else:
            assignedTask.set_a_star_route(self.current_site,running_map)
            self.expected_coords = self.cal_position(assignedTask.a_star_route,self.map.node_map)
            self.expected_sites = assignedTask.a_star_route
        self.waiting_list.append(assignedTask)
        self.running_route.append(self.waiting_list[0].s.name)
        self.is_working()

    def release_task(self):
        self.expected_coords = []
        self.expected_coords.append(self.current_coord)
        self.expected_sites = []
        self.expected_sites.append(self.current_site)
        # print("breakdown",self.current_site)
        self.fake_site = Site("fake_site","fake_site", self.current_coord)
        free_task = self.waiting_list[-1]
        self.waiting_list.pop(-1)
        return free_task

    def cal_position(self,positions,node_map):
        v_coords = []
        for site_index in range(len(positions)-1):
            s_site = positions[site_index]
            s_coord = node_map[s_site]
            e_site = positions[site_index+1]
            e_coord = node_map[e_site]
            t = 0
            max_d = np.linalg.norm(s_coord - e_coord, ord=2)
            if e_coord[1]-s_coord[1]==0:
                while t<max_d/self.vl:
                    v_coords.append([s_coord[0]+(e_coord[0]-s_coord[0])/max_d*self.vl*t,s_coord[1]])
                    t+=1/self.tfrate

            elif (e_coord[0]-s_coord[0]==0):
                while t<max_d/self.vl:
                    v_coords.append([s_coord[0],s_coord[1]+(e_coord[1]-s_coord[1])/max_d*self.vl*t])
                    t+=1/self.tfrate
            else:
                while t<max_d/self.vl:
                    v_coords.append([s_coord[0]+(e_coord[0]-s_coord[0])/max_d*self.vl*t,s_coord[1]+(e_coord[1]-s_coord[1])/max_d*self.vl*t])
                    t+=1/self.tfrate

        return v_coords

    def is_working(self):
        self.status = 1

    def is_free(self):
        self.status = 0

    def is_breakdown(self):
        self.status = 2

    def is_feasible_capability(self,task):
        # todo: feasible capability judgement
        return task.G==self.G

