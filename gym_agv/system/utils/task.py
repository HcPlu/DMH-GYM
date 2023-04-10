# -*- coding:utf-8 _*-

import numpy as np
from .node import Site
class Task(object):
    def __init__(self, name, s, e, d, tao, due, G, handling):
        self.distance_type = 1 # 0 Manhattan 1 Euclidean
        self.name = name
        self.s=s
        self.e=e
        self.d=d
        self.tao=tao
        self.G=G
        self.handling=handling
        self.due = due
        self.assigned=0
        self.waiting_time = 0
        self.is_back = 0

    @classmethod
    def toObj(cls,site):
        return cls(site["name"],Site.toObj(site["s"]),Site.toObj(site["e"]),site["d"],site["tao"],site["G"],site["handling"],site["due"])

    def _toJson(self):
        site={}
        site["name"] = self.name
        site["s"] = self.s._toJson()
        site["e"] = self.e._toJson()
        site["d"] = self.d
        site["tao"] = self.tao
        site["G"] = self.G
        site["handling"] = self.handling
        site["due"] = self.due
        return site

    def is_feasible_choice_constraint(self, vehicle):
        if self.assigned!=0:
            return 0

        if vehicle.status!=0:
            return 0

        # if (vehicle.c>=self.d):
        #     return 1
        # else:
        #     return 0
        return 1

    def is_feasible_choice_simple(self, vehicle):
        # todo: more constraints
        #  Hu's
        if self.assigned!=0:
            return 0

        if vehicle.status!=0:
            return 0
        return 1

    def is_released(self,frame):
        self.release_time=frame

    def is_assigned(self):
        self.assigned = 1

    def is_reset(self):
        self.assigned = 0

    def update_wait(self,time):
        self.waiting_time += time

    def get_actual_distance(self,vehicle,running_map):
        return  int(running_map[vehicle.current_site][self.s.name]["distance"]+running_map[self.s.name][self.e.name]["distance"])

    def get_minimum_distance(self,running_map):
        return int(running_map[self.s.name][self.e.name]["distance"])

    def get_start_distance(self,current_site,running_map):
        return int(running_map[current_site][self.s.name]["distance"])

    def get_start_route(self,current_site,running_map):
        return running_map[current_site][self.s.name]["route"]

    def get_minimum_route(self,running_map):
        return running_map[self.s.name][self.e.name]["route"]

    def _Manhattan_distance(self,a,b):
        return np.linalg.norm((a-b),ord=1)

    def _Euclidean_distance(self,a,b):
        return np.linalg.norm(a-b,ord=2)
    def __str__(self):
        return str((self.s.name,self.e.name,self.due))

class AssignedTask(Task):
    def __init__(self,task,vehicle):
        super(AssignedTask, self).__init__(task.name, task.s, task.e, task.d, task.tao, task.due, task.G, task.handling, )
        self.vehicle=vehicle
        self.minimum_time = 0
        self.actual_time = 0
        self.count_time = 0
        self.assigned_time = 0
        self.start_time = 0
        self.finished_time = 0
        self.delay = 0
        self.delay2 = 0
        self.a_star_route = []
        self.is_back=task.is_back
        self.assigned = task.assigned
        self.waiting_time = task.waiting_time


    def set_time(self,running_map):
        self.set_minimum_time(running_map)
        self.set_actual_time(running_map)

    def set_a_star_route(self,current_site,running_map):
        self.a_star_route =self.get_start_route(current_site,running_map=running_map)[:-1]+self.get_minimum_route(running_map=running_map)

    def set_minimum_time(self,running_map):
        self.minimum_time= self.get_minimum_distance(running_map=running_map) / self.vehicle.vl + self.handling

    def set_actual_time(self,running_map):
        self.actual_time= self.get_actual_distance(self.vehicle,running_map=running_map)/self.vehicle.vl+self.handling

    def set_start_time(self,time):
        self.start_time = time

    def set_assigned_time(self,time):
        self.assigned_time = time

    def set_delay_time(self):
        # denote the delay time
        if self.due!=-1:
            delay = self.finished_time -(self.tao+self.due)
            self.delay = 0 if delay<0 else delay
            self.set_delay_time2()

    def set_delay_time2(self):
        #relaxed delay
        if self.due!=-1:
            self.delay2 = self.finished_time-(self.tao+self.due)

    def update(self,frame,rate):
        if self.count_time==0:
            # start from 0
            self.set_start_time(frame-float(1/rate))

        self.count_time+=float(1/rate)
        if self.count_time>=self.actual_time:
            self.finished_time=frame
            self.set_delay_time()
            if self.is_back:
                #back
                return -2
            else:
                return -1 #finished
        else:
            return self.count_time


        # print(self.waiting_time)

