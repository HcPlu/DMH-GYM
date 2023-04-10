# -*- coding:utf-8 _*-


import logging.config
import numpy as np
root_logger = logging.getLogger("root")
# todo:more rules
class AHP():
    # The task with the longest waiting time will be selected first
    def __init__(self,vehicle_num=-1):
        self.name = "AHP"
        self.vehicle_num=vehicle_num

    def act(self,current_tasks,vehicles,running_map):
        free_vehicles = [(vehicles[i], i) for i in range(len(vehicles)) if vehicles[i].status == 0]
        if len(free_vehicles) == 0 or len(current_tasks) == 0:
            return None
        # root_logger.debug("current # task: " + str(len(current_tasks)))
        decision_list = []
        remain = []
        distance = []
        for task in current_tasks:
            remain.append(task.tao+task.due)
            distance.append(task.get_minimum_distance(running_map))
        ta = np.mean(remain)
        da = np.mean(distance)
        sorted_tasks = []
        w1=0.37
        w2=0.63
        for task in current_tasks:
            sorted_tasks.append((task,w1*((task.tao+task.due)/ta)+w2*((task.get_minimum_distance(running_map))/da)))
        # sorted_tasks = sorted(current_tasks, key=lambda x: x.waiting_time)
        sorted_tasks = sorted(sorted_tasks,key=lambda x:x[1],reverse=True)

        for j, vehicle in enumerate(free_vehicles):
            for i, task in enumerate(sorted_tasks):
                decision_list.append((current_tasks.index(task[0]),free_vehicles[j][0]))

        return decision_list[0]