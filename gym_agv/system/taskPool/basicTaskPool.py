# -*- coding:utf-8 _*-


import numpy as np


class BasicTaskPool(object):
    def __init__(self,max_per_time,max_task_num,map,seed=12):
        self.seed = seed
        self.max_per_time = max_per_time
        self.max_task_num = max_task_num
        self.map = map
        self.site_num = self.map.site_num
        self.vehicle_num = self.map.vehicle_num
        self.release_order = 0
        self.history_generated_tasks = []
        self.current_tasks = []
        self.assigned_tasks = []
        self.pool_pointer = 0
        self.released_num = 0
        self.res_num = self.max_task_num
        self.history_interval = []
        self.load_task = False



    def clear(self):
        self.res_num = self.max_task_num
        self.release_order = 0
        self.history_generated_tasks = []
        self.released_num = 0
        self.current_tasks = []
        self.assigned_tasks = []
        self.pool_pointer = 0
        self.gen_task_pool()
        self.history_interval = []


    def generate_task(self,order):
        pass

    def release_tasks(self,frame):
        pass

    def read_tasks_4map(self):
        pass

    def gen_task_pool(self):
        #todo: need to optimise running speed
        if self.load_task:
            self.read_tasks_4map()
        else:
            np.random.seed(self.seed)
            for i in range(self.max_task_num):
                self.history_generated_tasks.append(self.generate_task(i))

    def assign_task(self,task_id,vehicle,running_map):
        assigned_task = self.current_tasks[task_id]
        self.assigned_tasks.append(assigned_task)
        assigned_task.is_assigned()
        vehicle.get_task(assigned_task,running_map)

    def get_task(self,free_task):
        self.current_tasks.append(free_task)
        for i,task in enumerate(self.assigned_tasks):
            if task.name ==free_task.name:
                self.assigned_tasks.pop(i)
                break

    def remove_tasks(self,removing_tasks):
        removed_task_list = [self.current_tasks[index].name for index in removing_tasks]
        for index in sorted(removing_tasks,reverse=True):
            self.current_tasks.pop(index)
        return self.current_tasks,removed_task_list


