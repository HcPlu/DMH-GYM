# -*- coding:utf-8 _*-

import copy
from ..utils.task import Task
# from task import Task
import numpy as np
from gym_agv.system.taskPool.basicTaskPool import BasicTaskPool
import time


class mTaskPool(BasicTaskPool):
    def __init__(self,max_per_time,max_task_num,init_tasks_num,map,seed):
        super(mTaskPool, self).__init__(max_per_time, max_task_num, map, seed)
        self.use_fixed_random_seeds = 0
        self.random_seed_run = 0
        np.random.seed(self.seed)
        self.day_time = 3600
        self.perturb_choose_num = 10
        self.fixed_release_points = [np.random.randint(int(self.day_time/self.max_task_num)* i, int(self.day_time/self.max_task_num) * (i + 1)) for i in range(self.max_task_num)]
        # print(len(self.fixed_release_points))
        # self.fixed_release_points = sorted(np.random.randint(0, 60, (60,)) * 60)
        self.init_flag = 1
        self.seeds = np.random.randint(0, 200, size=(2000,))
        self.min_demand = 5
        self.max_demand = 20
        self.a_beta = 1.5
        self.b_beta = 2.5
        # self.current_interval = self._get_interval()
        # self.history_interval.append(self.current_interval)
        # self.last_trigger = 0
        self.release_points = []
        self.init_tasks_num = init_tasks_num
        # self.init_tasks_num = init_tasks_num
        self.set_init_num()
        self.gen_task_pool()
        self.set_release_point()

    # def set_fixed_release_points(self):
    #     np.random.seed(self.seed)
    #     self.fixed_release_points = [np.random.randint(self.max_task_num * i, self.max_task_num * (i + 1)) for i in
    #                                  range(int(self.day_time / self.max_task_num))]
    #     print(self.max_task_num)

    def set_init_num(self):
        if self.use_fixed_random_seeds:
            np.random.seed(self.seeds[self.random_seed_run])
        else:
            t = 1000 * time.time()
            np.random.seed(int(t) % 2 ** 32)
        self.init_tasks_num = np.random.randint(1,self.vehicle_num)
        # print("-------------------------------------------------")
        # print(self.init_tasks_num)

    # def _get_interval(self,loc=80,scale=15):
    #     if self.use_fixed_random_seeds:
    #         np.random.seed(self.seeds[self.random_seed_run][self.seed_index])
    #         self.seed_index += 1
    #     else:
    #         t = 1000 * time.time()
    #         np.random.seed(int(t) % 2 ** 32)
    #     return int(np.random.normal(loc=loc, scale=scale))

    def go_back(self,vehicle):
        if vehicle.current_site != "carport":
            # print("-----------assign back task-------------")
            back_task = Task("task_back", self.map.node_map_name[vehicle.current_site], self.map.node_map_name["carport"], -1, -1, -1, [], 0)
            back_task.is_back = 1
            back_task.is_assigned()
            vehicle.get_task(back_task, self.map.running_map)
            return 1
        return 0

    def interval_random(self,a,b):
        #a<b
        return (b-a)*np.random.random()+a

    def generate_task(self,order):
        #random generate one task based on current map site
        # no due
        name =  "task_" + str(len(self.history_generated_tasks))
        site_pair = np.random.choice(self.site_num,2,replace=False)
        start_site = self.map.sites[site_pair[0]]
        end_site = self.map.sites[site_pair[1]]
        while end_site.name == "warehouse" or end_site.name == "carport" or start_site.name == "carport":
            site_pair = np.random.choice(self.site_num, 2, replace=False)
            start_site = self.map.sites[site_pair[0]]
            end_site = self.map.sites[site_pair[1]]
        beta_x = np.random.beta(self.a_beta, self.b_beta)
        demand = (self.max_demand - self.min_demand) * beta_x + self.min_demand
        # demand = np.random.randint(5,20)
        tao = -1
        # due = self.map.running_map[start_site.name][end_site.name]["distance"]+np.random.randint(400,500)
        due = self.map.running_map[start_site.name][end_site.name]["distance"] + np.random.randint(200, 300)
        # due = np.random.randint(200, 300)
        # due = self.map.running_map[start_site.name][end_site.name]["distance"] * (1 + self.interval_random(1, 1.5))
        G = ["A","B","C"]
        handling = 0
        task = Task(name, start_site, end_site, int(demand), tao, int(due), G, handling)
        return task

    def release_tasks(self,frame):
        np.random.seed(self.seed)
        trigger = self.trigger_point(frame)
        if len(self.current_tasks) >= self.max_per_time:
            return self.current_tasks,0
        else:
            while_flag = 0
            while trigger:
                add_num = 1
                # if frame == 0 and self.release_order==0:
                #     add_num = self.init_tasks_num
                end_pointer = min(self.pool_pointer+add_num,self.max_task_num)
                releasing_tasks = self.history_generated_tasks[self.pool_pointer:end_pointer]
                self.res_num = self.max_task_num - end_pointer
                self.current_tasks += releasing_tasks
                # print(frame,len(self.current_tasks),self.release_order)
                self.pool_pointer = end_pointer
                for task in self.current_tasks:
                    if task.tao == -1:
                        task.tao = frame
                trigger = self.trigger_point(frame)
                # print(trigger,frame)
                while_flag = 1
                # self.release_order+=1
            # if while_flag:
            #     print("----------------------------")
        return self.current_tasks,trigger


    def trigger_point(self,frame):
        #todo:max per time

        if self.release_order>=len(self.release_points):
            return False
        trigger_point = self.release_points[self.release_order]

        if frame >= trigger_point and len(self.current_tasks)<=self.max_per_time:
            self.release_order+=1
            # print(trigger_point)
            return True
        return False

    def analyse_instance(self):
        res = {}
        res["tasks"] = {}
        res["init_num"] = self.init_tasks_num
        for task in self.history_generated_tasks:
            res["tasks"][task.name] = {}
        for task_index,task in enumerate(self.history_generated_tasks):
            res["tasks"][task.name]["s"] = task.s.name
            res["tasks"][task.name]["e"] = task.e.name
            res["tasks"][task.name]["due"] = task.due
            res["tasks"][task.name]["handling"] = task.handling
            res["tasks"][task.name]["travel"] = task.get_minimum_distance(self.map.running_map)
            res["tasks"][task.name]["d"] = task.d
            if task_index<self.init_tasks_num:
                res["tasks"][task.name]["tao"] = 0
            else:
                res["tasks"][task.name]["tao"] = self.release_points[task_index-self.init_tasks_num]
            # res[task.name]["t_d"] = float(task.get_minimum_distance(self.map.running_map) / task.handling)
        return res


    # def set_release_point(self,alpha=2,beta=2,day_time=3000):
    #     if self.use_fixed_random_seeds:
    #         np.random.seed(self.seeds[self.random_seed_run])
    #     else:
    #         t = 1000 * time.time()
    #         np.random.seed(int(t) % 2 ** 32)
    #     beta_x = np.random.beta(alpha, beta, (self.max_task_num,))
    #     self.release_points = sorted(np.array(beta_x * day_time, dtype=np.int32))
    #todo: fixed release points
    def set_release_point(self,alpha=2,beta=2,day_time=3000):
        if self.use_fixed_random_seeds:
            # print("seed",self.seeds[self.random_seed_run])
            np.random.seed(self.seeds[self.random_seed_run])
        else:
            t = 1000 * time.time()
            np.random.seed(int(t) % 2 ** 32)
        fixed_mean = np.mean(self.fixed_release_points)
        fixed_std = 200
        gaussion_releases = np.random.normal(fixed_mean,fixed_std,(self.perturb_choose_num,))
        chose_indexs = np.array(np.random.choice([i for i in range(self.max_task_num)],size=(self.perturb_choose_num,),replace=False))
        self.release_points = np.array(copy.deepcopy(self.fixed_release_points))
        self.release_points[chose_indexs] = gaussion_releases
        self.release_points = sorted(np.array(self.release_points, dtype=np.int32))
        self.release_points = [0 for _ in range(self.init_tasks_num)]+[ item for item in self.release_points if item>0]
        # print(self.fixed_release_points)

    def clear(self):
        # print("task pool------------------")
        super(mTaskPool, self).clear()
        self.init_flag = 1
        self.release_points= []
        self.set_init_num()
        self.set_release_point()
        # print(self.random_seed_run,self.release_points)

