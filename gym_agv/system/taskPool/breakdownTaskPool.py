# -*- coding:utf-8 _*-


import copy

from ..utils.task import Task
import numpy as np
from gym_agv.system.taskPool.basicTaskPool import BasicTaskPool
import time

class mTaskPool(BasicTaskPool):
    def __init__(self,max_per_time,max_task_num,init_tasks_num,map,seed,load_task=False):

        super(mTaskPool, self).__init__(max_per_time, max_task_num, map, seed)
        self.use_fixed_random_seeds = 1
        self.random_seed_run = 0
        self.load_task = load_task
        np.random.seed(self.seed)
        self.day_time = 1200
        self.perturb_choose_num = 10
        self.fixed_release_points = [np.random.randint(int(self.day_time/self.max_task_num)* i, int(self.day_time/self.max_task_num) * (i + 1)) for i in range(self.max_task_num)]
        self.init_flag = 1
        self.seeds = np.random.randint(0, 5000, size=(5000,))
        self.min_demand = 0
        self.max_demand = 0
        self.a_beta = 1.5
        self.b_beta = 2.5
        self.release_points = []
        self.init_tasks_num = init_tasks_num
        self.set_init_num()
        self.gen_task_pool()
        self.set_release_point()

    def set_init_num(self):
        if self.use_fixed_random_seeds:
            np.random.seed(self.seeds[self.random_seed_run])
        else:
            t = 1000 * time.time()
            np.random.seed(int(t) % 2 ** 32)
        if self.init_tasks_num==0:
            self.init_tasks_num = np.random.randint(1,self.vehicle_num)

    def interval_random(self,a,b):
        #a<b
        return (b-a)*np.random.random()+a

    def gen_task_pool(self):
        #todo: need to optimise running speed
        np.random.seed(self.seed)
        if self.load_task:
            for task in copy.deepcopy(self.map.raw_tasks):
                #todo: load instance
                task.due = self.map.running_map[task.s.name][task.e.name]["distance"] + np.random.randint(200, 300)
                self.history_generated_tasks.append(task)
        else:
            for i in range(self.max_task_num):
                self.history_generated_tasks.append(self.generate_task(i))

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
        demand = 0
        # demand = np.random.randint(5,20)
        tao = -1
        # due = self.map.running_map[start_site.name][end_site.name]["distance"] + np.random.randint(200, 300)
        due = self.map.max_distance*(1+np.random.random())
        # due = self.map.running_map[start_site.name][end_site.name]["distance"] * (1 + self.interval_random(1, 1.5))
        G = ["A","B","C"]
        handling = 0
        task = Task(name, start_site, end_site, int(demand), tao, int(due), G, handling)
        return task

    def release_tasks(self,frame):
        np.random.seed(self.seed)
        trigger = self.trigger_point(frame)

        if len(self.current_tasks) >= self.max_per_time:
            # todo:not trigger here
            # bugs here, we may need reduce trigger point for 1
            return self.current_tasks,0
        else:
            while trigger:
                add_num = 1
                end_pointer = min(self.pool_pointer+add_num,self.max_task_num)
                releasing_tasks = self.history_generated_tasks[self.pool_pointer:end_pointer]
                self.res_num = self.max_task_num - end_pointer
                self.current_tasks += releasing_tasks
                self.pool_pointer = end_pointer
                for task in self.current_tasks:
                    if task.tao == -1:
                        task.tao = frame
                trigger = self.trigger_point(frame)
        return self.current_tasks,trigger


    def trigger_point(self,frame):
        if self.release_order>=len(self.release_points):
            return False
        trigger_point = self.release_points[self.release_order]

        if frame >= trigger_point and len(self.current_tasks)<=self.max_per_time:
            self.release_order+=1
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
        return res


    def set_release_point(self,alpha=2,beta=2,day_time=3000):
        if self.use_fixed_random_seeds:
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
        self.release_points = self.release_points.tolist()
        self.release_points = [0 for _ in range(self.init_tasks_num)]+[ item for item in self.release_points if item>0]
        sorted_tasks = sorted(list(zip(self.release_points,self.history_generated_tasks)),key=lambda x:x[0])
        self.release_points = [item[0] for item in sorted_tasks]
        self.history_generated_tasks = [item[1] for item in sorted_tasks]

    def clear(self):
        super(mTaskPool, self).clear()
        self.init_flag = 1
        self.release_points= []
        self.set_init_num()
        self.set_release_point()



class IntervalTaskPool(mTaskPool):
    def __init__(self,max_per_time,max_task_num,init_tasks_num,map,seed,load_task=False):
        super(IntervalTaskPool, self).__init__(max_per_time, max_task_num, init_tasks_num,map, seed,load_task)
        self.fixed_release_points = []
        x = int(np.random.normal(45,15))
        for i in range(max_task_num):
            self.fixed_release_points.append(x)
            x+=int(np.random.normal(45,15))

    def set_release_point(self,alpha=2,beta=2,day_time=3000):
        if self.use_fixed_random_seeds:
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
        self.release_points = self.release_points.tolist()
        self.release_points = [0 for _ in range(self.init_tasks_num)]+[ item for item in self.release_points if item>0]
        sorted_tasks = sorted(list(zip(self.release_points,self.history_generated_tasks)),key=lambda x:x[0])
        self.release_points = [item[0] for item in sorted_tasks]
        self.history_generated_tasks = [item[1] for item in sorted_tasks]


class PointPool(mTaskPool):
    def __init__(self,max_per_time,max_task_num,init_tasks_num,map,seed,load_task=False):
        super(PointPool, self).__init__(max_per_time, max_task_num, init_tasks_num,map, seed,load_task)

    def set_release_point(self,alpha=2,beta=2,day_time=3000):
        if self.use_fixed_random_seeds:
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
        self.release_points = self.release_points.tolist()
        self.release_points = [0 for _ in range(self.init_tasks_num)]+[ item for item in self.release_points if item>0]
        sorted_tasks = sorted(list(zip(self.release_points,self.history_generated_tasks)),key=lambda x:x[0])
        self.release_points = [item[0] for item in sorted_tasks]
        self.history_generated_tasks = [item[1] for item in sorted_tasks]


class SetReleasePool(mTaskPool):
    def __init__(self,max_per_time,max_task_num,init_tasks_num,map,seed,load_task=False):

        super(SetReleasePool, self).__init__(max_per_time, max_task_num, init_tasks_num,map, seed,load_task)
        self.episodic_tasks = []
        self.raw_history_generated_tasks = copy.deepcopy(self.history_generated_tasks)

    def set_new_release_time(self,release_time):
        self.release_points = [0 for _ in range(self.init_tasks_num)]+[ item for item in release_time]
        if len(self.episodic_tasks)>0:
            self.history_generated_tasks = copy.deepcopy(self.raw_history_generated_tasks)
        sorted_tasks = sorted(list(zip(self.release_points,self.history_generated_tasks)),key=lambda x:x[0])
        self.release_points = [item[0] for item in sorted_tasks]
        self.history_generated_tasks = [item[1] for item in sorted_tasks]
        self.episodic_tasks = copy.deepcopy(self.history_generated_tasks)

    def clear(self):
        # print("no clear",self.release_points)
        self.res_num = self.max_task_num
        self.release_order = 0
        self.released_num = 0
        self.current_tasks = []
        self.assigned_tasks = []
        self.pool_pointer = 0
        self.history_interval = []
        self.init_flag = 1
        self.set_init_num()
        self.history_generated_tasks = copy.deepcopy(self.episodic_tasks)
        # print("release points",self.release_points)
        # print([str(item) for item in self.history_generated_tasks])
