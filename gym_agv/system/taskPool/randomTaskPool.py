# -*- coding:utf-8 _*-

from task import Task
import numpy as np
from system.taskPool.basicTaskPool import BasicTaskPool

class RandomTaskPool(BasicTaskPool):
    def __init__(self,max_per_time,max_task_num,map,seed):
        super(RandomTaskPool, self).__init__(max_per_time,max_task_num,map,seed)


    def generate_task(self,order):
        #random generate one task based on current map site
        G=["A","B","C"]
        site_pair = np.random.choice(self.site_num,2)
        handing = np.random.randint(0,20)
        demand = np.random.randint(1,40)
        due = np.random.randint(50,150)
        task = Task("task_"+str(len(self.history_generated_tasks)),self.map.sites[site_pair[0]],self.map.sites[site_pair[1]],demand,-1,due,G,handing)
        return task


    def release_tasks(self,frame):
        np.random.seed(self.seed)

        # todo:more way to release tasks
        if len(self.current_tasks)>=self.max_per_time:
            return self.current_tasks
        else:
            diff_num = self.max_per_time-len(self.current_tasks)
            add_num = np.random.choice(diff_num+1)
            # print('pool_pointer',self.pool_pointer,diff_num,add_num,len(self.current_tasks))
            # print("end",min(self.pool_pointer+add_num,self.max_task_num))
            end_pointer = min(self.pool_pointer+add_num,self.max_task_num)
            releasing_tasks = self.history_generated_tasks[self.pool_pointer:end_pointer]
            self.released_num+=len(releasing_tasks)
            self.res_num =self.max_task_num-end_pointer
            self.current_tasks+=releasing_tasks
            self.pool_pointer =end_pointer

            for task in self.current_tasks:
                if task.tao==-1:
                    task.tao=self.release_order
            self.release_order+=1
            return  self.current_tasks

