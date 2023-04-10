# -*- coding:utf-8 _*-


from task import Task
import numpy as np
class TaskPool(object):
    def __init__(self,max_per_time,max_task_num,map,seed=0):
        self.max_per_time = max_per_time
        self.max_task_num = max_task_num
        self.map = map
        self.seed = seed
        self.site_num = self.map.site_num
        self.release_order = 0
        self.history_generated_tasks = []
        self.current_tasks = []
        self.assigned_tasks = []
        self.pool_pointer = 0
        self.gen_task_pool()

    def clear(self):
        self.release_order = 0
        self.history_generated_tasks = []
        self.current_tasks = []
        self.assigned_tasks = []
        self.pool_pointer = 0
        self.gen_task_pool()


    def random_generate_task(self,order):
        #random generate one task based on current map site
        G=["A","B","C"]
        site_pair = np.random.choice(self.site_num,2)
        handing = np.random.randint(0,20)
        demand = np.random.randint(1,40)
        due = np.random.randint(40,60)
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
            self.current_tasks+=releasing_tasks
            self.pool_pointer =end_pointer
            for task in self.current_tasks:
                if task.tao==-1:
                    task.tao=self.release_order
            self.release_order+=1
        return  self.current_tasks

    def gen_task_pool(self):
        np.random.seed(12)
        for i in range(self.max_task_num):
            self.history_generated_tasks.append(self.random_generate_task(i))

    def assign_task(self,task_id,vehicle):
        assigned_task = self.current_tasks[task_id]
        self.assigned_tasks.append(assigned_task)
        assigned_task.is_assigned()
        vehicle.get_task(assigned_task)

    def remove_tasks(self,removing_tasks):
        removed_task_list = [self.current_tasks[index].name for index in removing_tasks]
        for index in sorted(removing_tasks,reverse=True):
            self.current_tasks.pop(index)
        return self.current_tasks,removed_task_list
