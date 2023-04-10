# -*- coding:utf-8 _*-

import numpy as np
from gym_agv.system.utils.task import Task


class vehiclePool():
    def __init__(self,vehicle_num):
        self.vehicle_num = vehicle_num
        self.recover = 300
        self.repairing_time = np.zeros(self.vehicle_num)
        self.breakdown_times = np.zeros(self.vehicle_num)
        self.breakdown_limits = [1,1,1,1]
        self.breakdown_interval = [400,700,1000,1400]
        self.breakdown_increase = [0,0,0,0]

    def breakdown(self,realtime,frame,rate,vehicles,task_pool):
        for i,v in enumerate(vehicles):
            if v.status !=2:
                if self.trigger(realtime,i,v,rate):
                    # release all task
                    if len(v.waiting_list)>0:
                        free_task = v.release_task()
                        original_task = Task(free_task.name,free_task.s,free_task.e,free_task.d,free_task.tao,free_task.due,free_task.G,free_task.handling)
                        task_pool.get_task(original_task)
                    v.is_breakdown()
            else:
                if self.repairing_time[i]>=self.recover:
                    self.repairing_time[i] = 0
                    v.is_free()
                else:
                    self.repairing_time[i] +=1/rate
                    v.is_breakdown()

        return vehicles

    def trigger(self,realtime,i,vehicle,rate):
        self.breakdown_increase[i]+=1/rate
        if self.breakdown_increase[i]>=self.breakdown_interval[i]:
            self.breakdown_increase[i]=0
            if self.breakdown_times[i]<self.breakdown_limits[i]:
                if vehicle.status!=2:
                    self.breakdown_times[i]+=1
                    return True
        return False

    def clear(self):
        self.repairing_time = np.zeros(self.vehicle_num)
        self.breakdown_times = np.zeros(self.vehicle_num)
        self.breakdown_limits = [1,1,1,1]
        self.breakdown_interval = [400,700,1000,1400]
        self.breakdown_increase = [0,0,0,0]
