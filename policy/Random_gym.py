# -*- coding:utf-8 _*-

import logging.config
import random
root_logger = logging.getLogger("root")
# todo:more rules
class Random_gym():
    # The vehicle will select the task with the nearest load point
    def __init__(self,vehicle_num=-1):
        self.name = "Random"
        self.vehicle_num=vehicle_num
        self.decisions = 0

    def act(self,current_tasks,vehicle,running_map):
        decision_list = []
        self.decisions+=1

        for i, task in enumerate(current_tasks):
            if task.is_feasible_choice_constraint(vehicle):
                decision_list.append((current_tasks.index(task),vehicle))
        random.shuffle(decision_list)
        if len(decision_list)>0:
            return decision_list[0]
        else:
            return None