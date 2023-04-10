# -*- coding:utf-8 _*-

import logging.config

root_logger = logging.getLogger("root")
# todo:more rules
class FCFS_gym():
    # The task is selected in the order of arrival
    def __init__(self,vehicle_num=-1):
        self.name = "FCFS"
        self.vehicle_num=vehicle_num

    def act(self,current_tasks,vehicle,running_map):

        # root_logger.debug("current # task: " + str(len(current_tasks)))
        decision_list = []

        sored_tasks = sorted(current_tasks,key=lambda x:x.tao)


        for i, task in enumerate(sored_tasks):
            if task.is_feasible_choice_constraint(vehicle):
                decision_list.append((current_tasks.index(task),vehicle))

        if len(decision_list)>0:
            return decision_list[0]
        else:
            return None