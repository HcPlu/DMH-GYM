# -*- coding:utf-8 _*-


import logging.config

root_logger = logging.getLogger("root")
# todo:more rules
class NVF_gym():
    # The vehicle will select the task with the nearest load point
    def __init__(self,vehicle_num=-1):
        self.name = "NVF"
        self.vehicle_num=vehicle_num
        self.decisions = 0
    def act(self,current_tasks,vehicle,running_map):


        # root_logger.debug("current # task: " + str(len(current_tasks)))
        decision_list = []
        self.decisions+=1

        sorted_tasks = sorted(current_tasks,
                              key=lambda x: x.get_start_distance(vehicle.current_site, running_map))
        for i, task in enumerate(sorted_tasks):
            if task.is_feasible_choice_constraint(vehicle):
                decision_list.append((current_tasks.index(task),vehicle))

        if len(decision_list)>0:
            return decision_list[0]
        else:
            return None
