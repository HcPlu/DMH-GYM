# -*- coding:utf-8 _*-

import logging.config

root_logger = logging.getLogger("root")
# todo:more rules
class EDD():
    # The task with the earliest due date will be selected first
    def __init__(self,vehicle_num=-1):
        self.name = "EDD"
        self.vehicle_num=vehicle_num

    def act(self,current_tasks,vehicles,running_map):
        free_vehicles = [(vehicles[i], i) for i in range(len(vehicles)) if vehicles[i].status == 0]
        if len(free_vehicles) == 0 or len(current_tasks) == 0:
            return None
        # root_logger.debug("current # task: " + str(len(current_tasks)))
        decision_list = []
        sorted_tasks = sorted(current_tasks,key=lambda x:x.due+x.tao)

        for j, vehicle in enumerate(free_vehicles):
            for i, task in enumerate(sorted_tasks):
                decision_list.append((current_tasks.index(task),free_vehicles[j][0]))

        return decision_list[0]