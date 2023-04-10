# -*- coding:utf-8 _*-

import logging.config

root_logger = logging.getLogger("root")
# todo:more rules
class RandomPolicy():
    def __init__(self,vehicle_num=-1):
        self.vehicle_num=vehicle_num

    def act(self,current_tasks,vehicles,running_map):
        free_vehicles = [(vehicles[i], i) for i in range(len(vehicles)) if vehicles[i].status == 0]
        if len(free_vehicles) == 0 or len(current_tasks) == 0:
            return None
        # root_logger.debug("current # task: " + str(len(current_tasks)))

        decision_list = []
        for i, task in enumerate(current_tasks):
            for j, vehicle in enumerate(free_vehicles):
                decision_list.append((i,free_vehicles[j][0]))

        return decision_list[0]