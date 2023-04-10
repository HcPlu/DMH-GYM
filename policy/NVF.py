# -*- coding:utf-8 _*-


import logging.config

root_logger = logging.getLogger("root")
# todo:more rules
class NVF():
    # The vehicle will select the task with the nearest load point
    def __init__(self,vehicle_num=-1):
        self.name = "NVF"
        self.vehicle_num=vehicle_num
        self.decisions = 0
    def act(self,current_tasks,vehicles,running_map):
        free_vehicles = [(vehicles[i], i) for i in range(len(vehicles)) if vehicles[i].status == 0]

        if len(free_vehicles) == 0 or len(current_tasks) == 0:
            return None
        # root_logger.debug("current # task: " + str(len(current_tasks)))
        decision_list = []
        self.decisions+=1
        for j, vehicle in enumerate(free_vehicles):
            sorted_tasks = sorted(current_tasks,
                                  key=lambda x: x.get_start_distance(vehicle[0].current_site, running_map))
            for i, task in enumerate(sorted_tasks):
                decision_list.append((current_tasks.index(task),free_vehicles[j][0]))

        return decision_list[0]
