# -*- coding:utf-8 _*-

class STD_gym():
    # The task with the shortest travel distance will be selected first
    def __init__(self,vehicle_num=-1):
        self.name = "STD"
        self.vehicle_num=vehicle_num

    def act(self,current_tasks,vehicle,running_map):
        # root_logger.debug("current # task: " + str(len(current_tasks)))
        decision_list = []
        sorted_tasks = sorted(current_tasks,
                              key=lambda x: x.get_actual_distance(vehicle, running_map))
        for i, task in enumerate(sorted_tasks):
            if task.is_feasible_choice_constraint(vehicle):
                decision_list.append((current_tasks.index(task),vehicle))

        if len(decision_list)>0:
            return decision_list[0]
        else:
            return None