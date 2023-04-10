# -*- coding:utf-8 _*-


import matplotlib.pyplot as plt
import numpy as np
# todo: static render , gantte
class StaticRender():
    def __init__(self,vehicle_num):
        self.vehicle_num = vehicle_num

    def save_figure(self,vehicles,title,path):
        result = {}
        bottom_actual_task = []
        bottom_minimum_task = []
        actual_task_cost = []
        minimum_task_cost = []
        task_vehicle_list = []
        max_task_len = 0
        all_task_list = []
        x_num = self.vehicle_num * 2
        for v in vehicles:
            max_task_len = max(max_task_len, len(v.history_list))

        for v in vehicles:
            bottom_actual = []
            bottom_minimum = []
            actual_task = []
            minimum_task = []
            tv = []
            all_task_list += [task for task in v.history_list]
            for task in v.history_list:
                bottom_actual.append((task.start_time))
                bottom_minimum.append((task.start_time))
                actual_task.append(task.actual_time)
                minimum_task.append(task.minimum_time)
                tv.append(task.name)

            while (len(bottom_minimum) < max_task_len):
                bottom_actual.append(0)
                bottom_minimum.append(0)
                actual_task.append(0)
                minimum_task.append(0)

            bottom_actual_task.append(bottom_actual)
            bottom_minimum_task.append(bottom_minimum)
            actual_task_cost.append(actual_task)
            minimum_task_cost.append(minimum_task)
            task_vehicle_list.append(tv)
        all_task_list = sorted(all_task_list, key=lambda x: x.finished_time)
        mix_bottom = []
        mix_cost = []
        for i in range(self.vehicle_num):
            mix_bottom.append(bottom_actual_task[i])
            mix_bottom.append(bottom_minimum_task[i])
            mix_cost.append(actual_task_cost[i])
            mix_cost.append(minimum_task_cost[i])
        mix_bottom = np.array(mix_bottom)
        mix_cost = np.array(mix_cost)
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(self.vehicle_num):
            for j, task in enumerate(task_vehicle_list[i]):
                plt.text(bottom_minimum_task[i][j], i * 2, "%s" % task.split("_")[1])
                plt.text(bottom_minimum_task[i][j], i * 2 + 1, "%s" % task.split("_")[1])
        for i in range(max_task_len):
            start = mix_bottom[:, i]
            value = mix_cost[:, i]
            plt.barh(range(x_num), value, left=start, align='center', alpha=0.8)
        y_ticks = []
        for i in range(self.vehicle_num):
            y_ticks.append("actual_v%d" % i)
            y_ticks.append("ideal_v%d" % i)
        plt.subplots_adjust(bottom=0.15, left=0.11, wspace=0.8, right=0.99, top=0.95)
        plt.yticks(np.arange(0, x_num), y_ticks)
        plt.savefig(path)

    def render(self,vehicles,title):
        result = {}
        bottom_actual_task = []
        bottom_minimum_task = []
        actual_task_cost = []
        minimum_task_cost = []
        task_vehicle_list = []
        max_task_len = 0
        all_task_list = []
        x_num = self.vehicle_num * 2
        for v in vehicles:
            max_task_len = max(max_task_len, len(v.history_list))

        for v in vehicles:
            bottom_actual = []
            bottom_minimum = []
            actual_task = []
            minimum_task = []
            tv = []
            all_task_list += [task for task in v.history_list]
            for task in v.history_list:
                bottom_actual.append((task.start_time))
                bottom_minimum.append((task.start_time))
                actual_task.append(task.actual_time)
                minimum_task.append(task.minimum_time)
                tv.append(task.name)

            while (len(bottom_minimum) < max_task_len):
                bottom_actual.append(0)
                bottom_minimum.append(0)
                actual_task.append(0)
                minimum_task.append(0)

            bottom_actual_task.append(bottom_actual)
            bottom_minimum_task.append(bottom_minimum)
            actual_task_cost.append(actual_task)
            minimum_task_cost.append(minimum_task)
            task_vehicle_list.append(tv)
        all_task_list = sorted(all_task_list, key=lambda x: x.finished_time)
        mix_bottom = []
        mix_cost = []
        for i in range(self.vehicle_num):
            mix_bottom.append(bottom_actual_task[i])
            mix_bottom.append(bottom_minimum_task[i])
            mix_cost.append(actual_task_cost[i])
            mix_cost.append(minimum_task_cost[i])
        mix_bottom = np.array(mix_bottom)
        mix_cost = np.array(mix_cost)
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(self.vehicle_num):
            for j, task in enumerate(task_vehicle_list[i]):
                plt.text(bottom_minimum_task[i][j], i * 2, "%s" % task.split("_")[1])
                plt.text(bottom_minimum_task[i][j], i * 2 + 1, "%s" % task.split("_")[1])
        for i in range(max_task_len):
            start = mix_bottom[:, i]
            value = mix_cost[:, i]
            plt.barh(range(x_num), value, left=start, align='center', alpha=0.8)
        y_ticks = []
        for i in range(self.vehicle_num):
            y_ticks.append("actual_v%d" % i)
            y_ticks.append("ideal_v%d" % i)
        plt.subplots_adjust(bottom=0.15, left=0.11, wspace=0.8, right=0.99, top=0.95)
        plt.yticks(np.arange(0, x_num), y_ticks)
        plt.title(title)
        plt.show()