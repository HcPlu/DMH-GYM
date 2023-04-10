# -*- coding:utf-8 _*-

import json
import random

import gym,itertools
import gym_agv
import os,datetime
import time
import numpy as np
from policy.STD_gym import STD_gym
from policy.FCFS_gym import FCFS_gym
from policy.EDD_gym import EDD_gym
from policy.NVF_gym import NVF_gym
from copy import deepcopy
from wrappers.scenario_wrapper import SetReleaseWrapper




def testRewardConstraint_multi(env,rule,figure_path,run):
    env.sys.task_pool.random_seed_run = run
    max_per_time = env.sys.max_per_time
    state = env.reset()
    policy = rule()
    episode_r = 0
    test_reward_makespan = 0
    res = {}
    res["name"] = policy.name
    step_reward = []
    info_frame = []
    for step in range(2000):
        t = 1000 * time.time()  # current time in milliseconds
        np.random.seed(int(t) % 2 ** 32)
        state_temp = deepcopy(state)
        v_free_index = []
        current_vehicles = state_temp[1]
        for i,v in enumerate(current_vehicles):
            if v.status==0:
                v_free_index.append(i)
        decision_list = []
        for v_i in v_free_index:
            v = current_vehicles[v_i]

            decision = policy.act(state_temp[0],v,env.sys.map.running_map)
            if decision:
                decision_list.append(v_i *max_per_time+decision[0])

        action = decision_list[0]
        statet2, r, done, info = env.step(action)
        step_reward.append(r)
        episode_r +=r
        test_reward_makespan+=info["reward_makespan"]
        state = statet2
        if done:
            result = env.sys.get_result()
            history_state = env.history_state
            history_task = []
            for state in history_state:
                history_task.append(len(state))
            print(step_reward)
            res["result"] = result
            res["eposode_r"] = 0
            res["history_task"] = history_task
            print(result)
            print(episode_r)
            break
    return res

def single_run_test(rules,timestamp,  scenario_name = "scenario1"):
    runs = 1
    res = {}
    base_path = "./result/result_constraint_analysis/"
    result_path = base_path + "dispatching_rules/%s/" % (timestamp)
    figure_path = result_path + "figures/"


    release_time =  [
        np.random.randint(int(1500 / 30) * i, int(1500 / 30) * (i + 1))
        for i in range(30)]
    print(release_time)
    env =  SetReleaseWrapper(gym.make("agv-breakdown-v0"),release_time, max_task_num = 30 ,max_per_time = 30,init_tasks_num = 5,scenario=scenario_name,render=1)

    env.sys.sr_flag = 0

    result = None
    for rule_name in rules:
        res[rule_name] = {}
        res_path = result_path + "/%s.json" % rule_name
        for run in range(runs):
            res[rule_name][run] = {}
            rule = rules[rule_name]
            st = time.time()
            result = testRewardConstraint_multi(env,rule, figure_path, 0)
            print(result)
            res[rule_name][run]["result"] = result
            res[rule_name][run]["timestamp"] = timestamp
            ed = time.time()
            res[rule_name][run]["cost"] = ed - st
            print(rule_name, "run", run, ed - st)
    return result





if __name__=="__main__":
    scenario_name = "scenario2"

    single_run_test({"edd": EDD_gym}, "test", scenario_name)
    # single_run_test({"nvf": NVF_gym},"test", scenario_name)
    # single_run_test({"fcfs": FCFS_gym}, "test", scenario_name)
    # single_run_test({"random": Random_gym}, "test", scenario_name)

