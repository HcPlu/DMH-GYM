# -*- coding:utf-8 _*-

import json,time
import numpy as np
import gym
from gym_agv.system.taskPool.breakdownTaskPool import mTaskPool, SetReleasePool




class SetReleaseWrapper(gym.Wrapper):
    def __init__(self, env,release_time ,max_task_num = 30 ,max_per_time = 30,init_tasks_num = 5,scenario ="scenario1",task_pool = SetReleasePool,render = 0):
        super().__init__(env)
        self.max_tas_num = max_task_num
        self.max_per_time = max_per_time
        self.init_tasks_num = init_tasks_num
        self.scenario = scenario
        self.render_flag = render
        self.env = self._set_instance(self.scenario,task_pool,release_time )
        self.reward_scale = 1

    def _set_instance(self,scenario,task_pool = SetReleasePool,release_time = None):
        self.env.set_instance(self.max_tas_num,self.max_per_time, self.init_tasks_num, 1, scenario,False,task_pool)
        self.env.sys.task_pool.use_fixed_random_seeds = 1
        self.env.sys.task_pool.set_new_release_time(release_time)
        self.env.sys.render_running = self.render_flag
        return self.env

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        return obs, self.reward_scale * rew,done, info

