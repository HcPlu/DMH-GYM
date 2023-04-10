# -*- coding:utf-8 _*-

import time

import gym
import numpy as np
from gym_agv.system.render.dynamicRender import DynamciRender
from gym_agv.system.render.staticRender import StaticRender
from gym_agv.system.taskPool.breakdownTaskPool import mTaskPool
from gym_agv.system.vehiclePool.vehiclePool import vehiclePool
from gym_agv.system.utils.map import Map
from policy.STD_gym import STD_gym
from policy.FCFS_gym import FCFS_gym
from policy.LWT_gym import LWT_gym
from policy.EDD_gym import EDD_gym
from policy.NVF_gym import NVF_gym

import matplotlib.pyplot as plt


class System_gym():
    def __init__(self, path, policy, task_pool, max_task_num=80, max_per_time=20, init_tasks_num=3, render_info=0,
                 seed=12, load_task=False, abc=None, render_running=0):
            self.map=Map()
            self.map.read_map(path,0)
            self.frame = 0
            self.realtime = 0
            self.tfrate=1
            self.finished_task = 0
            self.current_tasks = None
            self.site_num = self.map.site_num
            self.vehicle_num = self.map.vehicle_num
            self.sites = self.map.sites
            self.load_task = load_task
            self.vehicle_types = self.map.vehicle_types
            self.vehicles = self.map.vehicles
            for v in self.vehicles:
                v.tfrate = self.tfrate
            self.max_task_num = max_task_num
            self.max_per_time = max_per_time
            self.init_tasks_num = init_tasks_num
            self.render_info = render_info
            self.render_running = render_running
            self.seed = seed
            self.sr_flag = 0
            self.figure_path = ''
            self.task_pool = task_pool(self.max_per_time,self.max_task_num,self.init_tasks_num,self.map,load_task = self.load_task,seed=self.seed)
            self.vehicle_pool = vehiclePool(self.vehicle_num)
            self.history_released_list = []
            if policy:
                self.policy = policy(self.vehicle_num)
            if policy:
                if self.policy.name=="MIX":
                    if abc is not None:
                        self.policy.set_abc(abc)
            self.check_back=0
            self.history_staging_list = []
            self.init_render = 0



    def reset(self):
        self.frame=0
        self.init_render = 0
        self.realtime = 0
        self.finished_task = 0
        self.task_pool.clear()
        self.vehicle_pool.clear()
        self.check_back = 0
        self.current_tasks = None
        self.history_released_list = []
        self.history_staging_list = []
        self.current_tasks,_ = self.task_pool.release_tasks(self.realtime)
        self.history_released_list.append([(task.name,task.tao) for task in self.current_tasks])
        self.history_staging_list.append([(task.name, task.tao) for task in self.current_tasks])
        self.map.init()
        return self.current_tasks,self.vehicles,self.is_done(),None

    def is_done(self):
        # print(self.task_pool.release_points[self.task_pool.release_order])
        if self.finished_task<self.max_task_num:
            return False

        if self.check_back:
            for v in self.vehicles:
                if v.current_site!=v.start_site:
                    return False
        return True

    def test_map(self):
        fig,ax = self.map.draw_map()
        return fig,ax

    def available_task(self,tasks,v):
        for task in tasks:
            if task.is_feasible_choice_constraint(v):
                return  True
        return False

    def single_step(self, decision_list):
        np.random.seed(self.seed)
        success_assign_list = []
        fail_assign_list = []
        removed_task_list = []
        if decision_list:
            success_assign_list = []
            fail_assign_list = []

            result = self.single_assign(decision_list[0], decision_list[1])
            if result:
                success_assign_list.append(decision_list)
            else:
                fail_assign_list.append(decision_list)
            self.current_tasks, removed_task_list = self.task_pool.remove_tasks([assignment[0] for assignment in success_assign_list])

        if self.check_back:
            if len(self.current_tasks)==0:
                for v in self.vehicles:
                    if len(v.waiting_list)==0:
                        back = self.task_pool.go_back(v)

        finish_flag = 0 #one task finished at least
        arrival_flag = 0 # time to release

        # if self.task_pool.res_num > 0:
        #     self.current_tasks, arrival_flag = self.task_pool.release_tasks(self.realtime)

        while not (finish_flag or arrival_flag):
            if len([v for v in self.vehicles if v.status==0])>0 and len(self.current_tasks)>0:
                feasible_flag = 0
                for v in self.vehicles:
                    if self.available_task( self.current_tasks, v):
                        feasible_flag = 1
                if feasible_flag==1:
                    break

            self.frame += 1
            # print(self.frame/self.tfrate)
            self.realtime=float(self.frame/self.tfrate)
            # update assigned task
            self.vehicle_pool.breakdown(self.realtime,self.frame,self.tfrate,self.vehicles,self.task_pool)
            for v in self.vehicles:
                cost = v.update_time(self.realtime,self.tfrate)

                if cost == -1:
                    self.finished_task += 1
                    # 还有任务需要完成
                    if len(self.current_tasks)>0 or (self.task_pool.res_num==0):
                        finish_flag=1

                if cost == -2:
                    if self.task_pool.res_num==0:
                        free_vehicles_num = len([(self.vehicles[i], i) for i in range(self.vehicle_num) if self.vehicles[i].status == 0])
                        if free_vehicles_num==self.vehicle_num:
                            # all tasks have been finished and all vehicles have been back to carport
                            finish_flag = 1
            # print("current_coord", [v.current_coord for v in self.vehicles])
            data_coords = np.array([v.current_coord for v in self.vehicles])
            if self.render_running:
                self.test_render(data_coords)
                time.sleep(0.003)
            # update the waiting time
            for task in self.current_tasks:
                task.update_wait(float(1/self.tfrate))
            if self.task_pool.res_num>0:
                self.current_tasks,arrival_flag = self.task_pool.release_tasks(self.realtime)

            if len([v for v in self.vehicles if v.status==0])==self.vehicle_num:
                break

        done = self.is_done()
        self.history_released_list.append([(task.name,task.tao) for task in self.current_tasks])
        return self.current_tasks,self.vehicles,done,[success_assign_list, removed_task_list, fail_assign_list]

    def test_render(self,coords):
        if not self.init_render:
            #todo: some bugs maybe?
            self.init_render =  1
            self.fig, self.ax = self.map.draw_map()
            self.lns_v = []
            self.lns_route = []
            color_list= ["r","y","g","k","b"]
            for i,coord in enumerate(coords):
                (ln,) = self.ax.plot(coord[0], coord[1], "o", markersize=20, c=color_list[i], animated=True)
                self.lns_v.append(ln)
            for i,v in enumerate(self.vehicles):
                route = np.array(v.expected_coords)
                (ln,) = self.ax.plot(route[:,0], route[:,1], linewidth=4 , c=color_list[i],alpha=0.5, animated=True)
                self.lns_route.append(ln)


            plt.show(block=False)
            plt.pause(0.1)
            self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

            self.fig.canvas.blit(self.fig.bbox)
            self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            # draw the animated artist, this uses a cached renderer
            for ln in self.lns_v:
                self.ax.draw_artist(ln)

            self.fig.canvas.blit(self.fig.bbox)


        self.fig.canvas.restore_region(self.bg)
        # update the artist, neither the canvas state nor the screen have changed
        for i,ln in enumerate(self.lns_v):
            ln.set_xdata(coords[i][0])
            ln.set_ydata(coords[i][1])

        for i,ln in enumerate(self.lns_route):
            route = np.array(self.vehicles[i].expected_coords)
            ln.set_xdata(route[:,0])
            ln.set_ydata(route[:,1])

        # re-render the artist, updating the canvas state, but not the screen
        for ln in self.lns_v:
            self.ax.draw_artist(ln)

        for ln in self.lns_route:
            self.ax.draw_artist(ln)
        # self.ax.draw_artist(self.ln)
        # self.ax.draw_artist(self.ln2)
        # copy the image to the GUI state, but screen might not be changed yet
        self.fig.canvas.blit(self.fig.bbox)
        # flush any pending GUI events, re-painting the screen if needed
        self.fig.canvas.flush_events()
        # you can put a pause in if you want to slow things down

    def single_assign(self, task_index, vehicle):
        # assign task from current task list to the vehicle
        if self.current_tasks[task_index].is_feasible_choice_constraint(vehicle):
            self.task_pool.assign_task(task_index, vehicle,self.map.running_map)
            return 1
        else:
            return 0

    def step(self, decision_list):
        self.current_tasks,self.vehicles,done,[success_assign_list, removed_task_list, fail_assign_list] = self.single_step(decision_list)
        free_vehicles = [self.vehicles[i] for i in range(self.vehicle_num) if self.vehicles[i].status == 0]
        free_vehicles_num = len(free_vehicles)
        available_task_num = len(self.current_tasks)

        while (free_vehicles_num==0 or available_task_num==0 ) and not done:
            # print(len(self.history_staging_list),done)
            decision_list = None
            self.current_tasks,self.vehicles,done,[success_assign_list, removed_task_list, fail_assign_list] = self.single_step(decision_list)
            free_vehicles = [self.vehicles[i] for i in range(self.vehicle_num) if self.vehicles[i].status == 0]
            free_vehicles_num = len(free_vehicles)
            available_task_num = len(self.current_tasks)

        self.history_staging_list.append([(task.name,task.tao) for task in self.current_tasks])
        return self.current_tasks,self.vehicles,done,[self.realtime,success_assign_list, removed_task_list, fail_assign_list]

    def run(self,runs):
        run_results = {}
        run_results["alg"]=self.policy.name
        run_results["parameters"]={}
        run_results["parameters"]["max_task"]=self.max_task_num
        run_results["parameters"]["max_task_per_time"] = self.max_per_time
        for i in range(runs):
            current_tasks,current_vehicle,done,_ = self.reset()
            decision_time = 0
            while not done:
                decision_list = self.policy.act(current_tasks,current_vehicle,self.map.running_map)
                current_tasks,current_vehicle,done,_=self.step(decision_list)
                decision_time += 1

            title = self.policy.name+"_run_"+str(i)
            result = self.get_result(title, self.render_info)
            # print(decision_time)
            run_results[i] = result
        return run_results

    def is_decision_time(self):
        for v in self.vehicles:
            if v.status==0:
                return 1
        return 0

    def info(self):
        print("--------------------------------------------------")
        print("time: ",self.realtime)
        for i in range(self.vehicle_num):
            vehicle_name = "AGV_%d"%i
            history_task_list = [task.name for task in self.vehicles[i].history_list]

            current_task = self.vehicles[i].waiting_list[0].name if self.vehicles[i].status==1 else "free now"

            print(vehicle_name,"d:",self.vehicles[i].c,", current task: "+current_task,"history: ",history_task_list)
            if current_task!="free now":
                current_t = self.vehicles[i].waiting_list[0]
                print("        current task :",current_task,"           capacity  :",current_t.d,"   progress (ct/at/et)  : (%d / %d/ %d)"%(current_t.count_time,current_t.actual_time,current_t.minimum_time))
        print("--------------------------------------------------")

    def get_result(self, title="test", fig=1):
        # for line in self.history_staging_list:
        #     print(line)
        result = {}
        max_task_len=0
        all_task_list = []
        for v in self.vehicles:
            max_task_len=max(max_task_len,len(v.history_list))
        for v in self.vehicles:
            #cons
            all_task_list+=[task for task in v.history_list]
            # print([[task.start_time,task.name,task.actual_time,task.finished_time] for task in v.history_list])

        all_task_list = sorted(all_task_list,key= lambda x:x.finished_time)
        makespan = all_task_list[-1].finished_time

        # todo: redesign the name of task
        all_task_list_real = [task for task in all_task_list if task.name is not "task_back"]
        all_task_list_real = sorted(all_task_list_real, key=lambda x: int(x.name.split("_")[1]))
        ma_ratio=0

        delay_tasks=[(i,task) for i,task in enumerate(all_task_list_real) if task.is_back==0 and task.delay>0]
        delay_tasks_index = [item[0] for item in delay_tasks]
        delay_tasks_tardiness = [item[1].delay for item in delay_tasks]

        for task in all_task_list_real:
            ma_ratio +=(task.actual_time-task.minimum_time)/task.actual_time
        ma_ratio/=len(all_task_list_real)
        delay_ratio = round(sum([task.delay>0 for task in all_task_list_real if task.is_back==0])/self.max_task_num,4)

        avg_delay = np.mean([task.delay for task in all_task_list_real])
        result["makespan"] = makespan
        result["avg_delay"] = avg_delay
        # result["delay_task_tardiness"] = round(np.mean(delay_tasks_tardiness),4) #average delay of delayed task
        result["optimization_ratio"] = round(ma_ratio,4)
        result["delay_ratio"] = delay_ratio
        result["m_wait"]=[task.waiting_time for task in all_task_list if task.is_back==0]
        result["tasks_info"] = [ [task.tao,task.waiting_time,task.start_time,task.finished_time,task.delay,task.due] for task in all_task_list_real]


        max_wait = round(np.max([task.waiting_time for task in all_task_list if task.is_back==0]),4)
        mean_wait = round(np.mean([task.waiting_time for task in all_task_list if task.is_back == 0]),4)
        real_time_list = [task.actual_time for task in all_task_list_real]

        if fig:
            columns = ["makespan","tardiness", "delay_ratio", "max_wait", "mean_wait"]
            rows = [title]
            cells = [[makespan, result["avg_delay"],delay_ratio, max_wait, mean_wait]]
            real_tasks = [task for task in all_task_list if task.is_back==0]
            real_tasks = sorted(real_tasks,key=lambda x:x.tao)
            due_times = [task.due for task in real_tasks]
            actual_time = [task.actual_time for task in real_tasks]
            wait_time = [task.waiting_time for task in real_tasks]

            fig_flag = 0
            if fig_flag:
                fig2 = plt.figure(tight_layout=False)
                bar_width=0.3
                plt.bar(np.arange(len(real_tasks)), wait_time, bar_width, label="wait")
                plt.bar(np.arange(len(real_tasks)) + bar_width, actual_time, bar_width, align="center", label="act")
                plt.bar(np.arange(len(real_tasks))+bar_width*2, due_times,bar_width, align="center",label="due")
                plt.bar(delay_tasks_index,(-1)*np.array(delay_tasks_tardiness),bar_width)
                plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9)
                table = plt.table(cellText=cells,
                          rowLabels=rows,
                          colLabels=columns,
                                  cellLoc="left",
                          bbox=[0,-0.4,1,0.2])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                plt.title(title)
                plt.legend()
                plt.show()

            sr = StaticRender(self.vehicle_num)
            dr = DynamciRender(self.map, self.vehicles)
            if self.figure_path:
                sr.save_figure(self.vehicles, title, self.figure_path)
            dr_flag = 0
            if self.sr_flag:
                sr.render(self.vehicles,title)
            if dr_flag:
                dr.render()
        return result



class AgvEnv_breakdown0(gym.Env):
#no relaxed delay
    def __init__(self,gamma = 0.99,set_instance = False):
        file_path = "./gym_agv/system/dataset/scenario1/scenario1.json"

        self.max_task_num = 60
        self.max_per_time = 60
        self.init_tasks_num = 5
        self.scenario_name = "scenario1"
        abc = None
        self.gamma = gamma
        if not set_instance:
            self.sys = System_gym(file_path, None, mTaskPool, max_task_num=self.max_task_num,
                                  max_per_time=self.max_per_time, init_tasks_num=self.init_tasks_num, abc=abc)
            self.map = self.sys.map
            self.vehicle_num = self.sys.vehicle_num
            action_num = self.vehicle_num*self.max_per_time
            self.action_space = gym.spaces.Discrete(action_num)
            self.observation_space = gym.spaces.Box(np.finfo(np.float32).min,np.finfo(np.float32).max,shape=(51,),dtype =np.float32)
            self.current_tasks = None
            self.current_vehicles = None
            self.history_state = []

    def set_instance(self, max_task_num, max_per_time, init_tasks_num, f_gamma, scenario_name ="scenario1",load_task=False,task_pool = mTaskPool):
        file_path = "./gym_agv/system/dataset/%s/%s.json"%(scenario_name, scenario_name)
        self.scenario_name = scenario_name
        self.max_task_num = max_task_num
        self.max_per_time = max_per_time
        self.init_tasks_num = init_tasks_num
        self.gamma = f_gamma
        abc = None
        self.sys = System_gym(file_path, None, task_pool, max_task_num=self.max_task_num,
                              max_per_time=self.max_per_time, init_tasks_num=self.init_tasks_num, load_task=load_task,
                              abc=abc)
        # self.sys.task_pool.set_fixed_release_points()
        self.map=self.sys.map
        self.vehicle_num = self.sys.vehicle_num
        action_num = self.vehicle_num * self.max_per_time
        self.action_space = gym.spaces.Discrete(action_num)
        self.observation_space = gym.spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(51,),
                                                dtype=np.float32)
        self.current_tasks = None
        self.current_vehicles = None
        self.history_state = []

    def state_phi(self,s):
        tasks = s[0]
        vehicles = s[1]
        s = []
        for task in tasks:
            task_s = []
            task_s.append(task.get_minimum_distance(self.map.running_map))
            task_s.append(task.waiting_time)
            task_s.append(task.due)
            task_s.append(task.d)
            s +=task_s
        while len(s)<24:
            s.append(0)

        for v in vehicles:
            v_s = []
            v_s.append(v.vl)
            v_s.append(v.status)
            v_s.append(v.c)
            t_d = []
            for task in tasks:
                t_d.append(task.get_actual_distance(v,self.map.running_map))
            while len(t_d)<6:
                t_d.append(0)
            v_s+=t_d
            s+=v_s

        return s

    def valid_actions(self, state):
        current_tasks = state[0]
        current_vehicles = state[1]
        v_free_index = []
        for i,v in enumerate(current_vehicles):
            if v.status==0:
                v_free_index.append(i)
        valid_actions = []
        for i,task in enumerate(current_tasks):
            for v_i in v_free_index:
                free_v = current_vehicles[v_i]
                if task.is_feasible_choice_constraint(free_v):
                    valid_actions.append(v_i*self.max_per_time+i)
        return valid_actions

    def reward_phi(self,state):
        current_vechiles = state[1]
        tardiness = 0
        tasks_num = 0
        for v in current_vechiles:
            for task in v.history_list:
                #no relaxed
                tardiness +=task.delay
                tasks_num+=1
        if tasks_num==0:
            return 0
        avg_tardiness = -tardiness/tasks_num
        return avg_tardiness

    def step(self,action):
        task_index = int(action%self.max_per_time)
        vehicle_index = int(action/self.max_per_time)
        info = {}
        vehicle = self.current_vehicles[vehicle_index]
        decision_list =[task_index,vehicle]
        if decision_list==None:
            info["feasible"] = False
            return None, None, None, info

        if self.is_feasible_ours(decision_list[0],decision_list[1]):
            info["decision"] = (self.current_tasks[decision_list[0]].name,decision_list[1].name,self.sys.realtime)
            info["feasible"] = True
            r_phi_t = self.reward_phi([self.current_tasks, self.current_vehicles])
            r_m_t = self.reward_func([self.current_tasks, self.current_vehicles])
            self.current_tasks, self.current_vehicles, done, sys_info = self.sys.step(decision_list)
            state = [self.current_tasks, self.current_vehicles]
            r = self.reward_func(state)
            free_vehicles = [v for v in self.current_vehicles if v.status == 0]
            available_decision = self.available_vehicle(self.current_tasks,free_vehicles)
            # There are no avalable decision and the system is still running
            while not available_decision and not done:
                self.current_tasks, self.current_vehicles, done, sys_info = self.sys.step(None)
                free_vehicles = [v for v in self.current_vehicles if v.status == 0]
                state = [self.current_tasks, self.current_vehicles]
                r = self.reward_func(state)
                available_decision = self.available_vehicle(self.current_tasks, free_vehicles)

            r_phi_t2 = self.reward_phi([self.current_tasks, self.current_vehicles])
            r_m_t2 = self.reward_func([self.current_tasks, self.current_vehicles])
            r_f =self.gamma*r_phi_t2-r_phi_t
            reward = r_m_t2-r_m_t
            info["reward_F"] = r_f
            info["reward_makespan"] = reward
            if len(state[0])>0:
                self.history_state.append([task.name for task in state[0]])
            return state,r_f, done,info
        else:
            info["feasible"] = False
            return None,None,None,info

    def available_vehicle(self,tasks,vehicles):
        for task in tasks:
            for v in vehicles:
                if task.is_feasible_choice_constraint(v):
                    return True
        return False

    def is_feasible(self,task_index,vehicle):
        return self.current_tasks[task_index].is_feasible_choice_simple(vehicle)

    def is_feasible_ours(self,task_index,vehicle):
        return self.current_tasks[task_index].is_feasible_choice_constraint(vehicle)

    def reward_func(self,state):
        current_vechiles = state[1]
        tasks_num = 0
        tasks = []
        for v in current_vechiles:
            for task in v.history_list:
                tasks_num+=1
                tasks.append(task)
        if tasks_num == 0:
            return 0
        makespan = sorted(tasks,key=lambda x:x.finished_time)[-1].finished_time
        reward = -makespan
        return reward

    def reset(self):
        self.current_tasks = None
        self.current_vehicles = None
        self.history_state = []
        self.current_tasks, self.current_vehicles, done, _ = self.sys.reset()
        state = [self.current_tasks, self.current_vehicles]
        self.history_state.append([task.name for task in state[0]])
        return state

    def render(self, mode='human'):
        self.sys.info()


    def seed(self, seed):
        np.random.seed(seed)

    def close(self):
        pass

class AgvEnv_breakdown1(gym.Env):
#policy-vehicle, no relaxed delay
    def __init__(self,gamma = 0.99,set_instance = False):
        file_path = "./gym_agv/system/dataset/scenario1/scenario1.json"
        std = STD_gym()
        fcfs = FCFS_gym()
        edd = EDD_gym()
        nvf = NVF_gym()
        self.policys = [fcfs,edd,nvf,std]
        self.policy_num = len(self.policys)
        self.gamma = gamma
        self.max_task_num = 30
        self.max_per_time = 30
        self.init_tasks_num = 5
        self.scenario_name = "scenario2"
        abc = None
        self.gamma = gamma
        if not set_instance:
            self.sys = System_gym(file_path, None, mTaskPool, max_task_num=self.max_task_num,
                                  max_per_time=self.max_per_time, init_tasks_num=self.init_tasks_num, abc=abc)
            self.map = self.sys.map
            self.vehicle_num = self.sys.vehicle_num
            action_num = self.vehicle_num * self.policy_num
            self.action_space = gym.spaces.Discrete(action_num)
            self.observation_space = gym.spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(
            4 * self.max_task_num + self.vehicle_num * (self.max_task_num + 3),),
                                                    dtype=np.float32)
            self.current_tasks = None
            self.current_vehicles = None
            self.history_state = []


    def set_instance(self, max_task_num, max_per_time, init_tasks_num, f_gamma, scenario_name="scenario1", load_task=False,
                     task_pool=mTaskPool):
        file_path = "./gym_agv/system/dataset/%s/%s.json" % (scenario_name, scenario_name)
        self.scenario_name = scenario_name
        self.max_task_num = max_task_num
        self.max_per_time = max_per_time
        self.init_tasks_num = init_tasks_num
        self.gamma = f_gamma
        abc = None
        self.sys = System_gym(file_path, None, task_pool, max_task_num=self.max_task_num,
                              max_per_time=self.max_per_time, init_tasks_num=self.init_tasks_num, load_task=load_task,
                              abc=abc)
        # self.sys.task_pool.set_fixed_release_points()
        self.map = self.sys.map
        self.vehicle_num = self.sys.vehicle_num
        action_num = self.vehicle_num * self.policy_num
        self.action_space = gym.spaces.Discrete(action_num)
        self.observation_space = gym.spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(4*self.max_task_num+self.vehicle_num*(self.max_task_num+3),),
                                                dtype=np.float32)
        self.current_tasks = None
        self.current_vehicles = None
        self.history_state = []


    def state_phi(self,s):
        # todo: structre representation
        tasks = s[0]
        vehicles = s[1]
        s = []
        max_tasks = self.max_task_num
        for task in tasks:
            task_s = []
            task_s.append((task.get_minimum_distance(self.map.running_map) - 153) / 73)
            task_s.append((task.waiting_time - 126.5) / 223.5)
            task_s.append((task.due - 404) / 77)
            task_s.append((task.d - 10.5) / 3.5)
            s += task_s
        while len(s) < 4 * max_tasks:
            s.append(0)

        for v in vehicles:
            v_s = []
            v_s.append(v.vl / 1.5)
            v_s.append(v.status / 2)
            v_s.append(v.c / 20)
            t_d = []
            for task in tasks:
                t_d.append((task.get_actual_distance(v, self.map.running_map) - 288) / 130)
            while len(t_d) < max_tasks:
                t_d.append(0)
            v_s += t_d
            s += v_s
        return s

    def valid_actions(self, state):
        current_tasks = state[0]
        current_vehicles = state[1]
        v_free_index = []
        for i,v in enumerate(current_vehicles):
            if v.status==0:
                v_free_index.append(i)
        valid_actions = []
        for i,task in enumerate(current_tasks):
            for v_i in v_free_index:
                free_v = current_vehicles[v_i]
                if task.is_feasible_choice_constraint(free_v):
                    # print([v_i*self.policy_num +index for index in range(self.policy_num)])
                    valid_actions +=[v_i*self.policy_num +index for index in range(self.policy_num)]
        valid_actions = list(set(valid_actions))
        # print(valid_actions)
        return valid_actions


    def reward_phi(self,state):
        current_vechiles = state[1]
        tardiness = 0
        tasks_num = 0
        for v in current_vechiles:
            for task in v.history_list:
                tardiness +=task.delay
                tasks_num+=1
        if tasks_num==0:
            return 0
        avg_tardiness = -tardiness/tasks_num
        return avg_tardiness

    def step(self,action):
        policy_index = int(action%self.policy_num)
        vehicle_index = int(action/self.policy_num)
        info = {}
        vehicle = self.current_vehicles[vehicle_index]
        decision = self.policys[policy_index].act(self.current_tasks,vehicle,self.sys.map.running_map)
        decision_list = decision
        if decision_list==None:
            info["feasible"] = False
            return None, None, None, info

        if self.is_feasible_ours(decision_list[0],decision_list[1]):
            info["decision"] = (self.current_tasks[decision_list[0]].name,decision_list[1].name)
            info["feasible"] = True
            r_phi_t = self.reward_phi([self.current_tasks, self.current_vehicles])
            r_m_t = self.reward_func([self.current_tasks, self.current_vehicles])
            self.current_tasks, self.current_vehicles, done, _ = self.sys.step(decision_list)
            state = [self.current_tasks, self.current_vehicles]
            r = self.reward_func(state)
            free_vehicles = [v for v in self.current_vehicles if v.status == 0]
            available_decision = self.available_vehicle(self.current_tasks,free_vehicles)
            # There are no avalable decision and the system is still running
            while not available_decision and not done:
                self.current_tasks, self.current_vehicles, done, _ = self.sys.step(None)
                free_vehicles = [v for v in self.current_vehicles if v.status == 0]
                state = [self.current_tasks, self.current_vehicles]
                r = self.reward_func(state)
                available_decision = self.available_vehicle(self.current_tasks, free_vehicles)

            r_phi_t2 = self.reward_phi([self.current_tasks, self.current_vehicles])
            r_m_t2 = self.reward_func([self.current_tasks, self.current_vehicles])
            r_f = self.gamma * r_phi_t2 - r_phi_t
            reward = r_m_t2 - r_m_t
            info["reward_F"] = r_f
            info["reward_makespan"] = reward
            info["cost"] = r_f
            if len(state[0]) > 0:
                self.history_state.append([task.name for task in state[0]])
            if done:
                # print("release",self.sys.task_pool.release_points)
                result = self.sys.get_result()
                episode_makespan = result["makespan"]
                episode_tardienss = result["avg_delay"]
                reward = -episode_makespan
                info["cost"] = episode_tardienss
            else:
                reward=0
                info["cost"] = 0

            return state, reward, done, info
        else:
            info["feasible"] = False
            return None, None, None, info

    def available_vehicle(self,tasks,vehicles):
        for task in tasks:
            for v in vehicles:
                if task.is_feasible_choice_constraint(v):
                    return True
        return False

    def is_feasible(self,task_index,vehicle):
        return self.current_tasks[task_index].is_feasible_choice_simple(vehicle)

    def is_feasible_ours(self,task_index,vehicle):
        return self.current_tasks[task_index].is_feasible_choice_constraint(vehicle)

    def reward_func(self,state):
        current_vechiles = state[1]
        tasks_num = 0
        tasks = []
        for v in current_vechiles:
            for task in v.history_list:
                tasks_num+=1
                tasks.append(task)
        if tasks_num == 0:
            return 0
        makespan = sorted(tasks,key=lambda x:x.finished_time)[-1].finished_time
        reward = -makespan
        return reward

    def reset(self):
        self.current_tasks = None
        self.current_vehicles = None
        self.history_state = []
        self.current_tasks, self.current_vehicles, done, _ = self.sys.reset()
        state = [self.current_tasks, self.current_vehicles]
        self.history_state.append([task.name for task in state[0]])
        return state

    def render(self, mode='human'):
        self.sys.info()


    def seed(self, seed):
        np.random.seed(seed)

    def close(self):
        pass

class AgvEnv_breakdown2(AgvEnv_breakdown1):
    def __init__(self,gamma=0.99,set_instance = False):
        super(AgvEnv_breakdown1, self).__init__(gamma,set_instance)
        self.temp_episic_reward = 0

    def set_instance(self,max_task_num,max_per_time,init_tasks_num,f_gamma,instance_name = "scenario1",load_task = False,task_pool = mTaskPool):
        file_path = "./gym_agv/system/dataset/%s/%s.json"%(instance_name,instance_name)
        self.max_task_num = max_task_num
        self.max_per_time = max_per_time
        self.init_tasks_num = init_tasks_num
        self.gamma = f_gamma
        abc = None
        self.sys = System_gym(file_path, None, task_pool, max_task_num=self.max_task_num,
                              max_per_time=self.max_per_time, init_tasks_num=self.init_tasks_num, load_task=load_task,
                              abc=abc)
        self.map=self.sys.map
        self.vehicle_num = self.sys.vehicle_num
        action_num = self.vehicle_num*self.policy_num
        self.action_space = gym.spaces.Discrete(action_num)
        self.observation_space = gym.spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(150,),
                                                dtype=np.float32)
        self.current_tasks = None
        self.current_vehicles = None
        self.history_state = []

    def state_phi(self, s):
        # todo: structure representation
        tasks = s[0]
        vehicles = s[1]
        s = []
        max_tasks = 15
        for task in tasks:
            task_s = []
            task_s.append((task.get_minimum_distance(self.map.running_map) - 153) / 73)
            task_s.append((task.waiting_time - 126.5) / 223.5)
            task_s.append((task.due - 404) / 77)
            task_s.append((task.d - 10.5) / 3.5)
            s += task_s
        while len(s) < 4 * max_tasks:
            s.append(0)

        for v in vehicles:
            v_s = []
            v_s.append(v.vl / 1.5)
            v_s.append(v.status / 2)
            v_s.append(v.c / 20)
            t_d = []
            for task in tasks:
                t_d.append((task.get_actual_distance(v, self.map.running_map) - 288) / 130)
            while len(t_d) < max_tasks:
                t_d.append(0)
            v_s += t_d
            s += v_s
        return np.array(s)

    def step(self, action):
        policy_index = int(action % self.policy_num)
        vehicle_index = int(action / self.policy_num)
        info = {}
        vehicle = self.current_vehicles[vehicle_index]
        decision = self.policys[policy_index].act(self.current_tasks, vehicle, self.sys.map.running_map)
        decision_list = decision
        if decision_list == None:
            info["feasible"] = False
            return None, None, None, info

        if self.is_feasible_ours(decision_list[0], decision_list[1]):
            info["decision"] = (self.current_tasks[decision_list[0]].name, decision_list[1].name)
            info["feasible"] = True
            r_phi_t = self.reward_phi([self.current_tasks, self.current_vehicles])
            r_m_t = self.reward_func([self.current_tasks, self.current_vehicles])

            self.current_tasks, self.current_vehicles, done, _ = self.sys.step(decision_list)
            state = [self.current_tasks, self.current_vehicles]
            r = self.reward_func(state)
            free_vehicles = [v for v in self.current_vehicles if v.status == 0]
            available_decision = self.available_vehicle(self.current_tasks, free_vehicles)
            # There are no avalable decision and the system is still running
            while not available_decision and not done:
                self.current_tasks, self.current_vehicles, done, _ = self.sys.step(None)
                free_vehicles = [v for v in self.current_vehicles if v.status == 0]
                state = [self.current_tasks, self.current_vehicles]
                r = self.reward_func(state)
                available_decision = self.available_vehicle(self.current_tasks, free_vehicles)

            r_phi_t2 = self.reward_phi([self.current_tasks, self.current_vehicles])
            r_m_t2 = self.reward_func([self.current_tasks, self.current_vehicles])
            r_f = r_phi_t2 - r_phi_t
            reward = r_m_t2 - r_m_t
            info["reward_F"] = r_f
            info["reward_makespan"] = reward
            self.temp_episic_reward+=reward
            if len(state[0]) > 0:
                self.history_state.append([task.name for task in state[0]])
            if done:
                # print("release",self.sys.task_pool.release_points)

                result = self.sys.get_result()
                episode_makespan = result["makespan"]
                reward = -result["makespan"]
            else:
                reward=0
            return state, reward, done, info
        else:
            info["feasible"] = False
            return None, None, None, info

    def reset(self):
        self.temp_episic_reward=0
        self.current_tasks = None
        self.current_vehicles = None
        self.history_state = []
        self.current_tasks, self.current_vehicles, done, _ = self.sys.reset()
        state = [self.current_tasks, self.current_vehicles]
        self.history_state.append([task.name for task in state[0]])
        return state

class AgvEnv_breakdown_hu(gym.Env):
    # dqn, policy-vehicle, no relaxed delay
    def __init__(self, gamma=0.99, set_instance=False):
        file_path = "./gym_agv/system/dataset/scenario1/scenario1.json"
        std = STD_gym()
        fcfs = FCFS_gym()
        edd = EDD_gym()
        lwt = LWT_gym()
        nvf = NVF_gym()
        self.policys = [fcfs, edd, nvf,lwt, std]
        self.policy_num = len(self.policys)
        self.gamma = gamma
        self.max_task_num = 30
        self.max_per_time = 30
        self.init_tasks_num = 5
        self.scenario_name = "scenario2"
        abc = None
        self.gamma = gamma
        if not set_instance:
            self.sys = System_gym(file_path, None, mTaskPool, max_task_num=self.max_task_num,
                                  max_per_time=self.max_per_time, init_tasks_num=self.init_tasks_num, abc=abc)
            self.map = self.sys.map
            self.vehicle_num = self.sys.vehicle_num
            action_num = self.vehicle_num * self.max_per_time
            self.action_space = gym.spaces.Discrete(action_num)
            self.observation_space = gym.spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(7,),
                                                    dtype=np.float32)
            self.current_tasks = None
            self.current_vehicles = None
            self.history_state = []

    def set_instance(self, max_task_num, max_per_time, init_tasks_num, f_gamma, scenario_name="scenario1",
                     load_task=False,
                     task_pool=mTaskPool):
        file_path = "./gym_agv/system/dataset/%s/%s.json" % (scenario_name, scenario_name)
        self.scenario_name = scenario_name
        self.max_task_num = max_task_num
        self.max_per_time = max_per_time
        self.init_tasks_num = init_tasks_num
        self.gamma = f_gamma
        abc = None
        self.sys = System_gym(file_path, None, task_pool, max_task_num=self.max_task_num,
                              max_per_time=self.max_per_time, init_tasks_num=self.init_tasks_num, load_task=load_task,
                              abc=abc)
        # self.sys.task_pool.set_fixed_release_points()
        self.map = self.sys.map
        self.vehicle_num = self.sys.vehicle_num
        action_num = self.vehicle_num * self.max_per_time
        self.action_space = gym.spaces.Discrete(action_num)
        self.observation_space = gym.spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(7,),
                                                dtype=np.float32)
        self.current_tasks = None
        self.current_vehicles = None
        self.history_state = []

    def state_phi(self, s):
        tasks = s[0]
        vehicles = s[1]
        Nt = len(tasks)
        # print(tasks)
        if Nt==0:
            Nt=1
        Tart = 0
        Dadt = 0

        for task in tasks:
            Tart += task.due-task.waiting_time
            Dadt += task.get_minimum_distance(self.map.running_map)
        Tart /=Nt
        Dadt /=Nt
        Ast = 0
        Avt = []
        binary_list = [4,2,1]
        for i,v in enumerate(vehicles):
            status = 0 if v.status==0 else 1
            Ast +=status*binary_list[i]
            Avt.append(v.vl)
        st = [Nt,Tart,Dadt,Ast]+Avt
        return st

    def valid_actions(self, state):

        current_tasks = state[0]
        current_vehicles = state[1]
        v_free_index = []
        for i, v in enumerate(current_vehicles):
            if v.status == 0:
                v_free_index.append(i)
        valid_actions = []
        for i, task in enumerate(current_tasks):
            for v_i in v_free_index:
                free_v = current_vehicles[v_i]
                if task.is_feasible_choice_constraint(free_v):
                    # print([v_i*self.policy_num +index for index in range(self.policy_num)])
                    valid_actions += [v_i * self.policy_num + index for index in range(self.policy_num)]
        valid_actions = list(set(valid_actions))
        # print(valid_actions)
        return valid_actions

    def reward_phi(self, state):
        current_vechiles = state[1]
        tardiness = 0
        tasks_num = 0
        for v in current_vechiles:
            for task in v.history_list:
                tardiness += task.delay
                tasks_num += 1
        if tasks_num == 0:
            return 0
        avg_tardiness = -tardiness / tasks_num
        return avg_tardiness

    def step(self, action):
        policy_index = int(action % self.policy_num)
        vehicle_index = int(action / self.policy_num)
        info = {}
        vehicle = self.current_vehicles[vehicle_index]
        decision = self.policys[policy_index].act(self.current_tasks, vehicle, self.sys.map.running_map)
        decision_list = decision
        if decision_list == None:
            info["feasible"] = False
            return None, None, None, info

        if self.is_feasible_ours(decision_list[0], decision_list[1]):
            info["decision"] = (self.current_tasks[decision_list[0]].name, decision_list[1].name)
            info["feasible"] = True
            r_phi_t = self.reward_phi([self.current_tasks, self.current_vehicles])
            r_m_t = self.reward_func([self.current_tasks, self.current_vehicles])
            self.current_tasks, self.current_vehicles, done, _ = self.sys.step(decision_list)
            state = [self.current_tasks, self.current_vehicles]
            r = self.reward_func(state)
            free_vehicles = [v for v in self.current_vehicles if v.status == 0]
            available_decision = self.available_vehicle(self.current_tasks, free_vehicles)
            # There are no avalable decision and the system is still running
            while not available_decision and not done:
                self.current_tasks, self.current_vehicles, done, _ = self.sys.step(None)
                free_vehicles = [v for v in self.current_vehicles if v.status == 0]
                state = [self.current_tasks, self.current_vehicles]
                r = self.reward_func(state)
                available_decision = self.available_vehicle(self.current_tasks, free_vehicles)

            r_phi_t2 = self.reward_phi([self.current_tasks, self.current_vehicles])
            r_m_t2 = self.reward_func([self.current_tasks, self.current_vehicles])
            r_f = self.gamma * r_phi_t2 - r_phi_t
            reward = r_m_t2 - r_m_t
            info["reward_F"] = r_f
            info["reward_makespan"] = reward
            info["cost"] = r_f
            if len(state[0]) > 0:
                self.history_state.append([task.name for task in state[0]])
            if done:
                # print("release",self.sys.task_pool.release_points)
                result = self.sys.get_result()
                episode_makespan = result["makespan"]
                episode_tardienss = result["avg_delay"]
                reward = -episode_makespan
                info["cost"] = episode_tardienss
            else:
                reward = 0
                info["cost"] = 0

            return state, reward, done, info
        else:
            info["feasible"] = False
            return None, None, None, info

    def available_vehicle(self, tasks, vehicles):
        for task in tasks:
            for v in vehicles:
                if task.is_feasible_choice_constraint(v):
                    return True
        return False

    def is_feasible(self, task_index, vehicle):
        return self.current_tasks[task_index].is_feasible_choice_simple(vehicle)

    def is_feasible_ours(self, task_index, vehicle):
        return self.current_tasks[task_index].is_feasible_choice_constraint(vehicle)

    def reward_func(self, state):
        current_vechiles = state[1]
        tasks_num = 0
        tasks = []
        for v in current_vechiles:
            for task in v.history_list:
                tasks_num += 1
                tasks.append(task)
        if tasks_num == 0:
            return 0
        makespan = sorted(tasks, key=lambda x: x.finished_time)[-1].finished_time
        reward = -makespan
        return reward

    def reset(self):
        self.current_tasks = None
        self.current_vehicles = None
        self.history_state = []
        self.current_tasks, self.current_vehicles, done, _ = self.sys.reset()
        state = [self.current_tasks, self.current_vehicles]
        self.history_state.append([task.name for task in state[0]])
        return state

    def render(self, mode='human'):
        self.sys.info()

    def seed(self, seed):
        np.random.seed(seed)

    def close(self):
        pass




