# -*- coding:utf-8 _*-


from gym.envs.registration import register


register(
    id='agv-breakdown-v0',
    entry_point='gym_agv.envs:AgvEnv_breakdown0',
)

register(
    id='agv-breakdown-v1',
    entry_point='gym_agv.envs:AgvEnv_breakdown1',
)
register(
    id='agv-breakdown-v2',
    entry_point='gym_agv.envs:AgvEnv_breakdown2',
)

# register(
#     id='agv-breakdown-hu',
#     entry_point='gym_agv.envs:AgvEnv_breakdown_hu',
# )
