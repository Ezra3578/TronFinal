import pygame
import numpy as np
import time

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from environment import TronParallelEnv

CHECKPOINT_PATH = "./tron_ppo_2team"

def env_creator(_):
    return ParallelPettingZooEnv(TronParallelEnv(render_mode="human"))

register_env("tron_env", env_creator)

def policy_mapping_fn(agent_id):
    if agent_id in ["player_1", "player_3"]:
        return "red_team_policy"
    else:
        return "blue_team_policy"

algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)
env = env_creator({})

for episode in range(10):
    obs, _ = env.reset()
    terminated = {"__all__": False}
    truncated = {"__all__": False}

    while not terminated["__all__"] and not truncated["__all__"]:
        actions = {}

        for agent_id, agent_obs in obs.items():
            policy_id = policy_mapping_fn(agent_id)
            policy = algo.get_policy(policy_id)
            action, _, _ = policy.compute_single_action(agent_obs, explore=False)
            actions[agent_id] = action

        obs, rewards, terminated, truncated, infos = env.step(actions)
        time.sleep(0.015)

env.close()
