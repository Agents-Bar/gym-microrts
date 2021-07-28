
import json
import logging
import os
import xml.etree.ElementTree as ET

import gym
import gym_microrts
import jpype
import numpy as np
from jpype.imports import registerDomain
from jpype.types import JArray
from PIL import Image

from .grid_mode_vec_env import MicroRTSGridModeVecEnv


class MicroRTSBotVecEnv(MicroRTSGridModeVecEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 150
    }

    def __init__(self,
        ai1s=[],
        ai2s=[],
        partial_obs=False,
        max_steps=2000,
        map_path="maps/10x10/basesTwoWorkers10x10.xml",
        reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 5.0])):
        self.logger = logging.getLogger("")

        self.ai1s = ai1s
        self.ai2s = ai2s
        assert len(ai1s) == len(ai2s), "for each environment, a microrts ai should be provided"
        self.num_envs = len(ai1s)
        self.partial_obs = partial_obs
        self.max_steps = max_steps
        self.map_path = map_path
        self.reward_weight = reward_weight

        # read map
        self.microrts_path = os.path.join(gym_microrts.__path__[0], 'microrts')
        root = ET.parse(os.path.join(self.microrts_path, self.map_path)).getroot()
        self.height, self.width = int(root.get("height")), int(root.get("width"))

        # launch the JVM
        if not jpype._jpype.isStarted():
            registerDomain("ts", alias="tests")
            registerDomain("ai")
            jars = [
                "microrts.jar", "Coac.jar", "Droplet.jar", "GRojoA3N.jar",
                "Izanagi.jar", "MixedBot.jar", "RojoBot.jar", "TiamatBot.jar", "UMSBot.jar" # "MindSeal.jar"
            ]
            for jar in jars:
                jpype.addClassPath(os.path.join(self.microrts_path, jar))
            jpype.startJVM(convertStrings=False)

        # start microrts client
        from rts.units import UnitTypeTable
        self.real_utt = UnitTypeTable()
        from ai.rewardfunction import (AttackRewardFunction,
                                       CloserToEnemyBaseRewardFunction,
                                       ProduceBuildingRewardFunction,
                                       ProduceCombatUnitRewardFunction,
                                       ProduceWorkerRewardFunction,
                                       ResourceGatherRewardFunction,
                                       RewardFunctionInterface,
                                       WinLossRewardFunction)
        self.rfs = JArray(RewardFunctionInterface)([
            WinLossRewardFunction(), 
            ResourceGatherRewardFunction(),  
            ProduceWorkerRewardFunction(),
            ProduceBuildingRewardFunction(),
            AttackRewardFunction(),
            ProduceCombatUnitRewardFunction(),
            # CloserToEnemyBaseRewardFunction(),
        ])
        self.start_client()

        self.num_planes = [5, 5, 3, len(self.utt['unitTypes'])+1, 6]
        if partial_obs:
            self.num_planes = [5, 5, 3, len(self.utt['unitTypes'])+1, 6, 2]
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)

    def start_client(self) -> None:
        """Start Client to communicate with microRTS environment.
        
        Client is accessable as `vec_client` property on the instance.
        """

        from ai.core import AI
        from ts import JNIGridnetVecClient as Client
        self.vec_client = Client(
            self.max_steps,
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_path,
            JArray(AI)([ai1(self.real_utt) for ai1 in self.ai1s]),
            JArray(AI)([ai2(self.real_utt) for ai2 in self.ai2s]),
            self.real_utt,
            self.partial_obs,
        )
        self.render_client = self.vec_client.botClients[0]
        # get the unit type table
        self.utt = json.loads(str(self.render_client.sendUTT()))

    def seed(self, seed: int) -> None:
        """Sets seed for action space"""
        self.action_space.seed(seed)
        self.logger.warning("")

    def reset(self):
        responses = self.vec_client.reset([0]*self.num_envs)
        raw_obs = np.ones((self.num_envs,2)),
        reward = np.array(responses.reward)
        done = np.array(responses.done)
        info = {}
        return raw_obs

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        e = [0 for _ in range(self.num_envs)]
        self.logger.info(self.actions)
        self.logger.info(e)

        responses = self.vec_client.gameStep(self.actions, e)
        raw_obs, reward, done = np.ones((self.num_envs,2)), np.array(responses.reward), np.array(responses.done)
        infos = [{"raw_rewards": item} for item in reward]
        return raw_obs, reward @ self.reward_weight, done[:,0], infos

    def step(self, ac):
        self.step_async(ac)
        return self.step_wait()

    def getattr_depth_check(self, name, already_found):
        """Check if an attribute reference is being hidden in a recursive call to __getattr__

        :param name: (str) name of attribute to check for
        :param already_found: (bool) whether this attribute has already been found in a wrapper
        :return: (str or None) name of module whose attribute is being shadowed, if any.
        """
        if hasattr(self, name) and already_found:
            return "{0}.{1}".format(type(self).__module__, type(self).__name__)
        else:
            return None

    def render(self, mode="human"):
        if mode == "human":
            self.render_client.render(False)
        elif mode == 'rgb_array':
            bytes_array = np.array(self.render_client.render(True))
            image = Image.frombytes("RGB", (640, 640), bytes_array)
            return np.array(image)[:,:,::-1]

    def close(self):
        if jpype._jpype.isStarted():
            self.vec_client.close()
            jpype.shutdownJVM()
