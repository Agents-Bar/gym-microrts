import json
import logging
import os
import xml.etree.ElementTree as ET

import gym
import gym_microrts
import jpype
import jpype.imports
import numpy as np
from gym_microrts.utils import to_numpy
from jpype.imports import registerDomain
from jpype.types import JArray
from PIL import Image

JARS = [
    "microrts.jar", "Coac.jar", "Droplet.jar", "GRojoA3N.jar",
    "Izanagi.jar", "MixedBot.jar", "RojoBot.jar", "TiamatBot.jar", "UMSBot.jar" # "MindSeal.jar"
]

class MicroRTSGridModeVecEnv:
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 150
    }
    """
    [[0]x_coordinate*y_coordinate(x*y), [1]a_t(6), [2]p_move(4), [3]p_harvest(4), 
    [4]p_return(4), [5]p_produce_direction(4), [6]p_produce_unit_type(z), 
    [7]x_coordinate*y_coordinate(x*y)]

    Create a baselines VecEnv environment from a gym3 environment.

    :param env: gym3 environment to adapt
    """

    def __init__(self,
        num_selfplay_envs,
        num_bot_envs,
        partial_obs=False,
        max_steps=2000,
        frame_skip=0,
        ai2s=[],
        map_path="maps/10x10/basesTwoWorkers10x10.xml",
        reward_weight=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 5.0])):
        self.logger = logging.getLogger("MicroRTSGridEnv")

        self.num_selfplay_envs = num_selfplay_envs
        self.num_bot_envs = num_bot_envs
        self.num_envs = num_selfplay_envs + num_bot_envs
        assert self.num_bot_envs == len(ai2s), "for each environment, a microrts ai should be provided"
        self.partial_obs = partial_obs
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.ai2s = ai2s
        self.map_path = map_path
        self.reward_weight = to_numpy(reward_weight)

        # read map
        self.microrts_path = os.path.join(gym_microrts.__path__[0], 'microrts')
        root = ET.parse(os.path.join(self.microrts_path, self.map_path)).getroot()
        self.height, self.width = int(root.get("height")), int(root.get("width"))

        # launch the JVM
        if not jpype._jpype.isStarted():
            registerDomain("ts", alias="tests")
            registerDomain("ai")
            for jar in JARS:
                jpype.addClassPath(os.path.join(self.microrts_path, jar))
            jpype.startJVM(convertStrings=False)

        # start microrts client
        from rts.units import UnitTypeTable
        self.real_utt = UnitTypeTable()
        from ai.rewardfunction import (  # CloserToEnemyBaseRewardFunction,
            AttackRewardFunction, ProduceBuildingRewardFunction,
            ProduceCombatUnitRewardFunction, ProduceWorkerRewardFunction,
            ResourceGatherRewardFunction, RewardFunctionInterface,
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
        self.observation_space = gym.spaces.Box(low=0.0,
            high=1.0,
            shape=(self.height, self.width,
                    sum(self.num_planes)),
                    dtype=np.int32)
        self.action_space = gym.spaces.MultiDiscrete([
            self.height * self.width,
            6, 4, 4, 4, 4,
            len(self.utt['unitTypes']),
            7 * 7
        ])

    def start_client(self) -> None:
        """Start Client to communicate with microRTS environment.

        Client is accessable as `vec_client` property on the instance.
        """
        self.logger.debug("Initiatlize client")

        from ai.core import AI
        from ts import JNIGridnetVecClient as Client
        self.vec_client = Client(
            self.num_selfplay_envs,
            self.num_bot_envs,
            self.max_steps,
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_path,
            JArray(AI)([ai2(self.real_utt) for ai2 in self.ai2s]),
            self.real_utt,
            self.partial_obs,
        )
        self.render_client = self.vec_client.selfPlayClients[0] if len(self.vec_client.selfPlayClients) > 0 else self.vec_client.clients[0]
        # get the unit type table
        self.utt = json.loads(str(self.render_client.sendUTT()))

    def reset(self):
        self.logger.debug("Reseting environment")
        responses = self.vec_client.reset([0]*self.num_envs)
        raw_obs = np.array(responses.observation)
        reward = np.array(responses.reward)
        done = np.array(responses.done)
        info = {}
        obs = []
        for ro in raw_obs:
            obs += [self._encode_obs(ro)]
        return np.array(obs)

    def _encode_obs(self, obs):
        obs = obs.reshape(len(obs), -1).clip(0, np.array([self.num_planes]).T-1)
        obs_planes = np.zeros((self.height * self.width, sum(self.num_planes)), dtype=np.int)
        obs_planes[np.arange(len(obs_planes)),obs[0]] = 1

        for i in range(1, len(self.num_planes)):
            obs_planes[np.arange(len(obs_planes)),obs[i]+sum(self.num_planes[:i])] = 1
        return obs_planes.reshape(self.height, self.width, -1)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        e = [0 for _ in range(self.num_envs)]
        self.logger.info(self.actions)
        self.logger.info(e)

        responses = self.vec_client.gameStep(self.actions, e)
        raw_obs, reward, done = np.array(responses.observation), np.array(responses.reward), np.array(responses.done)
        obs = []
        for ro in raw_obs:
            obs += [self._encode_obs(ro)]
        infos = [{"raw_rewards": item} for item in reward]
        return np.array(obs), reward @ self.reward_weight, done[:,0], infos

    def step(self, ac):
        self.step_async(ac)
        return self.step_wait()
    
    def seed(self, seed) -> None:
        """Sets seed for random value generator.

        Currently only used by the Action Space.
        """
        self.logger.warning("This method does not set seed for the environment. Currently only sets for action space.")
        self.action_space.seed(seed)

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
        """Closes clients.
        This method should be used once the experiment is finished.
        """
        if jpype._jpype.isStarted():
            self.vec_client.close()
            jpype.shutdownJVM()