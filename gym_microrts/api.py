import logging
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, HTTPException

from gym_microrts.envs import MicroRTSBotVecEnv, MicroRTSGridModeVecEnv
from gym_microrts.microrts_ai import coacAI
from gym_microrts.types import (ActionType, EnvActionType, EnvStepType,
                                ObservationType, StepType)
from gym_microrts.utils import extract_space_info, to_list, to_numpy

app = FastAPI(title="Gym MicroRTS")
Environment = Union[MicroRTSGridModeVecEnv, MicroRTSBotVecEnv]
env: Optional[Environment] = None

last_step: Optional[StepType] = None
last_actions: Optional[ActionType] = None


@app.get("/ping")
def ping():
    "Purely for health checks. Unrelated to the gym."
    return "pong"


@app.post("/env", status_code=201)
def init_env(config: Optional[Dict[str, Any]] = None):
    global env
    config = config or {}
    env_name = config.get("env_name")
    if env_name is None or env_name != "BotVecEnv":
        num_selfplay_evs = config.get("num_selfplay_envs", 0)
        num_bot_envs = config.get("num_bot_envs", 1)
        max_steps = config.get("max_steps", 2000)
        ai2s_type = config.get("ai2s", "coacAI")

        assert ai2s_type == "coacAI", "Only 'coacAI' is supported"
        ai2s = [coacAI for _ in range(num_bot_envs)]
        map_path = config.get("map_path", "maps/16x16/basesWorkers16x16.xml")

        reward_weight = config.get("reward_weight", [10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
        assert len(reward_weight) == 6, "Reward weight is a list of 6 values"
        env = MicroRTSGridModeVecEnv(
            num_selfplay_envs=num_selfplay_evs,
            num_bot_envs=num_bot_envs,
            max_steps=max_steps,
            ai2s=ai2s,
            map_path=map_path,
            reward_weight=to_numpy(reward_weight)
        )
    else:
        env = MicroRTSBotVecEnv()


@app.post('/env/reset', response_model=ObservationType)
def reset_env(seed: Optional[int] = None) -> ObservationType:
    "Reset the environment to initial position."
    global env
    env = check_env(env)
    if seed:
        env.seed(seed)
    observation = env.reset()
    obs = to_list(observation)
    return obs


@app.post("/env/step", response_model=Optional[EnvStepType])
def post_step(env_action: EnvActionType):
    "Provides information necessary to step the environment."
    global env, last_step, last_actions
    env = check_env(env)
    last_actions = env_action.actions

    if env_action.commit:
        last_step = commit(env, last_actions)
        return last_step

    return None

@app.post('/env/commit', response_model=EnvStepType)
def post_commit() -> EnvStepType:
    "Commit last sent step data."
    global env, last_actions, last_step
    if last_actions is None:
        raise HTTPException(400, "Cannot commit action without passing action. Please use `/env/step` first.")
    env = check_env(env)
    last_step = commit(env, last_actions)
    # Clear action after commiting
    last_actions = None
    return last_step


@app.get('/env/last', response_model=EnvStepType)
def get_last() -> EnvStepType:
    "Retrieve last provided Step data."
    if last_step is None:
        raise HTTPException(404, detail="No environment information to return")
    return last_step

@app.post('/env/seed')
def set_seed(seed: int) -> None:
    "Set seed for environment's random number generator."
    global env
    env = check_env(env)
    env.seed(seed)
    return None


@app.get('/env/info')
def get_env_info() -> Dict[str, Any]:
    assert env
    obs_space = env.observation_space
    action_space = env.action_space
    return {
        "observation_space": extract_space_info(obs_space),
        "action_space": extract_space_info(action_space)
    }


def check_env(env: Optional[Environment]) -> Environment:
    """Make sure env (environment) is defined correctly.
    Intended to use directly and shortly after receiving API call.
    Raises:
        HTTPException if there is anything wrong with provided environment.
    """
    if env is None:
        detail ="Environment is not instantiated. Use POST /env to initiate."
        raise HTTPException(400, detail=detail)
    return env


def commit(env: Environment, actions: ActionType) -> EnvStepType:
    """Logic part that commits data to environment engine.
    Raises:
        HTTPException if there is anything wrong with provided data either before
        or after passing to the environment.
    """
    try:
        out = env.step([actions])
    except Exception as e:
        logging.exception("Something wrong while commiting step")
        raise HTTPException(500, str(e))
    obs = to_list(out[0])
    reward = to_list(out[1])
    done = to_list(out[2])
    info = {"info": str(out[3])}
    step = EnvStepType(observation=obs, reward=reward, done=done, info=info)
    return step
