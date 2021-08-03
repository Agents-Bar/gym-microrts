from typing import Any, Dict, List, Union
from pydantic import BaseModel

ActionType = List[List[int]]
StepType = Any
ObservationType = Any
DoneType = Union[bool, List[bool]]
RewardType = Union[int, float, List[float]]

class EnvStepType(BaseModel):
    observation: ObservationType
    reward: RewardType
    done: DoneType
    info: Dict

class EnvActionType(BaseModel):
    actions: ActionType
    commit: bool = True

