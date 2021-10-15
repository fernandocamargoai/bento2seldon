from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

import enum
import os

from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

PRED_UNIT_KEY = "predictive_unit_id"
PRED_UNIT_ID = os.environ.get("PREDICTIVE_UNIT_ID", "0")
DEPLOYMENT_ID = os.environ.get("SELDON_DEPLOYMENT_ID", "0")


class StatusFlag(enum.Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class Status(BaseModel):
    code: int = 200
    info: Optional[str]
    reason: Optional[str]
    status: str = StatusFlag.SUCCESS.value


class Meta(BaseModel):
    puid: str = ""
    tags: Dict[str, Any] = Field(default_factory=dict)
    routing: Dict[str, int] = Field(default_factory=dict)
    requestPath: Dict[str, str] = Field(default_factory=dict)

    def __init__(self, **data: Any):
        super().__init__(**data)
        if PRED_UNIT_KEY not in self.tags:
            self.tags[PRED_UNIT_KEY] = PRED_UNIT_ID


R = TypeVar("R", bound=BaseModel)


class SeldonMessage(GenericModel, Generic[R]):
    status: Optional[Status]
    meta: Meta = Field(default_factory=Meta)
    jsonData: Optional[R]


class SeldonMessageRequest(SeldonMessage[R], Generic[R]):
    jsonData: R


S = TypeVar("S", bound=BaseModel)


class Feedback(GenericModel, Generic[R, S]):
    request: Optional[SeldonMessage[R]]
    response: Optional[SeldonMessage[S]]
    reward: Optional[float]
    truth: Optional[SeldonMessage[S]]


class Tensor(BaseModel):
    shape: List[int]
    values: List[Union[int, float]]


class DefaultData(BaseModel):
    names: Optional[List[str]]
    tensor: Optional[Tensor]


class RoutingSeldonMessage(BaseModel):
    status: Optional[Status]
    meta: Meta = Field(default_factory=Meta)
    data: DefaultData
