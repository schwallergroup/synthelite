"""Classes defining the typing for synthelite outputs."""

from typing import Optional
from pydantic import BaseModel
import wandb


class RouteMetadata(BaseModel):
    time_total: float
    time_retro: float
    time_conditions: float


class EvalFormat(RouteMetadata):
    target_smi: str
    target_img: Optional[wandb.Image] = None
    json_validity: int
    routes_found: int = 0
    routes_solved: int = 0

    class Config:
        arbitrary_types_allowed = True
