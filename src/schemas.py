from typing import Optional, Union

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        example="Data entry clerk needed. Work from home. No experience required.",
    )


class PredictionResponse(BaseModel):
    prediction: Union[int, str]
    is_fraud: Optional[bool]
    score: Optional[float]
    model_name: str
