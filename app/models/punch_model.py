from pydantic import Field, BaseModel 

class PunchUpdate(BaseModel):
    x : float = Field(alias='x', description='x-axis of accelerometer')
    y : float = Field(alias='y', description='y-axis of accelerometer')
    z : float = Field(alias='z', description='z-axis of accelerometer')
    session_id: str = Field(alias='session_id', description='session being recorded')

    #no need to add timestamp and session id