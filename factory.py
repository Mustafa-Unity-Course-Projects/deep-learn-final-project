import random

from pydantic import BaseModel, Field


class FactoryState(BaseModel):
    employees_exist: bool
    brightness: float = Field(le=1.0, ge=0.0)
    temp: float = Field(ge=0, le=40)

    @classmethod
    def create_randomly(cls) -> "FactoryState":
        return cls(
            employees_exist=bool(random.getrandbits(1)),
            brightness=float(random.randrange(0, 100)) / 100.0,
            temp=float(random.randrange(0, 40)),
        )

    def to_list(self):
        return [self.employees_exist, self.brightness, self.temp]


class Factory:
    state: FactoryState

    def __init__(self):
        self.next_state()

    def next_state(self):
        self.state = FactoryState.create_randomly()
        return self.state

    def action_reward(self, action) -> float:
        decision = action[0]
        light_on = decision % 2
        ac_on = decision // 2
        state = self.state

        light_required = state.employees_exist and state.brightness < 0.5
        ac_required = state.employees_exist and state.temp > 25

        if light_on == light_required and ac_on == ac_required:
            return 10.0
        else:
            return -10.0
