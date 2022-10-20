from dataclasses import dataclass


@dataclass
class ExpConfig:
    exp_name: str
    render: bool
    repetitions: int
    cem_stages: int
    episodes: int
    timesteps: int
    vel_burn_in_time: int
    pem_path: str
    safety_func: str
