# Vendored from HunyuanVideo-Foley utils/schedulers/scheduling_flow_match_discrete.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin

logger = logging.get_logger(__name__)

@dataclass
class FlowMatchDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor

class FlowMatchDiscreteScheduler(SchedulerMixin, ConfigMixin):
    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        reverse: bool = True,
        solver: str = "euler",
        use_flux_shift: bool = False,
        flux_base_shift: float = 0.5,
        flux_max_shift: float = 1.15,
        n_tokens: Optional[int] = None,
    ):
        sigmas = torch.linspace(1, 0, num_train_timesteps + 1)
        if not reverse:
            sigmas = sigmas.flip(0)
        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)
        self.timesteps_full = (sigmas * num_train_timesteps).to(dtype=torch.float32)
        self._step_index = None
        self._begin_index = None
        self.supported_solver = ["euler", "heun-2", "midpoint-2", "kutta-4"]
        if solver not in self.supported_solver:
            raise ValueError(f"Solver {solver} not supported. Supported solvers: {self.supported_solver}")
        self.derivative_1 = None
        self.derivative_2 = None
        self.derivative_3 = None
        self.dt = None

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    @property
    def state_in_first_order(self):
        return self.derivative_1 is None

    @property
    def state_in_second_order(self):
        return self.derivative_2 is None

    @property
    def state_in_third_order(self):
        return self.derivative_3 is None

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None, n_tokens: int = None):
        self.num_inference_steps = num_inference_steps
        sigmas = torch.linspace(1, 0, num_inference_steps + 1)
        if self.config.use_flux_shift:
            assert isinstance(n_tokens, int), "n_tokens should be provided for flux shift"
            mu = self.get_lin_function(y1=self.config.flux_base_shift, y2=self.config.flux_max_shift)(n_tokens)
            sigmas = self.flux_time_shift(mu, 1.0, sigmas)
        elif self.config.shift != 1.0:
            sigmas = self.sd3_time_shift(sigmas)
        if not self.config.reverse:
            sigmas = 1 - sigmas
        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.config.num_train_timesteps).to(dtype=torch.float32, device=device)
        self.timesteps_full = (sigmas * self.config.num_train_timesteps).to(dtype=torch.float32, device=device)
        self.derivative_1 = None
        self.derivative_2 = None
        self.derivative_3 = None
        self.dt = None
        self._step_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        return sample

    @staticmethod
    def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    @staticmethod
    def flux_time_shift(mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def sd3_time_shift(self, t: torch.Tensor):
        return (self.config.shift * t) / (1 + (self.config.shift - 1) * t)

    def step(self, model_output: torch.FloatTensor, timestep: Union[float, torch.FloatTensor], sample: torch.FloatTensor,
             pred_uncond: torch.FloatTensor = None, generator: Optional[torch.Generator] = None,
             n_tokens: Optional[int] = None, return_dict: bool = True) -> Union[FlowMatchDiscreteSchedulerOutput, Tuple]:
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError("Pass one of scheduler.timesteps as timestep, not integer indices.")
        if self.step_index is None:
            self._init_step_index(timestep)
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)
        pred_uncond = pred_uncond.to(torch.float32) if pred_uncond is not None else None
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        last_inner_step = True
        if self.config.solver == "euler":
            derivative, dt, sample, last_inner_step = self.first_order_method(model_output, sigma, sigma_next, sample)
        elif self.config.solver in ["heun-2", "midpoint-2"]:
            derivative, dt, sample, last_inner_step = self.second_order_method(model_output, sigma, sigma_next, sample)
        elif self.config.solver == "kutta-4":
            derivative, dt, sample, last_inner_step = self.fourth_order_method(model_output, sigma, sigma_next, sample)
        else:
            raise ValueError(f"Unsupported solver {self.config.solver}")
        prev_sample = sample + derivative * dt
        if last_inner_step:
            self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return FlowMatchDiscreteSchedulerOutput(prev_sample=prev_sample)

    def first_order_method(self, model_output, sigma, sigma_next, sample):
        derivative = model_output.float()
        dt = sigma_next - sigma
        return derivative, dt, sample, True

    def second_order_method(self, model_output, sigma, sigma_next, sample):
        if self.state_in_first_order:
            self.derivative_1 = model_output
            self.dt = sigma_next - sigma
            self.sample = sample
            derivative = model_output
            dt = self.dt if self.config.solver == "heun-2" else self.dt / 2
            last_inner_step = False
        else:
            derivative = 0.5 * (self.derivative_1 + model_output) if self.config.solver == "heun-2" else model_output
            dt = self.dt
            sample = self.sample
            last_inner_step = True
            self.derivative_1 = None
            self.dt = None
            self.sample = None
        return derivative, dt, sample, last_inner_step

    def fourth_order_method(self, model_output, sigma, sigma_next, sample):
        if self.state_in_first_order:
            self.derivative_1 = model_output
            self.dt = sigma_next - sigma
            self.sample = sample
            derivative = model_output
            dt = self.dt / 2
            last_inner_step = False
        elif self.state_in_second_order:
            self.derivative_2 = model_output
            derivative = model_output
            dt = self.dt / 2
            last_inner_step = False
        elif self.state_in_third_order:
            self.derivative_3 = model_output
            derivative = model_output
            dt = self.dt
            last_inner_step = False
        else:
            derivative = 1/6 * self.derivative_1 + 1/3 * self.derivative_2 + 1/3 * self.derivative_3 + 1/6 * model_output
            dt = self.dt
            sample = self.sample
            last_inner_step = True
            self.derivative_1 = None
            self.derivative_2 = None
            self.derivative_3 = None
            self.dt = None
            self.sample = None
        return derivative, dt, sample, last_inner_step

