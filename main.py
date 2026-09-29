from typing import Protocol

import rerun as rr
import torch

import models.fossen_torch as rov_plant
import models.fossen_diff as rov_predicition_model

def controller(state: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
	kp = 10.0
	return kp * (reference - state)


def simulate(plant: Plant, x0: torch.Tensor, reference: torch.Tensor, dt: float, steps: int) -> torch.Tensor:
	x = x0
	trajectory = [x]
	for k in range(steps):
		tau = controller(x, reference)
		x = plant.step(x, tau, dt)
		trajectory.append(x)

		rr.set_time("sim_time", duration=(k + 1) * dt)
		for i, name in enumerate(plant.state_names):
			rr.log(f"state/{name}", rr.Scalars(x[0, i].item()))
		for i, name in enumerate(plant.input_names):
			rr.log(f"input/{name}", rr.Scalars(tau[0, i].item()))
	return torch.stack(trajectory, dim=1)  # (batch, steps + 1, n_state)


if __name__ == "__main__":
	rr.init("uuv_sim", spawn=True)
	x0 = torch.zeros(1, len(rov_plant))
	reference = torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
	simulate(rov_plant, x0, reference, dt=0.01, steps=1000)
