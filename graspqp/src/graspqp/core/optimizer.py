"""
Based on Dexgraspnet: https://pku-epic.github.io/DexGraspNet/
"""

import torch
from torch.distributions import Normal


import torch

normal = Normal(0, 1)


class AnnealingDexGraspNet:
    def __init__(self, hand_model, switch_possibility=0.5, starting_temperature=18, temperature_decay=0.95, annealing_period=30, step_size=0.005, stepsize_period=50, mu=0.98, device="cpu", **kwargs):
        """
        Create a optimizer

        Use random resampling to update contact point indices

        Use RMSProp to update translation, rotation, and joint angles, use step size decay

        Use Annealing to accept / reject parameter updates

        Parameters
        ----------
        hand_model: hand_model.HandModel
        switch_possibility: float
            possibility to resample each contact point index each step
        starting_temperature: float
        temperature_decay: float
            temperature decay rate and step size decay rate
        annealing_period: int
        step_size: float
        stepsize_period: int
        mu: float
            `1 - decay_rate` of RMSProp
        """

        self.hand_model = hand_model
        self.device = device
        self.switch_possibility = switch_possibility
        self.starting_temperature = torch.tensor(starting_temperature, dtype=torch.float, device=device)
        self.temperature_decay = torch.tensor(temperature_decay, dtype=torch.float, device=device)
        self.annealing_period = torch.tensor(annealing_period, dtype=torch.long, device=device)
        self.step_size = torch.tensor(step_size, dtype=torch.float, device=device)
        self.step_size_period = torch.tensor(stepsize_period, dtype=torch.long, device=device)
        self.mu = torch.tensor(mu, dtype=torch.float, device=device)
        self.step = 0

        self.old_hand_pose = None
        self.old_contact_point_indices = None
        self.old_global_transformation = None
        self.old_global_rotation = None
        self.old_current_status = None
        self.old_contact_points = None
        self.old_grad_hand_pose = None
        self.ema_grad_hand_pose = torch.zeros(self.hand_model.n_dofs + 9, dtype=torch.float, device=device)

    def try_step(self, *args, **kwargs):
        """
        Try to update translation, rotation, joint angles, and contact point indices

        Returns
        -------
        s: torch.Tensor
            current step size
        """

        s = self.step_size * self.temperature_decay ** torch.div(self.step, self.step_size_period, rounding_mode="floor")
        step_size = torch.zeros(*self.hand_model.hand_pose.shape, dtype=torch.float, device=self.device) + s

        self.ema_grad_hand_pose = self.mu * (self.hand_model.hand_pose.grad**2).mean(0) + (1 - self.mu) * self.ema_grad_hand_pose

        hand_pose = self.hand_model.hand_pose - step_size * self.hand_model.hand_pose.grad / (torch.sqrt(self.ema_grad_hand_pose) + 1e-6)
        batch_size, n_contact = self.hand_model.contact_point_indices.shape
        switch_mask = torch.rand(batch_size, n_contact, dtype=torch.float, device=self.device) < self.switch_possibility
        contact_point_indices = self.hand_model.contact_point_indices.clone()
        contact_point_indices[switch_mask] = torch.randint(self.hand_model.n_contact_candidates, size=[switch_mask.sum()], device=self.device)

        self.old_hand_pose = self.hand_model.hand_pose
        self.old_contact_point_indices = self.hand_model.contact_point_indices
        self.old_global_transformation = self.hand_model.global_translation
        self.old_global_rotation = self.hand_model.global_rotation
        self.old_current_status = self.hand_model.current_status
        self.old_contact_points = self.hand_model.contact_points
        self.old_grad_hand_pose = self.hand_model.hand_pose.grad
        self.hand_model.set_parameters(hand_pose, contact_point_indices)

        self.step += 1

        return s

    def accept_step(self, energy, new_energy, *args, **kwargs):
        """
        Accept / reject updates using annealing

        Returns
        -------
        accept: (N,) torch.BoolTensor
        temperature: torch.Tensor
            current temperature
        """

        batch_size = energy.shape[0]
        temperature = self.starting_temperature * self.temperature_decay ** torch.div(self.step, self.annealing_period, rounding_mode="floor")

        alpha = torch.rand(batch_size, dtype=torch.float, device=self.device)
        accept = alpha < torch.exp((energy - new_energy) / temperature)

        with torch.no_grad():
            reject = ~accept
            self.hand_model.hand_pose[reject] = self.old_hand_pose[reject]
            self.hand_model.contact_point_indices[reject] = self.old_contact_point_indices[reject]
            self.hand_model.global_translation[reject] = self.old_global_transformation[reject]
            self.hand_model.global_rotation[reject] = self.old_global_rotation[reject]

            self.hand_model.current_status = self.hand_model.fk(self.hand_model.hand_pose[:, 9:])
            self.hand_model.contact_points[reject] = self.old_contact_points[reject]
            self.hand_model.hand_pose.grad[reject] = self.old_grad_hand_pose[reject]

        return accept, temperature

    def zero_grad(self):
        """
        Sets the gradients of translation, rotation, and joint angles to zero
        """
        if self.hand_model.hand_pose.grad is not None:
            self.hand_model.hand_pose.grad.data.zero_()

    def reset_envs(self, mask):
        pass


class MalaStar:
    def __init__(self, hand_model, switch_possibility=0.5, starting_temperature=18, temperature_decay=0.95, annealing_period=30, step_size=0.005, stepsize_period=50, mu=0.98, device="cpu", global_ema=False, clip_grad=False, batch_size=-1):
        """
        Implementation of MALA* optimizer introduced in GraspQP.
        """

        self.hand_model = hand_model
        self.batch_size = batch_size
        self.device = device
        self.switch_possibility = switch_possibility
        self.starting_temperature = torch.tensor(starting_temperature, dtype=torch.float, device=device)
        self.temperature_decay = torch.tensor(temperature_decay, dtype=torch.float, device=device)
        self.annealing_period = torch.tensor(annealing_period, dtype=torch.long, device=device)
        self.step_size = torch.tensor(step_size, dtype=torch.float, device=device)
        self.step_size_period = torch.tensor(stepsize_period, dtype=torch.long, device=device)
        self.mu = torch.tensor(mu, dtype=torch.float, device=device)

        self.step = torch.zeros(hand_model.hand_pose.shape[0], dtype=torch.long, device=device)

        self.old_hand_pose = None
        self.old_contact_point_indices = None
        self.old_global_transformation = None
        self.old_global_rotation = None
        self.old_current_status = None
        self.old_contact_points = None
        self.old_grad_hand_pose = None
        self.old_old_grad_hand_pose = None
        self.clip_grad = clip_grad

        self.ema_grad_hand_pose = torch.zeros(self.hand_model.hand_pose.shape[0], self.hand_model.n_dofs + 9, dtype=torch.float, device=device)

    def try_step(self):
        """
        Try to update translation, rotation, joint angles, and contact point indices

        Returns
        -------
        s: torch.Tensor
            current step size
        """

        s = self.step_size * self.temperature_decay ** torch.div(self.step, self.step_size_period, rounding_mode="floor")
        step_size = torch.zeros(*self.hand_model.hand_pose.shape, dtype=torch.float, device=self.device) + s[..., None]
        if self.clip_grad:
            gradient = self.hand_model.hand_pose.grad.clip(min=-100, max=100)
            gradient[torch.isnan(gradient)] = 0
        else:
            gradient = self.hand_model.hand_pose.grad
        # gradient = self.hand_model.hand_pose.grad.clip(min = -self.clip_grad, max = self.clip_grad)
        # gradient[torch.isnan(gradient)] = 0 # self.batch_size
        mean_over_envs = False
        if mean_over_envs:
            n_grasps, n_dofs = self.hand_model.hand_pose.shape
            n_assets = n_grasps // self.batch_size

            grad = gradient.view(n_assets, self.batch_size, n_dofs).mean(0, keepdim=True).repeat_interleave(n_assets, dim=0).view(n_grasps, n_dofs)
            grad = (grad**2).mean(0)
        else:
            grad = (gradient**2).mean(0)

        self.ema_grad_hand_pose = self.mu * (grad) + (1 - self.mu) * self.ema_grad_hand_pose

        if self.ema_grad_hand_pose.isnan().any():
            self.ema_grad_hand_pose[torch.isnan(self.ema_grad_hand_pose)] = 0
        # self.ema_grad_hand_pose = self.mu * (self.hand_model.hand_pose.grad ** 2).mean(0) + \
        #     (1 - self.mu) * self.ema_grad_hand_pose

        hand_pose = self.hand_model.hand_pose - step_size * gradient / (torch.sqrt(self.ema_grad_hand_pose) + 1e-6)

        if hand_pose.isnan().any():
            nan_mask = hand_pose.isnan().any(dim=-1)
            hand_pose[nan_mask] = 0
            print("Nan in hand pose")
            print(hand_pose)
            print(self.hand_model.hand_pose.grad)
            print(self.ema_grad_hand_pose)
            print(step_size)
            print(self.hand_model.hand_pose)

        batch_size, n_contact = self.hand_model.contact_point_indices.shape
        switch_mask = torch.rand(batch_size, n_contact, dtype=torch.float, device=self.device) < self.switch_possibility
        contact_point_indices = self.hand_model.contact_point_indices.clone()
        contact_point_indices[switch_mask] = torch.randint(self.hand_model.n_contact_candidates, size=[switch_mask.sum()], device=self.device)

        self.old_hand_pose = self.hand_model.hand_pose
        self.old_contact_point_indices = self.hand_model.contact_point_indices
        self.old_global_transformation = self.hand_model.global_translation
        self.old_global_rotation = self.hand_model.global_rotation
        self.old_current_status = self.hand_model.current_status
        self.old_contact_points = self.hand_model.contact_points

        self.old_grad_hand_pose = self.hand_model.hand_pose.grad
        self.hand_model.set_parameters(hand_pose, contact_point_indices)
        # if wandb.run is not None:
        #     wandb.log({"gradients": gradient}, commit=False)
        #     wandb.log({"moving_avg_grad": self.ema_grad_hand_pose}, commit=False)
        self.step += 1

        return s

    def reset_envs(self, mask):
        self.step[mask] = 0
        self.ema_grad_hand_pose[mask] = 0

        self.old_hand_pose[mask] = self.hand_model.hand_pose[mask]
        self.old_contact_point_indices[mask] = self.hand_model.contact_point_indices[mask]
        self.old_global_transformation[mask] = self.hand_model.global_translation[mask]
        self.old_global_rotation[mask] = self.hand_model.global_rotation[mask]
        self.old_contact_points[mask] = self.hand_model.contact_points[mask]
        self.old_grad_hand_pose[mask] = 0 * self.old_grad_hand_pose[mask]

        if self.old_old_grad_hand_pose is not None:
            self.old_old_grad_hand_pose[mask] = 0 * self.old_old_grad_hand_pose[mask]

    def accept_step(self, energy, new_energy, reset_mask=None, z_score=None, z_score_threshold=2.0):
        """
        Accept / reject updates using annealing

        Returns
        -------
        accept: (N,) torch.BoolTensor
        temperature: torch.Tensor
            current temperature
        """
        batch_size = energy.shape[0]

        temperature = self.starting_temperature * self.temperature_decay ** torch.div(self.step, self.annealing_period, rounding_mode="floor")

        alpha = torch.rand(batch_size, dtype=torch.float, device=self.device)

        if z_score is not None:
            proba = normal.cdf(z_score.detach())  # .clip(min = 0.3) ?
            temperature = temperature * (1 + proba)

        # new_energy is smaller than energy -> Energy decreases (good) -> acceptance rate is 1
        # new energy is bigger (bad case) -> Acceptance rate is prop to exp(-energy_increase / temperature)
        accept = alpha < torch.exp((energy - new_energy) / temperature)

        if reset_mask is not None:
            accept[reset_mask] = True

        # accept = energy > new_energy

        # wandb.log({"accept_rate": accept.float().mean()}, commit=False)
        # wandb.log({"temperature": temperature.mean()}, commit=False)
        # wandb.log({"energy_change": energy - new_energy}, commit=False)
        # wandb.log({"good_energy_ratio": (energy > new_energy).float().mean()}, commit=False)

        with torch.no_grad():
            reject = ~accept
            self.hand_model.hand_pose[reject] = self.old_hand_pose[reject]
            self.hand_model.contact_point_indices[reject] = self.old_contact_point_indices[reject]
            self.hand_model.global_translation[reject] = self.old_global_transformation[reject]
            self.hand_model.global_rotation[reject] = self.old_global_rotation[reject]
            self.hand_model.current_status = self.hand_model.fk(self.hand_model.hand_pose[:, 9:])
            self.hand_model.contact_points[reject] = self.old_contact_points[reject]

            if self.old_grad_hand_pose is not None and self.hand_model.hand_pose.grad is not None:
                if self.old_old_grad_hand_pose is not None:
                    self.old_grad_hand_pose[reject] = self.old_old_grad_hand_pose[reject]

                self.hand_model.hand_pose.grad[reject] = self.old_grad_hand_pose[reject]

        return accept, temperature

    def zero_grad(self):
        """
        Sets the gradients of translation, rotation, and joint angles to zero
        """
        if self.hand_model.hand_pose.grad is not None:
            self.hand_model.hand_pose.grad.data.zero_()
