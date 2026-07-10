# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT
#
# Portions derived from DexGraspNet (https://github.com/PKU-EPIC/DexGraspNet),
# MIT License, Copyright (c) 2023 Jialiang Zhang, Ruicheng Wang.

"""Simulated-annealing optimizers for grasp synthesis.

Provides two Metropolis-style optimizers that jointly update the continuous
hand pose (translation, 6D rotation, joint angles) and the discrete
contact-point indices:

* :class:`AnnealingDexGraspNet` -- the original DexGraspNet annealing sampler
  with a single global temperature and step size.
* :class:`MalaStar` -- the MALA* sampler introduced in GraspQP, which keeps a
  per-grasp step counter, temperature and gradient EMA so grasps anneal
  independently.

Both follow the same ``try_step`` (propose) / ``accept_step`` (Metropolis
accept-or-revert) / ``zero_grad`` loop and operate in place on a shared
:class:`~graspqp.core.hand_model.HandModel`.

Based on Dexgraspnet: https://pku-epic.github.io/DexGraspNet/
"""

import torch
from torch.distributions import Normal

normal = Normal(0, 1)


class AnnealingDexGraspNet:
    """DexGraspNet simulated-annealing optimizer with a single global schedule.

    Proposes new grasps by taking an RMSProp-style gradient step on the
    continuous pose and randomly resampling a fraction of the discrete contact
    indices, then accepts or rejects each grasp with the Metropolis criterion.
    Temperature and step size decay together on fixed periods shared across the
    whole batch.
    """

    def __init__(
        self,
        hand_model,
        switch_possibility=0.5,
        starting_temperature=18,
        temperature_decay=0.95,
        annealing_period=30,
        step_size=0.005,
        stepsize_period=50,
        mu=0.98,
        device="cpu",
        **kwargs
    ):
        """Create the annealing optimizer.

        Args:
            hand_model (HandModel): Hand whose parameters are optimized in place.
            switch_possibility (float): Per-step probability of resampling each
                contact-point index.
            starting_temperature (float): Initial Metropolis temperature.
            temperature_decay (float): Multiplicative decay applied to both the
                temperature and the step size at each period boundary.
            annealing_period (int): Number of steps between temperature decays.
            step_size (float): Base gradient step size (meters/radians scale).
            stepsize_period (int): Number of steps between step-size decays.
            mu (float): RMSProp gradient-EMA coefficient (``1 - decay_rate``).
            device (str | torch.device): Device for the optimizer state tensors.
            **kwargs: Ignored; accepted for interface compatibility.
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
        """Propose a new grasp and write it into the hand model.

        Takes one RMSProp gradient step on the continuous pose and resamples a
        random subset of contact indices. The previous state is cached so
        :meth:`accept_step` can revert rejected grasps.

        Returns:
            torch.Tensor: The current (decayed) step size ``s``.
        """

        s = self.step_size * self.temperature_decay ** torch.div(self.step, self.step_size_period, rounding_mode="floor")
        step_size = torch.zeros(*self.hand_model.hand_pose.shape, dtype=torch.float, device=self.device) + s

        self.ema_grad_hand_pose = (
            self.mu * (self.hand_model.hand_pose.grad**2).mean(0) + (1 - self.mu) * self.ema_grad_hand_pose
        )

        hand_pose = self.hand_model.hand_pose - step_size * self.hand_model.hand_pose.grad / (
            torch.sqrt(self.ema_grad_hand_pose) + 1e-6
        )
        batch_size, n_contact = self.hand_model.contact_point_indices.shape
        switch_mask = torch.rand(batch_size, n_contact, dtype=torch.float, device=self.device) < self.switch_possibility
        contact_point_indices = self.hand_model.contact_point_indices.clone()
        contact_point_indices[switch_mask] = torch.randint(
            self.hand_model.n_contact_candidates, size=[switch_mask.sum()], device=self.device
        )

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
        """Metropolis accept/reject the proposed grasps, reverting rejects.

        Grasps are accepted with probability ``min(1, exp((energy - new_energy)
        / temperature))``; rejected grasps have their pose, contact indices,
        transforms, contact points and gradient restored to the pre-proposal
        state cached by :meth:`try_step`.

        Args:
            energy (torch.Tensor): ``(N,)`` energy of the previous (current)
                grasps.
            new_energy (torch.Tensor): ``(N,)`` energy of the proposed grasps.
            *args: Ignored; accepted for interface compatibility.
            **kwargs: Ignored; accepted for interface compatibility.

        Returns:
            tuple: ``(accept, temperature)`` where ``accept`` is an ``(N,)``
            bool tensor and ``temperature`` is the current annealing temperature.
        """

        batch_size = energy.shape[0]
        temperature = self.starting_temperature * self.temperature_decay ** torch.div(
            self.step, self.annealing_period, rounding_mode="floor"
        )

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
        """Zero the accumulated gradient on the hand pose (translation, rotation and joint angles)."""
        if self.hand_model.hand_pose.grad is not None:
            self.hand_model.hand_pose.grad.data.zero_()

    def reset_envs(self, mask):
        pass


class MalaStar:
    def __init__(
        self,
        hand_model,
        switch_possibility=0.5,
        starting_temperature=18,
        temperature_decay=0.95,
        annealing_period=30,
        step_size=0.005,
        stepsize_period=50,
        mu=0.98,
        device="cpu",
        global_ema=False,
        clip_grad=False,
        batch_size=-1,
    ):
        """MALA* simulated-annealing optimizer introduced in GraspQP.

        Unlike :class:`AnnealingDexGraspNet`, MALA* keeps the annealing state
        (step counter ``self.step``, temperature and step-size schedule) as
        per-grasp tensors so each grasp anneals on its own schedule and can be
        reset independently via :meth:`reset_envs`. The proposal is the same
        RMSProp-style pose step plus random contact resampling, with optional
        gradient clipping and NaN-safe handling.

        Args:
            hand_model (HandModel): Hand whose parameters are optimized in place.
            switch_possibility (float): Per-step probability of resampling each
                contact-point index.
            starting_temperature (float): Initial Metropolis temperature.
            temperature_decay (float): Multiplicative decay for the temperature
                and step size at each period boundary.
            annealing_period (int): Number of steps between temperature decays.
            step_size (float): Base gradient step size (meters/radians scale).
            stepsize_period (int): Number of steps between step-size decays.
            mu (float): RMSProp gradient-EMA coefficient (``1 - decay_rate``).
            device (str | torch.device): Device for the optimizer state tensors.
            global_ema (bool): Accepted for interface compatibility.
            clip_grad (bool): If True, clip the pose gradient to ``[-100, 100]``
                and zero any NaNs before stepping.
            batch_size (int): Number of grasps per object; used when averaging
                gradients across environments.
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

        self.ema_grad_hand_pose = torch.zeros(
            self.hand_model.hand_pose.shape[0], self.hand_model.n_dofs + 9, dtype=torch.float, device=device
        )

    def try_step(self):
        """Propose a new grasp per environment and write it into the hand model.

        Applies a per-grasp RMSProp step (optionally clipped/NaN-guarded) to the
        continuous pose and resamples a random subset of contact indices, caching
        the previous state for :meth:`accept_step`.

        Returns:
            torch.Tensor: The current per-grasp (decayed) step size ``s``.
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

            grad = (
                gradient.view(n_assets, self.batch_size, n_dofs)
                .mean(0, keepdim=True)
                .repeat_interleave(n_assets, dim=0)
                .view(n_grasps, n_dofs)
            )
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
            # NaN proposals (degenerate gradients on a few grasps) are zeroed here and then
            # rejected by accept_step, reverting to the prior pose. Tensor-dump prints removed.
            nan_mask = hand_pose.isnan().any(dim=-1)
            hand_pose[nan_mask] = 0

        batch_size, n_contact = self.hand_model.contact_point_indices.shape
        switch_mask = torch.rand(batch_size, n_contact, dtype=torch.float, device=self.device) < self.switch_possibility
        contact_point_indices = self.hand_model.contact_point_indices.clone()
        contact_point_indices[switch_mask] = torch.randint(
            self.hand_model.n_contact_candidates, size=[switch_mask.sum()], device=self.device
        )

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
        """Reset the per-grasp annealing state for the masked environments.

        Zeros the step counter and gradient EMA and re-syncs the cached
        pre-proposal state to the current hand state for every grasp selected by
        ``mask`` (used when an environment is re-initialized with a fresh grasp).

        Args:
            mask (torch.Tensor): Boolean ``(N,)`` mask selecting environments to
                reset.
        """
        # hand_pose is now a leaf requiring grad, so these in-place syncs must run under
        # no_grad (an in-place op on a leaf-requiring-grad tensor is otherwise disallowed).
        with torch.no_grad():
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
        """Metropolis accept/reject the proposed grasps, reverting rejects.

        Grasps are accepted with probability ``min(1, exp((energy - new_energy)
        / temperature))`` using the per-grasp temperature. Rejected grasps are
        restored to the state cached by :meth:`try_step`.

        Args:
            energy (torch.Tensor): ``(N,)`` energy of the previous grasps.
            new_energy (torch.Tensor): ``(N,)`` energy of the proposed grasps.
            reset_mask (torch.Tensor | None): Optional ``(N,)`` bool mask of
                grasps to force-accept (e.g. freshly reset environments).
            z_score (torch.Tensor | None): Optional per-grasp z-score; when
                given, the effective temperature is scaled by ``1 + Phi(z)``
                (standard-normal CDF) to raise acceptance for uncertain grasps.
            z_score_threshold (float): Kept for API compatibility.

        Returns:
            tuple: ``(accept, temperature)`` where ``accept`` is an ``(N,)`` bool
            tensor and ``temperature`` is the per-grasp annealing temperature.
        """
        batch_size = energy.shape[0]

        temperature = self.starting_temperature * self.temperature_decay ** torch.div(
            self.step, self.annealing_period, rounding_mode="floor"
        )

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
        """Zero the accumulated gradient on the hand pose (translation, rotation and joint angles)."""
        if self.hand_model.hand_pose.grad is not None:
            self.hand_model.hand_pose.grad.data.zero_()
