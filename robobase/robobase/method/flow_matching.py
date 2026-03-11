from functools import partial

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from diffusers.training_utils import EMAModel

from robobase.method.bc import BC
from robobase.method.utils import (
    extract_from_spec,
    extract_many_from_spec,
    flatten_time_dim_into_channel_dim,
    stack_tensor_dictionary,
)
from robobase.models.diffusion_models import replace_bn_with_gn
from robobase.models.fully_connected import FullyConnectedModule


class Actor(nn.Module):
    def __init__(
        self,
        action_space: spaces.Box,
        actor_model: FullyConnectedModule,
        num_inference_steps: int,
        use_ema_inference: bool = True,
        clip_sample: bool = True,
    ):
        super().__init__()
        assert len(action_space.shape) == 2
        self.action_space = action_space
        self.actor = actor_model
        self.num_inference_steps = num_inference_steps
        self.use_ema_inference = use_ema_inference
        self.clip_sample = clip_sample
        self.sequence_length = action_space.shape[0]
        self.action_dim = action_space.shape[1]
        self.ema = EMAModel(
            parameters=self.actor.parameters(),
            power=0.75,
        )
        self.ema_actor = copy.deepcopy(self.actor)
        self.register_buffer(
            "action_low",
            torch.as_tensor(action_space.low, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "action_high",
            torch.as_tensor(action_space.high, dtype=torch.float32),
            persistent=False,
        )

    @property
    def preferred_optimiser(self) -> callable:
        return getattr(
            self.actor,
            "preferred_optimiser",
            partial(torch.optim.Adam, self.parameters()),
        )

    def _combine(self, low_dim_obs, fused_view_feats):
        flat_feats = []
        if low_dim_obs is not None:
            flat_feats.append(low_dim_obs)
        if fused_view_feats is not None:
            flat_feats.append(fused_view_feats)
        obs_features = torch.cat(flat_feats, dim=-1)
        return obs_features

    def _clip_to_action_bounds(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.clip_sample:
            return actions
        return torch.clamp(actions, min=self.action_low, max=self.action_high)

    def forward(
        self, low_dim_obs, fused_view_feats, action
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_features = self._combine(low_dim_obs, fused_view_feats)
        b = obs_features.shape[0]

        # Sample t ~ Uniform(0, 1) during training, as in standard flow matching.
        # x0: source noise sample from N(0, I), x1: expert action sequence.
        source_noise = torch.randn((b,) + self.action_space.shape, device=obs_features.device)
        timesteps = torch.rand((b,), device=obs_features.device)
        t = timesteps.view(b, 1, 1)

        # Linear interpolation path x_t = (1 - t) * x0 + t * x1.
        flow_state = (1.0 - t) * source_noise + t * action
        target_velocity = action - source_noise

        net_ins = {
            "actions": flow_state,
            "features": obs_features,
            "timestep": timesteps,
        }
        velocity_pred = self.actor(net_ins)
        return velocity_pred, target_velocity

    def infer(self, low_dim_obs, fused_view_feats) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve probability-flow ODE from source noise to target actions with Euler steps.
        """
        obs_features = self._combine(low_dim_obs, fused_view_feats)
        with torch.no_grad():
            if self.use_ema_inference:
                actor = self.ema_actor
                self.ema.copy_to(actor.parameters())
            else:
                actor = self.actor

            b = obs_features.shape[0]
            flow_state = torch.randn(
                (b, self.sequence_length, self.action_dim), device=obs_features.device
            )

            velocity_pred = None
            schedule = torch.linspace(
                0.0,
                1.0,
                self.num_inference_steps + 1,
                device=obs_features.device,
                dtype=flow_state.dtype,
            )
            for i in range(self.num_inference_steps):
                timestep = schedule[i].expand(b)
                dt = schedule[i + 1] - schedule[i]
                net_ins = {
                    "actions": flow_state,
                    "features": obs_features,
                    "timestep": timestep,
                }
                velocity_pred = actor(net_ins)
                flow_state = flow_state + dt * velocity_pred
                flow_state = self._clip_to_action_bounds(flow_state)

        if velocity_pred is None:
            velocity_pred = torch.zeros_like(flow_state)
        return velocity_pred, flow_state


class FlowMatching(BC):
    def __init__(
        self,
        num_inference_steps: int = 2,
        use_ema_inference: bool = True,
        clip_sample: bool = True,
        *args,
        **kwargs,
    ):
        if not kwargs["frame_stack_on_channel"]:
            raise NotImplementedError(
                "frame_stack_on_channel must be true for flow matching policies."
            )
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1.")
        self.num_inference_steps = num_inference_steps
        self.use_ema_inference = use_ema_inference
        self.clip_sample = clip_sample
        super().__init__(*args, **kwargs)

    def build_actor(self):
        # Keep the same UNet constraints as diffusion policy.
        if self.action_space.shape[-2] % 4 != 0:
            raise ValueError(
                "Action sequence length has to be a multiple of 4 for flow matching model."
            )
        self.actor_model = self.actor_model(
            input_shapes=self.get_fully_connected_inputs(),
            output_shape=self.action_space.shape[-1],
        ).to(self.device)
        self.actor = Actor(
            self.action_space,
            self.actor_model,
            self.num_inference_steps,
            self.use_ema_inference,
            self.clip_sample,
        ).to(self.device)
        self.actor_opt = (self.actor.preferred_optimiser)(lr=self.lr)

    def build_encoder(self):
        super().build_encoder()
        rgb_spaces = extract_many_from_spec(
            self.observation_space, r"rgb.*", missing_ok=True
        )
        if len(rgb_spaces) > 0:
            # Match diffusion policy behavior for pixel stability.
            self.encoder = replace_bn_with_gn(self.encoder)
            self.encoder.to(self.device)
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)

    def get_fully_connected_inputs(self) -> dict[str, tuple]:
        input_shapes = super().get_fully_connected_inputs()
        input_shapes["actions"] = self.action_space.shape[-1:]
        return input_shapes

    def _act(self, observations: dict[str, torch.Tensor], eval_mode: bool):
        low_dim_obs = fused_rgb_feats = None
        if self.low_dim_size > 0:
            low_dim_obs = flatten_time_dim_into_channel_dim(
                extract_from_spec(observations, "low_dim_state")
            )
        if self.use_pixels:
            rgb_obs = flatten_time_dim_into_channel_dim(
                stack_tensor_dictionary(
                    extract_many_from_spec(observations, r"rgb.*"), 1
                ),
                has_view_axis=True,
            )
            with torch.no_grad():
                multi_view_rgb_feats = self.encoder(rgb_obs.float())
                fused_rgb_feats = self.view_fusion(multi_view_rgb_feats)
        _, flow_state = self.actor.infer(low_dim_obs, fused_rgb_feats)
        return flow_state.detach()

    def update_actor(
        self, low_dim_obs, fused_view_feats, action, loss_coeff, action_mask=None
    ):
        metrics = dict()
        velocity_pred, target_velocity = self.actor(low_dim_obs, fused_view_feats, action)
        step_mse = F.mse_loss(velocity_pred, target_velocity, reduction="none").mean(-1)
        if action_mask is not None:
            action_mask = action_mask.float()
            valid_steps = action_mask.sum(-1).clamp_min(1.0)
            mse_loss = ((step_mse * action_mask).sum(-1) / valid_steps).unsqueeze(-1)
        else:
            mse_loss = step_mse.mean(-1, keepdims=True)
        actor_loss = (mse_loss * loss_coeff.unsqueeze(1)).mean()

        new_pri = torch.sqrt(mse_loss + 1e-10)
        self._new_priority = (new_pri / torch.max(new_pri)).cpu().detach().numpy()

        if self.use_pixels and self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.actor_grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_opt.step()
        if self.use_pixels and self.encoder is not None:
            if self.actor_grad_clip:
                nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), self.critic_grad_clip
                )
            self.encoder_opt.step()
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.actor.ema.step(self.actor)

        if self.logging:
            metrics["actor_loss"] = actor_loss.item()
        return metrics
