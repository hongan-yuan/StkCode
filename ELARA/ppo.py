from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .device import resolve_torch_device_name
from .model import ELARANetwork, F, require_torch, state_to_tensors, torch
from .state import ServingSelectionState


@dataclass
class PPOTransition:
    state: ServingSelectionState
    action: int
    old_log_prob: float
    value: float
    reward: float
    done: bool


class PPOAgent:
    def __init__(self, config, device: str = "auto"):
        require_torch()
        device = resolve_torch_device_name(device, torch)
        self.device = torch.device(device)
        self.config = config
        self.network = ELARANetwork(
            num_services=config.num_services,
            hidden_dim=config.hidden_dim,
            graph_layers=config.graph_layers,
            attention_heads=config.attention_heads,
            service_embedding_dim=config.service_embedding_dim,
            max_future_horizon=max(8, config.future_topology_horizon),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.ppo_learning_rate)
        self.buffer: list[PPOTransition] = []

    def act(self, state: ServingSelectionState, deterministic: bool = False):
        observation = state_to_tensors(state, self.device)
        with torch.no_grad():
            logits, value = self.network(observation)
            distribution = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits) if deterministic else distribution.sample()
            log_prob = distribution.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def remember(self, transition: PPOTransition) -> None:
        self.buffer.append(transition)

    def _bootstrap_value(self, next_state: ServingSelectionState | None) -> float:
        if next_state is None:
            return 0.0
        observation = state_to_tensors(next_state, self.device)
        with torch.no_grad():
            _, value = self.network(observation)
        return float(value.item())

    def update(self, next_state: ServingSelectionState | None = None) -> dict[str, float]:
        if not self.buffer:
            return {}
        bootstrap = self._bootstrap_value(next_state)
        advantages = np.zeros(len(self.buffer), dtype=np.float32)
        returns = np.zeros(len(self.buffer), dtype=np.float32)
        gae = 0.0
        next_value = bootstrap
        for index in reversed(range(len(self.buffer))):
            transition = self.buffer[index]
            continuation = 0.0 if transition.done else 1.0
            delta = transition.reward + self.config.ppo_gamma * next_value * continuation - transition.value
            gae = delta + self.config.ppo_gamma * self.config.ppo_gae_lambda * continuation * gae
            advantages[index] = gae
            returns[index] = gae + transition.value
            next_value = transition.value
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

        indices = list(range(len(self.buffer)))
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "samples": 0}
        minibatch_size = min(self.config.ppo_minibatch_size, len(indices))
        for _ in range(self.config.ppo_epochs):
            random.shuffle(indices)
            for start in range(0, len(indices), minibatch_size):
                batch_indices = indices[start:start + minibatch_size]
                policy_losses = []
                value_losses = []
                entropies = []
                for index in batch_indices:
                    transition = self.buffer[index]
                    observation = state_to_tensors(transition.state, self.device)
                    logits, value = self.network(observation)
                    distribution = torch.distributions.Categorical(logits=logits)
                    action = torch.as_tensor(
                        transition.action, dtype=torch.long, device=self.device
                    )
                    new_log_prob = distribution.log_prob(action)
                    ratio = torch.exp(new_log_prob - transition.old_log_prob)
                    advantage = torch.as_tensor(
                        advantages[index], dtype=torch.float32, device=self.device
                    )
                    unclipped = ratio * advantage
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.config.ppo_clip_epsilon,
                        1.0 + self.config.ppo_clip_epsilon,
                    ) * advantage
                    policy_losses.append(-torch.min(unclipped, clipped))
                    target = torch.as_tensor(
                        returns[index], dtype=torch.float32, device=self.device
                    )
                    value_losses.append(F.mse_loss(value, target))
                    entropies.append(distribution.entropy())

                policy_loss = torch.stack(policy_losses).mean()
                value_loss = torch.stack(value_losses).mean()
                entropy = torch.stack(entropies).mean()
                loss = policy_loss + self.config.ppo_value_coef * value_loss
                loss = loss - self.config.ppo_entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                sample_count = len(batch_indices)
                totals["policy_loss"] += float(policy_loss.item()) * sample_count
                totals["value_loss"] += float(value_loss.item()) * sample_count
                totals["entropy"] += float(entropy.item()) * sample_count
                totals["samples"] += sample_count
        count = max(1, totals.pop("samples"))
        self.buffer.clear()
        return {key: value / count for key, value in totals.items()}

    def save(self, path, control_state: dict | None = None) -> None:
        torch.save(
            {
                "model": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "control_state": control_state,
            },
            path,
        )

    def load(self, path, load_optimizer: bool = False) -> dict | None:
        payload = torch.load(path, map_location=self.device)
        self.network.load_state_dict(payload["model"])
        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        return payload.get("control_state")
