from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .device import resolve_torch_device_name
from .model import (
    ELARANetwork,
    F,
    batch_states_to_tensors,
    require_torch,
    state_to_tensors,
    torch,
)
from .state import ServingSelectionState


@dataclass
class PPOTransition:
    state: ServingSelectionState
    action: int
    old_log_prob: float
    value: float
    reward: float
    done: bool
    collection_slot: int = 0
    update_uses: int = 0


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
        result = self.act_batch([state], deterministic=deterministic)
        return result[0]

    def act_batch(
        self,
        states: list[ServingSelectionState],
        deterministic: bool = False,
    ) -> list[tuple[int, float, float]]:
        observation = batch_states_to_tensors(states, self.device)
        with torch.no_grad():
            logits, values = self.network.forward_batch(observation)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = (
                torch.argmax(logits, dim=-1)
                if deterministic
                else distribution.sample()
            )
            log_probs = distribution.log_prob(actions)
            packed = torch.stack(
                (actions.to(values.dtype), log_probs, values), dim=-1
            ).cpu().numpy()
        return [
            (int(row[0]), float(row[1]), float(row[2]))
            for row in packed
        ]

    def remember(self, transition: PPOTransition) -> None:
        self.buffer.append(transition)

    def clear_transactions(self) -> None:
        self.buffer.clear()

    def _eligible_transitions(
        self, current_slot: int | None
    ) -> list[PPOTransition]:
        if current_slot is None:
            return list(self.buffer)
        oldest_slot = current_slot - self.config.ppo_transaction_history_slots
        self.buffer = [
            transition
            for transition in self.buffer
            if transition.collection_slot >= oldest_slot
            and transition.update_uses < self.config.ppo_transaction_max_reuse
        ]
        return list(self.buffer)

    def eligible_transition_count(self, current_slot: int | None) -> int:
        return len(self._eligible_transitions(current_slot))

    def _bootstrap_value(self, next_state: ServingSelectionState | None) -> float:
        if next_state is None:
            return 0.0
        observation = state_to_tensors(next_state, self.device)
        with torch.no_grad():
            _, value = self.network(observation)
        return float(value.item())

    def update(
        self,
        next_state: ServingSelectionState | None = None,
        *,
        current_slot: int | None = None,
    ) -> dict[str, float]:
        transitions = self._eligible_transitions(current_slot)
        if not transitions:
            return {}
        bootstrap = self._bootstrap_value(next_state)
        advantages = np.zeros(len(transitions), dtype=np.float32)
        returns = np.zeros(len(transitions), dtype=np.float32)
        gae = 0.0
        next_value = bootstrap
        for index in reversed(range(len(transitions))):
            transition = transitions[index]
            continuation = 0.0 if transition.done else 1.0
            delta = transition.reward + self.config.ppo_gamma * next_value * continuation - transition.value
            gae = delta + self.config.ppo_gamma * self.config.ppo_gae_lambda * continuation * gae
            advantages[index] = gae
            returns[index] = gae + transition.value
            next_value = transition.value
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

        indices = list(range(len(transitions)))
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "samples": 0}
        minibatch_size = min(self.config.ppo_minibatch_size, len(indices))
        for _ in range(self.config.ppo_epochs):
            random.shuffle(indices)
            for start in range(0, len(indices), minibatch_size):
                batch_indices = indices[start:start + minibatch_size]
                batch = [transitions[index] for index in batch_indices]
                observation = batch_states_to_tensors(
                    [transition.state for transition in batch], self.device
                )
                logits, values = self.network.forward_batch(observation)
                distribution = torch.distributions.Categorical(logits=logits)
                actions = torch.as_tensor(
                    [transition.action for transition in batch],
                    dtype=torch.long,
                    device=self.device,
                )
                new_log_probs = distribution.log_prob(actions)
                old_log_probs = torch.as_tensor(
                    [transition.old_log_prob for transition in batch],
                    dtype=torch.float32,
                    device=self.device,
                )
                batch_advantages = torch.as_tensor(
                    advantages[batch_indices],
                    dtype=torch.float32,
                    device=self.device,
                )
                ratio = torch.exp(new_log_probs - old_log_probs)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.ppo_clip_epsilon,
                    1.0 + self.config.ppo_clip_epsilon,
                ) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                targets = torch.as_tensor(
                    returns[batch_indices],
                    dtype=torch.float32,
                    device=self.device,
                )
                value_loss = F.mse_loss(values, targets)
                entropy = distribution.entropy().mean()
                loss = policy_loss + self.config.ppo_value_coef * value_loss
                loss = loss - self.config.ppo_entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                sample_count = len(batch_indices)
                totals["policy_loss"] += float(policy_loss.item()) * sample_count
                totals["value_loss"] += float(value_loss.item()) * sample_count
                totals["entropy"] += float(entropy.item()) * sample_count
                totals["samples"] += sample_count
        count = max(1, totals.pop("samples"))
        if current_slot is None:
            self.buffer.clear()
        else:
            for transition in transitions:
                transition.update_uses += 1
            self._eligible_transitions(current_slot)
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

    def load_control_state(self, path) -> dict | None:
        """Load only the saved environment state without changing the policy."""
        payload = torch.load(path, map_location=self.device)
        return payload.get("control_state")
