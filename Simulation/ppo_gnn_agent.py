from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from .chain_encoder import encode_service_chain
from .config import SimulationConfig
from .execution_agent import CandidateDecision, ServiceExecutionAgent
from .graph_encoder import encode_satellite_graph
from .request import SFCRequest
from .routing import route_data
from .service import compute_service_execution
from .topology import slot_from_time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None

    class _NoTorchModule:
        pass

    class _NoTorchNN:
        Module = _NoTorchModule

    nn = _NoTorchNN()
    F = None


@dataclass
class PPOTransition:
    log_prob: object
    value: object
    reward: float
    entropy: object
    done: bool


# if torch is not None:

class GraphMessagePassing(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(layers)
        )

    def forward(self, node_features, adjacency):
        h = F.relu(self.input_proj(node_features))
        for layer in self.layers:
            messages = []
            for node_idx, neighbors in enumerate(adjacency):
                if neighbors:
                    neighbor_tensor = h[neighbors].mean(dim=0)
                else:
                    neighbor_tensor = torch.zeros_like(h[node_idx])
                messages.append(neighbor_tensor)
            msg = torch.stack(messages, dim=0)
            h = F.relu(layer(torch.cat([h, msg], dim=-1)))
        return h


class CandidateMaskedPolicy(nn.Module):
    def __init__(
            self,
            node_feature_dim: int,
            chain_feature_dim: int,
            hidden_dim: int = 64,
    ):
        super().__init__()
        self.gnn = GraphMessagePassing(node_feature_dim, hidden_dim)
        self.chain_proj = nn.Linear(chain_feature_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
            self,
            node_features,
            adjacency,
            chain_features,
            current_index: int,
            destination_index: int,
            candidate_indices,
            candidate_extra_features,
    ):
        h = self.gnn(node_features, adjacency)
        z_chain = F.relu(self.chain_proj(chain_features))
        current_h = h[current_index]
        dest_h = h[destination_index]
        logits = []
        for idx, extra in zip(candidate_indices, candidate_extra_features):
            candidate_h = h[idx]
            score_input = torch.cat(
                [candidate_h, current_h, dest_h, z_chain, extra], dim=-1
            )
            logits.append(self.scorer(score_input).squeeze(-1))
        logits = torch.stack(logits)
        global_state = torch.cat([current_h, dest_h, z_chain], dim=-1)
        value = self.value_head(global_state).squeeze(-1)
        return logits, value


class PPOGNNExecutionAgent(ServiceExecutionAgent):
    """Fast-layer PPO + GNN encoder + candidate masked softmax agent.

    When PyTorch is available, this class uses a real message-passing network
    and masked candidate softmax. In environments without PyTorch it falls back
    to a deterministic GNN-style message passing scorer so the simulator remains
    executable; training then becomes unavailable but the runtime interface is
    unchanged.
    """

    def __init__(
            self,
            config: SimulationConfig,
            hidden_dim: int = 64,
            train_mode: bool = False,
            device: str | None = None,
    ):
        super().__init__(config)
        self.hidden_dim = hidden_dim
        self.train_mode = train_mode
        self.transitions: list[PPOTransition] = []
        self.device = None
        self.policy = None
        self.optimizer = None
        if torch is not None:
            if device is None or device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            self.policy = CandidateMaskedPolicy(9, 6, hidden_dim).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=config.ppo_learning_rate
            )

    @property
    def training_available(self) -> bool:
        return torch is not None and self.policy is not None

    def select_replica(
            self,
            request: SFCRequest,
            service_index: int,
            current_node: int,
            current_time: float,
            data_gb: float,
            context: dict,
    ) -> CandidateDecision:
        if self.training_available:
            return self._select_with_torch_policy(
                request, service_index, current_node, current_time, data_gb, context
            )
        return self._select_with_fallback_gnn(
            request, service_index, current_node, current_time, data_gb, context
        )

    def _select_with_torch_policy(
            self,
            request: SFCRequest,
            service_index: int,
            current_node: int,
            current_time: float,
            data_gb: float,
            context: dict,
    ) -> CandidateDecision:
        service_id = request.services[service_index]
        candidates = context["microservices"][service_id].replicas
        abs_slot, slot_mod = slot_from_time(
            current_time, context["slot_duration"], context["slot_count"]
        )
        graph = context["snapshots"][slot_mod]
        node_ids = sorted(graph.nodes)
        node_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        node_features = encode_satellite_graph(
            graph, context, current_node, request.destination_node, service_id
        )
        node_tensor = torch.tensor(
            [node_features[node_id] for node_id in node_ids],
            dtype=torch.float32,
            device=self.device,
        )
        adjacency = [
            [node_to_index[nbr] for nbr in graph.neighbors(node_id) if nbr in node_to_index]
            for node_id in node_ids
        ]
        chain_tensor = torch.tensor(
            encode_service_chain(
                request, service_index, context["microservices"], self.config
            ),
            dtype=torch.float32,
            device=self.device,
        )

        candidate_records = self._candidate_records(
            request, service_index, candidates, current_node, current_time, data_gb, service_id, context
        )
        # print(f"candidate_records: {candidate_records}")
        reachable = [record for record in candidate_records if record["reachable"]]
        if not reachable:
            return CandidateDecision(service_id, None, math.inf, None, None, candidate_records)

        candidate_indices = [node_to_index[record["node_id"]] for record in reachable]

        # print(f"candidate_indices: {candidate_indices}")

        candidate_extra = torch.tensor(
            [record["extra_features"] for record in reachable],
            dtype=torch.float32,
            device=self.device,
        )
        logits, value = self.policy(
            node_tensor,
            adjacency,
            chain_tensor,
            node_to_index[current_node],
            node_to_index[request.destination_node],
            candidate_indices,
            candidate_extra,
        )
        probs = torch.softmax(logits, dim=0)
        if self.train_mode:
            dist = torch.distributions.Categorical(probs=probs)
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor)
            entropy = dist.entropy()
            action_idx = int(action_tensor.item())
        else:
            action_idx = int(torch.argmax(probs).item())
            log_prob = torch.log(probs[action_idx].clamp_min(1.0e-12))
            entropy = -(probs * torch.log(probs.clamp_min(1.0e-12))).sum()
        selected = reachable[action_idx]
        selected["policy_probability"] = float(probs[action_idx].detach().cpu())
        selected["value_estimate"] = float(value.detach().cpu())

        # print(f"CandidateDecision: \nservice_id={service_id}, \nselected_node={selected['node_id']}, "
        #       f"")
        # sys.exit(1)
        return CandidateDecision(
            service_id=service_id,
            selected_node=selected["node_id"],
            score=-selected["policy_probability"],
            route_estimate=selected["route"],
            compute_estimate=selected["compute"],
            candidate_scores=candidate_records,
            metadata={"log_prob": log_prob, "value": value, "entropy": entropy},
        )

    def _select_with_fallback_gnn(
            self,
            request: SFCRequest,
            service_index: int,
            current_node: int,
            current_time: float,
            data_gb: float,
            context: dict,
    ) -> CandidateDecision:
        service_id = request.services[service_index]
        candidates = context["microservices"][service_id].replicas
        records = self._candidate_records(
            request, service_index, candidates, current_node, current_time, data_gb, service_id, context
        )
        reachable = [record for record in records if record["reachable"]]
        if not reachable:
            return CandidateDecision(service_id, None, math.inf, None, None, records)

        # Candidate masked softmax over negative normalized costs.
        logits = [-record["score"] for record in reachable]
        max_logit = max(logits)
        exp_values = [math.exp(logit - max_logit) for logit in logits]
        denom = sum(exp_values)
        for record, exp_value in zip(reachable, exp_values):
            record["policy_probability"] = exp_value / denom if denom > 0 else 0.0
        selected = max(reachable, key=lambda item: item["policy_probability"])
        return CandidateDecision(
            service_id=service_id,
            selected_node=selected["node_id"],
            score=selected["score"],
            route_estimate=selected["route"],
            compute_estimate=selected["compute"],
            candidate_scores=records,
        )

    def _candidate_records(
            self,
            request: SFCRequest,
            service_index: int,
            candidates: list[int],
            current_node: int,
            current_time: float,
            data_gb: float,
            service_id: int,
            context: dict,
    ) -> list[dict]:
        records = []
        for node_id in candidates:
            route = route_data(current_node, node_id, data_gb, current_time, context)
            if not route["reachable"]:
                records.append(
                    {
                        "node_id": node_id,
                        "reachable": False,
                        "score": math.inf,
                        "failure_reason": route.get("failure_reason", "route_failed"),
                        "route": route,
                    }
                )
                continue
            compute = compute_service_execution(
                service_id, node_id, route["arrival_time"], context
            )
            delay_cost = route["delay_s"] + compute["queue_delay_s"] + compute["compute_delay_s"]
            energy_cost = route["communication_energy_j"] + compute["compute_energy_j"]
            score = (
                    self.config.delay_weight * delay_cost
                    + self.config.energy_weight * energy_cost / 1000.0
                    + self.config.slot_switch_penalty_weight * route["slot_crossings"]
                    + self.config.route_failure_risk_weight
                    * float(route.get("route_failure_risk", 0.0))
            )
            bottleneck_gb = float(route.get("bottleneck_capacity_gb", 0.0))
            bottleneck_shortage = 0.0
            if data_gb > 1.0e-9:
                bottleneck_shortage = max(0.0, 1.0 - min(1.0, bottleneck_gb / data_gb))
            bottleneck_shortage_feature = min(
                1.0,
                self.config.candidate_bottleneck_shortage_penalty_weight * bottleneck_shortage,
            )
            score += self.config.candidate_bottleneck_shortage_penalty_weight * bottleneck_shortage
            egress = self._estimate_egress_capacity(
                request,
                service_index,
                node_id,
                compute["compute_finish_s"],
                context,
            )
            egress_shortage = float(egress["egress_bottleneck_shortage"])
            combined_shortage_feature = min(
                1.0,
                max(
                    bottleneck_shortage_feature,
                    self.config.candidate_egress_shortage_penalty_weight * egress_shortage,
                ),
            )
            score += self.config.candidate_egress_shortage_penalty_weight * egress_shortage
            records.append(
                {
                    "node_id": node_id,
                    "reachable": True,
                    "score": score,
                    "delay_cost_s": delay_cost,
                    "energy_cost_j": energy_cost,
                    "bottleneck_capacity_gb": bottleneck_gb,
                    "bottleneck_shortage": bottleneck_shortage,
                    **{
                        key: value
                        for key, value in egress.items()
                        if key != "egress_route"
                    },
                    "extra_features": [
                        min(1.0, route["delay_s"] / 10.0),
                        min(1.0, float(route.get("route_failure_risk", 0.0))),
                        combined_shortage_feature,
                        min(
                            1.0,
                            (
                                    compute["queue_delay_s"]
                                    + compute["compute_delay_s"] / 100_000.0
                                    + energy_cost / 10_000_000.0
                            ),
                        ),
                    ],
                    "route": route,
                    "compute": compute,
                }
            )
        return records

    def record_step_outcome(
            self, decision: CandidateDecision, reward: float, done: bool = False
    ) -> None:
        if not self.training_available or not decision.metadata:
            return
        self.transitions.append(
            PPOTransition(
                log_prob=decision.metadata["log_prob"],
                value=decision.metadata["value"],
                reward=reward,
                entropy=decision.metadata["entropy"],
                done=done,
            )
        )

    def ppo_update(
            self,
            transitions: list[PPOTransition] | None = None,
            clip_epsilon: float = 0.2,
            gamma: float = 0.99,
            epochs: int = 1,
    ) -> dict:
        if not self.training_available:
            raise RuntimeError("PyTorch is required for PPO training.")
        transitions = transitions if transitions is not None else self.transitions
        if not transitions:
            return {"updated": False, "loss": 0.0, "transition_count": 0}

        rewards = [transition.reward for transition in transitions]
        returns = []
        running = 0.0
        for transition in reversed(transitions):
            running = transition.reward + gamma * running * (0.0 if transition.done else 1.0)
            returns.append(running)
        returns.reverse()

        values = torch.stack([transition.value for transition in transitions])
        log_probs = torch.stack([transition.log_prob for transition in transitions])
        old_log_probs = log_probs.detach()
        entropies = torch.stack([transition.entropy for transition in transitions])
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = returns_t - values.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

        last_stats = {}
        for _ in range(epochs):
            ratio = torch.exp(log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns_t)
            entropy_bonus = entropies.mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
            self.optimizer.zero_grad()
            loss.backward(retain_graph=_ < epochs - 1)
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            last_stats = {
                "updated": True,
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy_bonus.detach().cpu()),
                "transition_count": len(transitions),
            }
        if transitions is self.transitions:
            self.transitions.clear()
        return last_stats

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.training_available:
            path.with_suffix(".json").write_text(
                '{"saved": false, "reason": "PyTorch is not installed."}',
                encoding="utf-8",
            )
            return
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "hidden_dim": self.hidden_dim,
                "train_mode": self.train_mode,
                "device": str(self.device),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        if not self.training_available:
            raise RuntimeError("PyTorch is required to load PPO-GNN checkpoints.")
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
