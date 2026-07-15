from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata


MODEL_VERSION = "gcartpole.capture_funnel:polynomial-logistic-v1"


def effective_state(
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    progress: float,
    qpos_scale_power: float,
    qvel_scale_power: float,
    qpos_scale_start: float = 0.0,
    qpos_scale_end: float = 1.0,
    qvel_scale_start: float = 0.0,
    qvel_scale_end: float = 1.0,
    cart_target: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the state-list curriculum transform without simulator noise."""
    qpos = np.asarray(qpos, dtype=np.float64).copy()
    qvel = np.asarray(qvel, dtype=np.float64).copy()
    if qpos.ndim != 1 or qvel.shape != qpos.shape or qpos.size < 2:
        raise ValueError("qpos and qvel must be same-length one-dimensional arrays")
    p = float(np.clip(progress, 0.0, 1.0))
    qpos_scale = float(qpos_scale_start) + (float(qpos_scale_end) - float(qpos_scale_start)) * p ** float(qpos_scale_power)
    qvel_scale = float(qvel_scale_start) + (float(qvel_scale_end) - float(qvel_scale_start)) * p ** float(qvel_scale_power)
    target = np.zeros_like(qpos)
    target[0] = float(cart_target)
    delta = qpos - target
    delta[1:] = (delta[1:] + np.pi) % (2.0 * np.pi) - np.pi
    return target + qpos_scale * delta, qvel_scale * qvel


def normalized_capture_coordinates(
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    cart_position_bound: float,
    angle_bound: float,
    cart_velocity_bound: float,
    hinge_velocity_bound: float,
) -> np.ndarray:
    """Map a physical state to cart/absolute-angle/velocity funnel coordinates."""
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    if qpos.ndim != 1 or qvel.shape != qpos.shape or qpos.size < 2:
        raise ValueError("qpos and qvel must be same-length one-dimensional arrays")
    absolute_angles = np.cumsum(qpos[1:])
    return np.r_[
        qpos[0] / float(cart_position_bound),
        absolute_angles / float(angle_bound),
        qvel[0] / float(cart_velocity_bound),
        qvel[1:] / float(hinge_velocity_bound),
    ]


def polynomial_features(coordinates: np.ndarray) -> np.ndarray:
    """Return first- and second-order terms, excluding a constant column."""
    x = np.asarray(coordinates, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("coordinates must be one- or two-dimensional")
    upper_i, upper_j = np.triu_indices(x.shape[1])
    return np.concatenate([x, x[:, upper_i] * x[:, upper_j]], axis=1)


def deterministic_stratified_split(labels: np.ndarray, identifiers: list[str], fraction: float) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(identifiers) != labels.size:
        raise ValueError("labels and identifiers must have matching lengths")
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("fraction must be between zero and one")
    development: list[int] = []
    holdout: list[int] = []
    for label in np.unique(labels):
        group = np.flatnonzero(labels == label).tolist()
        if len(group) < 2:
            raise ValueError("each class needs at least two examples for a stratified split")
        group.sort(key=lambda index: hashlib.sha256(identifiers[index].encode("utf-8")).digest())
        holdout_count = int(np.clip(round(len(group) * float(fraction)), 1, max(1, len(group) - 1)))
        holdout.extend(group[:holdout_count])
        development.extend(group[holdout_count:])
    return np.asarray(sorted(development), dtype=np.int64), np.asarray(sorted(holdout), dtype=np.int64)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(logits)
    positive = logits >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    result[~positive] = exp_logits / (1.0 + exp_logits)
    return result


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities >= float(threshold)
    positive = labels == 1
    tp = int(np.sum(predictions & positive))
    fp = int(np.sum(predictions & ~positive))
    tn = int(np.sum(~predictions & ~positive))
    fn = int(np.sum(~predictions & positive))
    ranks = rankdata(probabilities, method="average")
    positive_count = int(np.sum(positive))
    negative_count = int(labels.size - positive_count)
    auc = 0.5
    if positive_count and negative_count:
        auc = float((np.sum(ranks[positive]) - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count))
    return {
        "count": int(labels.size),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": float((tp + tn) / max(1, labels.size)),
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "roc_auc": auc,
    }


def conservative_threshold(labels: np.ndarray, probabilities: np.ndarray, minimum_precision: float) -> tuple[float, dict[str, float | int]]:
    candidates = np.unique(np.r_[probabilities, 1.0])
    accepted: list[tuple[float, float, dict[str, float | int]]] = []
    for threshold in candidates:
        metrics = binary_metrics(labels, probabilities, float(threshold))
        if int(metrics["true_positive"]) > 0 and float(metrics["precision"]) >= float(minimum_precision):
            accepted.append((float(metrics["recall"]), -float(threshold), metrics))
    if not accepted:
        metrics = binary_metrics(labels, probabilities, 1.0)
        return 1.0, metrics
    _, negative_threshold, metrics = max(accepted, key=lambda item: (item[0], item[1]))
    return -negative_threshold, metrics


@dataclass
class CaptureFunnelModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: float
    coordinate_bounds: dict[str, float]
    coordinate_abs_limits: np.ndarray
    acceptance_threshold: float

    def coordinates(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        return normalized_capture_coordinates(qpos, qvel, **self.coordinate_bounds)

    def domain_distance(self, qpos: np.ndarray, qvel: np.ndarray) -> float:
        coordinates = self.coordinates(qpos, qvel)
        scale = np.maximum(self.coordinate_abs_limits, 1e-12)
        excess = np.maximum(np.abs(coordinates) / scale - 1.0, 0.0)
        return float(np.linalg.norm(excess))

    def predict_probability(self, qpos: np.ndarray, qvel: np.ndarray) -> float:
        coordinates = self.coordinates(qpos, qvel)
        if np.any(np.abs(coordinates) > self.coordinate_abs_limits):
            return 0.0
        features = polynomial_features(coordinates)[0]
        standardized = (features - self.feature_mean) / self.feature_scale
        return float(_sigmoid(np.asarray([standardized @ self.weights + self.bias]))[0])

    def accepts(self, qpos: np.ndarray, qvel: np.ndarray) -> bool:
        return self.predict_probability(qpos, qvel) >= self.acceptance_threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": MODEL_VERSION,
            "feature_mean": self.feature_mean.astype(float).tolist(),
            "feature_scale": self.feature_scale.astype(float).tolist(),
            "weights": self.weights.astype(float).tolist(),
            "bias": float(self.bias),
            "coordinate_bounds": {key: float(value) for key, value in self.coordinate_bounds.items()},
            "coordinate_abs_limits": self.coordinate_abs_limits.astype(float).tolist(),
            "acceptance_threshold": float(self.acceptance_threshold),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CaptureFunnelModel":
        if payload.get("model_version") != MODEL_VERSION:
            raise ValueError(f"unsupported capture funnel model: {payload.get('model_version')!r}")
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=float(payload["bias"]),
            coordinate_bounds={key: float(value) for key, value in payload["coordinate_bounds"].items()},
            coordinate_abs_limits=np.asarray(payload["coordinate_abs_limits"], dtype=np.float64),
            acceptance_threshold=float(payload["acceptance_threshold"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CaptureFunnelModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8"))["model"])


def fit_capture_funnel(
    coordinates: np.ndarray,
    labels: np.ndarray,
    development_indices: np.ndarray,
    *,
    l2: float,
    max_iterations: int,
) -> tuple[CaptureFunnelModel, dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 1 or set(np.unique(labels).tolist()) != {0.0, 1.0}:
        raise ValueError("capture funnel fitting requires both binary classes")
    features = polynomial_features(coordinates)
    development = features[np.asarray(development_indices, dtype=np.int64)]
    targets = labels[np.asarray(development_indices, dtype=np.int64)]
    mean = np.mean(development, axis=0)
    scale = np.std(development, axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (development - mean) / scale
    positive_weight = targets.size / max(1.0, 2.0 * float(np.sum(targets)))
    negative_weight = targets.size / max(1.0, 2.0 * float(np.sum(1.0 - targets)))
    sample_weight = np.where(targets > 0.5, positive_weight, negative_weight)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights = parameters[:-1]
        bias = parameters[-1]
        logits = standardized @ weights + bias
        probabilities = _sigmoid(logits)
        eps = 1e-12
        loss = -np.mean(sample_weight * (targets * np.log(probabilities + eps) + (1.0 - targets) * np.log(1.0 - probabilities + eps)))
        loss += 0.5 * float(l2) * float(weights @ weights)
        error = sample_weight * (probabilities - targets) / targets.size
        gradient = np.r_[standardized.T @ error + float(l2) * weights, np.sum(error)]
        return float(loss), gradient

    initial = np.zeros(features.shape[1] + 1, dtype=np.float64)
    result = minimize(objective, initial, jac=True, method="L-BFGS-B", options={"maxiter": int(max_iterations)})
    model = CaptureFunnelModel(
        feature_mean=mean,
        feature_scale=scale,
        weights=result.x[:-1],
        bias=float(result.x[-1]),
        coordinate_bounds={},
        coordinate_abs_limits=np.full(coordinates.shape[1], np.inf, dtype=np.float64),
        acceptance_threshold=0.5,
    )
    diagnostics = {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "objective": float(result.fun),
        "l2": float(l2),
    }
    return model, diagnostics
