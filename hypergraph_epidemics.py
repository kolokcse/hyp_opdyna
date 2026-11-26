"""Fast parallel SIS-awareness experiments on random hypergraphs.

The module provides:
  * ER, Chung-Lu, Barabasi-Albert and Bianconi-Barabasi hypergraph generators.
  * Vectorized tau-leap approximation of a continuous-time SIS awareness model.
  * Vectorized RK4 integration of its individual-based mean-field approximation.
  * Parallel parameter sweeps over network size, model parameters and samples.
  * Helpers for fractal-dimension fits and degree-conditioned prevalence.

Hyperedges are stored in CSR-style incidence arrays.  Node states are NumPy bool
arrays.  Sparse matrix-vector products aggregate node values to hyperedges and
back to nodes without Python loops over nodes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class Hypergraph:
    n: int
    edges: tuple[np.ndarray, ...]
    incidence: sparse.csr_matrix
    edge_sizes: np.ndarray
    degree: np.ndarray

    @classmethod
    def from_edges(cls, n: int, edges: Iterable[Sequence[int]]) -> "Hypergraph":
        clean: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()
        for edge in edges:
            a = np.unique(np.asarray(edge, dtype=np.int32))
            if a.size < 2 or a[0] < 0 or a[-1] >= n:
                continue
            key = tuple(int(x) for x in a)
            if key not in seen:
                seen.add(key)
                clean.append(a)
        if not clean:
            raise ValueError("The generated hypergraph has no valid hyperedges.")
        rows = np.concatenate(clean)
        cols = np.repeat(np.arange(len(clean), dtype=np.int32), [x.size for x in clean])
        data = np.ones(rows.size, dtype=np.float64)
        B = sparse.csr_matrix((data, (rows, cols)), shape=(n, len(clean)))
        return cls(n, tuple(clean), B, np.asarray(B.sum(axis=0)).ravel(),
                   np.asarray(B.sum(axis=1)).ravel())


def _edge_sizes(rng: np.random.Generator, m: int, edge_size: int | Sequence[int]) -> np.ndarray:
    if np.isscalar(edge_size):
        return np.full(m, max(2, int(edge_size)), dtype=np.int32)
    values = np.asarray(edge_size, dtype=np.int32)
    values = values[values >= 2]
    if values.size == 0:
        raise ValueError("edge_size must contain integers at least 2.")
    return rng.choice(values, size=m, replace=True)


def er_hypergraph(n: int, m: int, edge_size: int | Sequence[int] = 3,
                  seed: int | None = None) -> Hypergraph:
    """Uniform random hyperedges; duplicates are removed."""
    rng = np.random.default_rng(seed)
    sizes = _edge_sizes(rng, m, edge_size)
    edges = [rng.choice(n, size=min(int(k), n), replace=False) for k in sizes]
    return Hypergraph.from_edges(n, edges)


def power_law_weights(n: int, gamma: float, mean_degree: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Truncated Pareto expected hyperdegrees, rescaled to the requested mean."""
    if gamma <= 1:
        raise ValueError("gamma must be greater than 1.")
    w = 1.0 + rng.pareto(gamma - 1.0, n)
    w = np.minimum(w, n ** (1.0 / max(gamma - 1.0, 1.0)))
    return w * (float(mean_degree) / w.mean())


def chung_lu_hypergraph(n: int, mean_degree: float = 6.0, gamma: float = 2.5,
                        edge_size: int | Sequence[int] = 3,
                        seed: int | None = None) -> Hypergraph:
    """Expected-hyperdegree model using weighted sampling without replacement.

    Conditional on edge size, vertices are sampled proportional to target
    expected hyperdegrees.  The number of edges is chosen so expected incidence
    count is approximately n times mean_degree.
    """
    rng = np.random.default_rng(seed)
    w = power_law_weights(n, gamma, mean_degree, rng)
    mean_size = float(edge_size if np.isscalar(edge_size) else np.mean(edge_size))
    m = max(1, int(round(n * mean_degree / mean_size)))
    sizes = _edge_sizes(rng, m, edge_size)
    p = w / w.sum()
    edges = [rng.choice(n, size=min(int(k), n), replace=False, p=p) for k in sizes]
    return Hypergraph.from_edges(n, edges)


def preferential_hypergraph(n: int, edges_per_new_node: int = 2,
                            edge_size: int | Sequence[int] = 3,
                            fitness: np.ndarray | None = None,
                            attractiveness: float = 1.0,
                            seed: int | None = None) -> Hypergraph:
    """BA hypergraph, or Bianconi-Barabasi when fitness is supplied."""
    rng = np.random.default_rng(seed)
    first_k = int(edge_size if np.isscalar(edge_size) else max(edge_size))
    first_k = min(max(2, first_k), n)
    eta = np.ones(n) if fitness is None else np.asarray(fitness, dtype=float)
    if eta.shape != (n,) or np.any(eta <= 0):
        raise ValueError("fitness must be a positive vector of length n.")
    degree = np.zeros(n, dtype=float)
    edges: list[np.ndarray] = [np.arange(first_k, dtype=np.int32)]
    degree[:first_k] = 1.0
    for new in range(first_k, n):
        for _ in range(edges_per_new_node):
            k = int(_edge_sizes(rng, 1, edge_size)[0])
            old_count = min(k - 1, new)
            score = (degree[:new] + attractiveness) * eta[:new]
            chosen = rng.choice(new, size=old_count, replace=False, p=score / score.sum())
            edge = np.concatenate(([new], chosen)).astype(np.int32)
            edges.append(edge)
            degree[edge] += 1.0
    return Hypergraph.from_edges(n, edges)


def barabasi_hypergraph(n: int, edges_per_new_node: int = 2,
                        edge_size: int | Sequence[int] = 3,
                        seed: int | None = None) -> Hypergraph:
    return preferential_hypergraph(n, edges_per_new_node, edge_size, None, 1.0, seed)


def bianconi_barabasi_hypergraph(n: int, edges_per_new_node: int = 2,
                                 edge_size: int | Sequence[int] = 3,
                                 fitness_shape: float = 1.0,
                                 seed: int | None = None) -> Hypergraph:
    rng = np.random.default_rng(seed)
    fitness = rng.gamma(shape=fitness_shape, scale=1.0, size=n) + 1e-12
    return preferential_hypergraph(n, edges_per_new_node, edge_size, fitness, 1.0, seed)


def generate_hypergraph(model: str, n: int, seed: int | None = None, **kwargs: Any) -> Hypergraph:
    model = model.lower().replace("-", "_")
    fn = {
        "er": er_hypergraph,
        "chung_lu": chung_lu_hypergraph,
        "barabasi": barabasi_hypergraph,
        "ba": barabasi_hypergraph,
        "bianconi_barabasi": bianconi_barabasi_hypergraph,
        "bb": bianconi_barabasi_hypergraph,
    }.get(model)
    if fn is None:
        raise ValueError(f"Unknown hypergraph model: {model}")
    return fn(n=n, seed=seed, **kwargs)


@dataclass(frozen=True)
class SISConfig:
    beta: float = 1.0
    mu: float = 1.0
    alpha_s: float = 0.0
    alpha_i: float = 1.0
    awareness_strength: float = 1.0
    synergy: float = 1.0
    dt: float = 0.02
    t_max: float = 100.0
    record_every: int = 10
    initial_fraction: float = 0.1
    quasi_stationary: bool = True
    burn_in_fraction: float = 0.5


def _signals(H: Hypergraph, x: np.ndarray) -> np.ndarray:
    denominator = np.maximum(
        H.edge_sizes - 1.0,
        1.0,
    )

    edge_infected = np.asarray(
        H.incidence.T @ x
    ).ravel()

    total_signal = np.asarray(
        H.incidence @ (edge_infected / denominator)
    ).ravel()

    self_weight = np.asarray(
        H.incidence @ (1.0 / denominator)
    ).ravel()

    signal = total_signal - x * self_weight

    return np.maximum(signal, 0.0)

def infection_pressure(
    H: Hypergraph,
    x: np.ndarray,
    cfg: SISConfig,
) -> np.ndarray:

    signal = _signals(H, x)

    awareness = np.exp(
        -cfg.awareness_strength * signal
    )

    infectious_node = (
        x
        * np.power(
            awareness,
            cfg.alpha_i,
        )
    )

    edge_load = np.asarray(
        H.incidence.T @ infectious_node
    ).ravel()

    denominator = np.maximum(
        H.edge_sizes - 1.0,
        1.0,
    )

    normalized_load = edge_load / denominator

    if cfg.synergy != 1.0:
        normalized_load = np.power(
            np.maximum(normalized_load, 0.0),
            cfg.synergy,
        )

    pressure = np.asarray(
        H.incidence @ normalized_load
    ).ravel()

    return (
        cfg.beta
        * np.power(
            awareness,
            cfg.alpha_s,
        )
        * pressure
    )


def simulate_tau_leap(H: Hypergraph, cfg: SISConfig, seed: int | None = None) -> dict[str, Any]:
    """Synchronous tau-leap approximation; transition probabilities are exact
    for constant rates during each step: p equals 1 minus exp of negative rate dt.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(H.n, dtype=np.float64)
    initial = max(1, int(round(cfg.initial_fraction * H.n)))
    x[rng.choice(H.n, size=min(initial, H.n), replace=False)] = 1.0
    steps = int(math.ceil(cfg.t_max / cfg.dt))
    rec = max(1, int(cfg.record_every))
    times, counts = [0.0], [int(x.sum())]
    node_sum = np.zeros(H.n, dtype=np.float64)
    node_obs = 0
    burn = int(cfg.burn_in_fraction * steps)
    for step in range(1, steps + 1):
        rate = infection_pressure(H, x, cfg)
        p_inf = -np.expm1(-cfg.dt * rate)
        p_rec = -math.expm1(-cfg.dt * cfg.mu)
        old = x.astype(bool)
        infect = (~old) & (rng.random(H.n) < p_inf)
        recover = old & (rng.random(H.n) < p_rec)
        x[infect] = 1.0
        x[recover] = 0.0
        if cfg.quasi_stationary and x.sum() == 0:
            x[rng.integers(H.n)] = 1.0
        if step >= burn:
            node_sum += x
            node_obs += 1
        if step % rec == 0 or step == steps:
            times.append(step * cfg.dt)
            counts.append(int(x.sum()))
    tail = np.asarray(counts)[max(1, len(counts) // 2):]
    return {
        "time": np.asarray(times), "infected": np.asarray(counts),
        "stationary_I": float(tail.mean()), "stationary_density": float(tail.mean() / H.n),
        "degree": H.degree.astype(np.int32),
        "node_prevalence": node_sum / max(node_obs, 1),
    }

def meanfield_signal(
    u: np.ndarray,
    H: Hypergraph,
) -> np.ndarray:

    denominator = np.maximum(
        H.edge_sizes - 1.0,
        1.0,
    )

    edge_mean = np.asarray(
        H.incidence.T @ u
    ).ravel()

    total_signal = np.asarray(
        H.incidence @ (edge_mean / denominator)
    ).ravel()

    self_weight = np.asarray(
        H.incidence @ (1.0 / denominator)
    ).ravel()

    signal = total_signal - u * self_weight

    return np.maximum(signal, 0.0)

def meanfield_rhs(
    u: np.ndarray,
    H: Hypergraph,
    cfg: SISConfig,
) -> np.ndarray:

    denominator = np.maximum(
        H.edge_sizes - 1.0,
        1.0,
    )

    signal = meanfield_signal(u, H)

    awareness = np.exp(
        -cfg.awareness_strength * signal
    )

    effective_infection = (
        u
        * np.power(
            awareness,
            cfg.alpha_i,
        )
    )

    edge_load = np.asarray(
        H.incidence.T @ effective_infection
    ).ravel()

    normalized_load = edge_load / denominator

    if cfg.synergy != 1.0:
        normalized_load = np.power(
            np.maximum(normalized_load, 0.0),
            cfg.synergy,
        )

    pressure = np.asarray(
        H.incidence @ normalized_load
    ).ravel()

    infection_term = (
        cfg.beta
        * (1.0 - u)
        * np.power(
            awareness,
            cfg.alpha_s,
        )
        * pressure
    )

    return -cfg.mu * u + infection_term

def solve_meanfield(H: Hypergraph, cfg: SISConfig, seed: int | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    u = np.full(H.n, cfg.initial_fraction, dtype=float)
    u += rng.normal(0.0, 1e-8, H.n)
    u = np.clip(u, 0.0, 1.0)
    steps = int(math.ceil(cfg.t_max / cfg.dt))
    rec = max(1, int(cfg.record_every))
    times, counts = [0.0], [float(u.sum())]
    node_sum = np.zeros(H.n)
    node_obs = 0
    burn = int(cfg.burn_in_fraction * steps)
    for step in range(1, steps + 1):
        k1 = meanfield_rhs(u, H, cfg)
        k2 = meanfield_rhs(np.clip(u + 0.5 * cfg.dt * k1, 0, 1), H, cfg)
        k3 = meanfield_rhs(np.clip(u + 0.5 * cfg.dt * k2, 0, 1), H, cfg)
        k4 = meanfield_rhs(np.clip(u + cfg.dt * k3, 0, 1), H, cfg)
        u = np.clip(u + cfg.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, 0.0, 1.0)
        if step >= burn:
            node_sum += u
            node_obs += 1
        if step % rec == 0 or step == steps:
            times.append(step * cfg.dt)
            counts.append(float(u.sum()))
    tail = np.asarray(counts)[max(1, len(counts) // 2):]
    return {
        "time": np.asarray(times), "infected": np.asarray(counts),
        "stationary_I": float(tail.mean()), "stationary_density": float(tail.mean() / H.n),
        "degree": H.degree.astype(np.int32),
        "node_prevalence": node_sum / max(node_obs, 1),
    }


def _one_job(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = int(job["seed"])
    H = generate_hypergraph(job["model"], int(job["n"]), seed=seed, **job["graph_params"])
    cfg = SISConfig(**job["sis_params"])
    out = solve_meanfield(H, cfg, seed + 1) if job["method"] == "meanfield" else simulate_tau_leap(H, cfg, seed + 1)
    meta = {"method": job["method"], "model": job["model"], "n": H.n,
            "sample": int(job["sample"]), "seed": seed, **job["sweep_values"]}
    return meta, out


def run_sweep(model: str, method: str, network_sizes: Sequence[int], samples: int,
              graph_params: Mapping[str, Any], sis_config: SISConfig,
              sweep: Mapping[str, Sequence[Any]] | None = None,
              workers: int | None = None, seed: int = 12345,
              output_prefix: str = "hypergraph_sweep") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a parallel Cartesian sweep. Sweep keys can be graph parameters or
    SISConfig fields. Results are saved as summary CSV and node-level CSV.
    """
    from itertools import product
    sweep = dict(sweep or {})
    keys = list(sweep)
    values = [list(sweep[k]) for k in keys]
    combinations = list(product(*values)) if keys else [()]
    ss = np.random.SeedSequence(seed)
    job_count = len(network_sizes) * samples * len(combinations)
    child_seeds = ss.spawn(job_count)
    jobs, q = [], 0
    sis_fields = set(asdict(sis_config))
    for combo in combinations:
        selected = dict(zip(keys, combo))
        for n in network_sizes:
            for sample in range(samples):
                gp = dict(graph_params)
                sp = asdict(sis_config)
                for key, value in selected.items():
                    if key in sis_fields:
                        sp[key] = value
                    else:
                        gp[key] = value
                jobs.append({"model": model, "method": method, "n": int(n), "sample": sample,
                             "seed": int(child_seeds[q].generate_state(1)[0]),
                             "graph_params": gp, "sis_params": sp, "sweep_values": selected})
                q += 1
    summaries, node_rows = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one_job, job) for job in jobs]
        for future in as_completed(futures):
            meta, out = future.result()
            summaries.append({**meta, "stationary_I": out["stationary_I"],
                              "stationary_density": out["stationary_density"]})
            for degree, prevalence in zip(out["degree"], out["node_prevalence"]):
                node_rows.append({**meta, "degree": int(degree), "prevalence": float(prevalence)})
    summary = pd.DataFrame(summaries).sort_values(keys + ["n", "sample"] if keys else ["n", "sample"])
    nodes = pd.DataFrame(node_rows)
    summary.to_csv(f"{output_prefix}_summary.csv", index=False)
    nodes.to_csv(f"{output_prefix}_nodes.csv", index=False)
    return summary, nodes


def fit_fractal_dimension(summary: pd.DataFrame, group_by: Sequence[str] = ("gamma",),
                          response: str = "stationary_I") -> pd.DataFrame:
    """Fit log E[I] = intercept + d_f log n after averaging samples."""
    cols = list(group_by)
    averaged = summary.groupby(cols + ["n"], as_index=False)[response].mean()
    rows = []
    groups = averaged.groupby(cols[0] if len(cols) == 1 else cols, dropna=False)
    for key, frame in groups:
        frame = frame[(frame[response] > 0) & (frame["n"] > 0)]
        if len(frame) < 2:
            continue
        slope, intercept = np.polyfit(np.log(frame["n"]), np.log(frame[response]), 1)
        pred = intercept + slope * np.log(frame["n"])
        y = np.log(frame[response])
        r2 = 1.0 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-15)
        key_tuple = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(cols, key_tuple)), "fractal_dimension": slope,
                     "intercept": intercept, "r2": r2, "points": len(frame)})
    return pd.DataFrame(rows)


def degree_prevalence(nodes: pd.DataFrame, group_by: Sequence[str] = ("gamma",)) -> pd.DataFrame:
    return nodes.groupby(list(group_by) + ["degree"], as_index=False).agg(
        mean_prevalence=("prevalence", "mean"), sem_prevalence=("prevalence", "sem"),
        observations=("prevalence", "size"))


def example() -> None:
    # np.arange(2, 4, 6) contains only 2.  For six values on [2, 4), use np.linspace(2, 4, 6, endpoint=False).
    gammas = np.linspace(2.0, 4.0, 6, endpoint=False)
    cfg = SISConfig(beta=1.4, mu=1.0, alpha_s=0.0, alpha_i=1.0,
                    awareness_strength=1.0, dt=0.02, t_max=50.0,
                    record_every=10, initial_fraction=0.1)
    summary, nodes = run_sweep(
        model="chung_lu", method="monte_carlo",
        network_sizes=[500, 1000, 2000, 4000], samples=20,
        graph_params={"mean_degree": 6.0, "edge_size": 3},
        sis_config=cfg, sweep={"gamma": gammas},
        workers=max(1, (os.cpu_count() or 2) - 1), seed=20261009,
        output_prefix="chung_lu_mc")
    fit_fractal_dimension(summary, ["gamma"]).to_csv("chung_lu_mc_fractal.csv", index=False)
    degree_prevalence(nodes, ["gamma", "n"]).to_csv("chung_lu_mc_degree_prevalence.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", action="store_true", help="run the documented Chung-Lu example")
    args = parser.parse_args()
    if args.example:
        example()
    else:
        parser.print_help()
