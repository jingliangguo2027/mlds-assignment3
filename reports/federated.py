"""FedAvg and Ray Core clients. No federated-learning library is used."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from zipfile import ZipFile
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Config:
    clients: int = 10
    epochs: int = 3
    lr: float = 0.1
    rounds: int = 100
    seed: int = 42
    inputs: str = "ink"
    noise: float = 0.0


def load_clients(archive: Path, split_seed: int = 42):
    """Load only local data. The held-out test file is opened separately."""
    with ZipFile(archive) as z:
        with z.open("Assignment3-data/train_data.npy") as f:
            records = np.load(f, allow_pickle=True)
    rng = np.random.default_rng(split_seed)
    clients = []
    for record in records:
        x = np.asarray(record["images"], dtype=np.float32).reshape(-1, 784)
        y = np.asarray(record["labels"], dtype=np.int64)
        if len(y) < 2:
            raise ValueError("A client needs at least two images for an 80/20 split")
        order = rng.permutation(len(y))
        cut = int(0.8 * len(y))
        a, b = order[:cut], order[cut:]
        clients.append({
            "x": torch.from_numpy(x[a].copy()), "y": torch.from_numpy(y[a].copy()),
            "vx": torch.from_numpy(x[b].copy()), "vy": torch.from_numpy(y[b].copy()),
            "train_indices": a, "val_indices": b,
            "counts": np.bincount(y, minlength=62),
        })
    return clients


def load_test(archive: Path):
    with ZipFile(archive) as z:
        with z.open("Assignment3-data/test_data.npy") as f:
            records = np.load(f, allow_pickle=True)
    x = np.concatenate([r["images"] for r in records]).astype(np.float32).reshape(-1, 784)
    y = np.concatenate([r["labels"] for r in records]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def transform(x: torch.Tensor, inputs: str):
    if inputs == "raw":
        return x
    if inputs == "ink":
        return 1.0 - x
    raise ValueError(f"Unknown input transform: {inputs}")


def model(seed: int = 42):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 62))


def copy_state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def noise_images(clients, scale: float, seed: int):
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("Noise scale must be finite and nonnegative")
    if scale == 0:
        return [c["x"].clone() for c in clients]
    rng = np.random.default_rng(seed)
    return [c["x"] + torch.from_numpy(rng.laplace(0.0, scale, c["x"].shape).astype(np.float32))
            for c in clients]


def local_fit(state, x, y, *, inputs, epochs, lr, seed):
    local = model(seed)
    local.load_state_dict(state)
    local.train()
    # Copy Ray's possibly read-only arrays before constructing tensors.
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(np.asarray(x), dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(np.asarray(y), dtype=torch.long)
    data = TensorDataset(transform(x, inputs), y)
    loader = DataLoader(data, batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.SGD(local.parameters(), lr=lr)
    for _ in range(epochs):
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(local(images), labels)
            loss.backward()
            optimizer.step()
    return len(y), copy_state(local)


def weighted_mean(updates):
    if not updates or any(n <= 0 for n, _ in updates):
        raise ValueError("FedAvg needs positive-sized client updates")
    total = sum(n for n, _ in updates)
    result = {k: torch.zeros_like(v) for k, v in updates[0][1].items()}
    for n, state in updates:
        for k, value in state.items():
            result[k].add_(value, alpha=n / total)
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise FloatingPointError("A local update produced non-finite parameters")
    return result


def client_seed(run_seed, round_id, client_id):
    return run_seed * 1_000_000 + round_id * 1000 + int(client_id)


def serial_round(net, clients, selected, cfg, round_id, training_images=None):
    state = copy_state(net)
    updates = []
    for i in selected:
        c = clients[i]
        x = c["x"] if training_images is None else training_images[i]
        updates.append(local_fit(state, x, c["y"], inputs=cfg.inputs, epochs=cfg.epochs,
                                 lr=cfg.lr, seed=client_seed(cfg.seed, round_id, i)))
    net.load_state_dict(weighted_mean(updates))


@torch.no_grad()
def evaluate(net, x, y, inputs="ink"):
    net.eval()
    total_loss, correct = 0.0, 0
    for a in range(0, len(y), 512):
        xb, yb = transform(x[a:a + 512], inputs), y[a:a + 512]
        scores = net(xb)
        total_loss += nn.functional.cross_entropy(scores, yb, reduction="sum").item()
        correct += int((scores.argmax(1) == yb).sum())
    return {"loss": total_loss / len(y), "accuracy": correct / len(y)}


def evaluate_clients(net, clients, split, inputs, training_images=None):
    # In this simulation, evaluation runs locally; only scalar metrics are aggregated.
    n_total = 0
    total_loss = total_acc = 0.0
    for i, c in enumerate(clients):
        x, y = (c["x"], c["y"]) if split == "train" else (c["vx"], c["vy"])
        if split == "train" and training_images is not None:
            x = training_images[i]
        value = evaluate(net, x, y, inputs)
        n_total += len(y)
        total_loss += len(y) * value["loss"]
        total_acc += len(y) * value["accuracy"]
    return {"loss": total_loss / n_total, "accuracy": total_acc / n_total}


def run(clients, cfg: Config, parallel=False, progress=None):
    if cfg.clients < 1 or cfg.clients > len(clients):
        raise ValueError("Invalid number of participating clients")
    net = model(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    images = noise_images(clients, cfg.noise, cfg.seed + 81) if cfg.noise else None
    history = []
    update_seconds = 0.0
    start = time.perf_counter()
    for r in range(1, cfg.rounds + 1):
        selected = rng.permutation(len(clients))[:cfg.clients]
        before = time.perf_counter()
        if parallel:
            ray_round(net, clients, selected, cfg, r, images)
        else:
            serial_round(net, clients, selected, cfg, r, images)
        update_seconds += time.perf_counter() - before
        metrics = {
            "round": r,
            "train": evaluate_clients(net, clients, "train", cfg.inputs, images),
            "validation": evaluate_clients(net, clients, "val", cfg.inputs),
        }
        if images is not None:
            metrics["clean_train"] = evaluate_clients(net, clients, "train", cfg.inputs)
        else:
            metrics["clean_train"] = metrics["train"].copy()
        history.append(metrics)
        if progress is not None and (r == 1 or r % 25 == 0 or r == cfg.rounds):
            progress(r, metrics)
    return net, {"config": asdict(cfg), "history": history, "update_seconds": update_seconds,
                 "elapsed_seconds": time.perf_counter() - start, "parallel": parallel}


_actor_class = None


def actor_class():
    global _actor_class
    if _actor_class is None:
        import ray

        @ray.remote(num_cpus=1, num_gpus=0)
        class ClientActor:
            def fit(self, state, x, y, inputs, epochs, lr, seed, fail=False):
                torch.set_num_threads(1)
                if fail:
                    raise RuntimeError("Injected client failure")
                return local_fit(state, x, y, inputs=inputs, epochs=epochs, lr=lr, seed=seed)

        _actor_class = ClientActor
    return _actor_class


def ray_round(net, clients, selected, cfg, round_id, training_images=None,
              fail_client=None, audit=None):
    import ray
    if not ray.is_initialized():
        raise RuntimeError("Call ray.init before running parallel clients")
    slots = int(ray.cluster_resources().get("CPU", 0))
    if slots < 1:
        raise RuntimeError("Ray needs at least one CPU")
    state = copy_state(net)
    updates = []
    Actor = actor_class()
    for a in range(0, len(selected), slots):
        actors = []
        try:
            jobs = []
            group = selected[a:a + slots]
            if audit is not None:
                audit.append(len(group))
            for i in group:
                c = clients[i]
                actor = Actor.remote()
                actors.append(actor)
                x = c["x"] if training_images is None else training_images[i]
                jobs.append(actor.fit.remote(state, x.numpy(), c["y"].numpy(), cfg.inputs,
                    cfg.epochs, cfg.lr, client_seed(cfg.seed, round_id, i), int(i) == fail_client))
            updates.extend(ray.get(jobs))
        finally:
            for actor in actors:
                ray.kill(actor)
    # Commit only after every selected client succeeds. A failed round leaves net unchanged.
    net.load_state_dict(weighted_mean(updates))


def max_difference(a, b):
    return max(float((a[k] - b[k]).abs().max()) for k in a)
