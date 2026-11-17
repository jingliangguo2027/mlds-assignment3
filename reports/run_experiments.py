"""Run actual experiments and save their outputs before writing the report."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
from copy import deepcopy

import numpy as np
import torch

from federated import (Config, copy_state, load_clients, load_test, run, model, transform,
    evaluate, evaluate_clients, local_fit, weighted_mean, serial_round, ray_round,
    max_difference, client_seed, noise_images)

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
RUNS.mkdir(exist_ok=True)


def experiment(clients, cfg, parallel=False, name=None):
    # Include the algorithm source in each cache key, so changed code is never silently reused.
    source = (ROOT / "federated.py").read_bytes()
    key = hashlib.sha256(source + json.dumps(asdict(cfg), sort_keys=True).encode()
                         + str(parallel).encode()).hexdigest()[:12]
    label = name or f'{cfg.inputs}-lr{cfg.lr}-c{cfg.clients}-e{cfg.epochs}-r{cfg.rounds}-b{cfg.noise}-s{cfg.seed}'
    path = RUNS / f"{label}-{key}.json"
    state_path = path.with_suffix(".pt")
    if path.exists() and state_path.exists():
        print("CACHED", label, flush=True)
        return json.loads(path.read_text()), state_path
    print("RUN", label, flush=True)
    def progress(r, m):
        print(f'  {r:3} train={m["train"]["accuracy"]:.3%} val={m["validation"]["accuracy"]:.3%}', flush=True)
    net, result = run(clients, cfg, parallel=parallel, progress=progress)
    result["source_sha256"] = hashlib.sha256(source).hexdigest()
    result["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
                             "numpy": np.__version__, "threads": torch.get_num_threads(), "device": "cpu"}
    torch.save(copy_state(net), state_path)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result, state_path


def tune(clients):
    rows = []
    for inputs in ("raw", "ink"):
        for lr in (0.01, 0.03, 0.1, 0.3):
            cfg = Config(inputs=inputs, lr=lr, rounds=100)
            result, path = experiment(clients, cfg)
            rows.append({"config": asdict(cfg), "validation": result["history"][-1]["validation"],
                         "run": path.with_suffix(".json").name, "state": path.name})
    chosen = max(rows, key=lambda x: x["validation"]["accuracy"])
    summary = {"selection": "Final validation accuracy after 100 rounds; test data were not loaded.",
               "rows": rows, "chosen": chosen}
    (ROOT / "tuning_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SELECTED", json.dumps(chosen), flush=True)


def select(clients):
    tuning = json.loads((ROOT / "tuning_results.json").read_text())
    rows = tuning["rows"]
    for lr in (0.6, 1.0):
        cfg = Config(inputs="ink", lr=lr, rounds=100)
        result, path = experiment(clients, cfg)
        rows.append({"config": asdict(cfg), "validation": result["history"][-1]["validation"],
                     "run": path.with_suffix(".json").name, "state": path.name})
    # Discard duplicate entries when this stage is resumed.
    rows = list({r["state"]: r for r in rows}.values())
    chosen_lr = max(rows, key=lambda x: x["validation"]["accuracy"])
    tuning["rows"], tuning["chosen"] = rows, chosen_lr
    (ROOT / "tuning_results.json").write_text(json.dumps(tuning, indent=2) + "\n")
    base = Config(**chosen_lr["config"])
    budgets = []
    for rounds in (100, 200, 300):
        cfg = replace(base, rounds=rounds)
        result, path = experiment(clients, cfg)
        budgets.append({"rounds": rounds, "validation": result["history"][-1]["validation"],
                        "run": path.with_suffix(".json").name, "state": path.name})
    chosen_budget = max(budgets, key=lambda x: x["validation"]["accuracy"])
    grid = []
    for num_clients in (5, 10):
        for epochs in (1, 3, 5):
            cfg = replace(base, rounds=chosen_budget["rounds"], clients=num_clients, epochs=epochs)
            result, path = experiment(clients, cfg)
            grid.append({"config": asdict(cfg), "train": result["history"][-1]["train"],
                         "validation": result["history"][-1]["validation"],
                         "run": path.with_suffix(".json").name, "state": path.name})
    chosen = max(grid, key=lambda x: x["validation"]["accuracy"])
    result = {"selection": "Choose learning rate, round budget, and C/E by validation accuracy only.",
              "budgets": budgets, "grid": grid, "chosen": chosen}
    (ROOT / "selected_config.json").write_text(json.dumps(result, indent=2) + "\n")
    print("FINAL CONFIG", json.dumps(chosen), flush=True)


def selected():
    return Config(**json.loads((ROOT / "selected_config.json").read_text())["chosen"]["config"])


def restore(path):
    net = model()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net


def ray_experiments(clients):
    import ray
    cfg = replace(selected(), clients=4, noise=0.0)
    serial, serial_path = experiment(clients, cfg, name="four-client-serial")
    ray.init(num_cpus=4, num_gpus=0, include_dashboard=False, logging_level="ERROR")
    try:
        parallel, parallel_path = experiment(clients, cfg, parallel=True, name="four-client-ray")
        error = max_difference(copy_state(restore(serial_path)), copy_state(restore(parallel_path)))
        curve_error = max(abs(a["validation"]["accuracy"] - b["validation"]["accuracy"])
                          for a, b in zip(serial["history"], parallel["history"]))
        assert error < 1e-6 and curve_error < 1e-9
        result = {"config": asdict(cfg), "serial_run": serial_path.with_suffix(".json").name,
                  "parallel_run": parallel_path.with_suffix(".json").name,
                  "state": parallel_path.name, "max_parameter_error": error,
                  "max_validation_curve_error": curve_error,
                  "serial_update_seconds": serial["update_seconds"],
                  "ray_update_seconds": parallel["update_seconds"], "cpus": 4}
        (ROOT / "ray_results.json").write_text(json.dumps(result, indent=2) + "\n")
        print("RAY MATCH", json.dumps(result), flush=True)
    finally:
        ray.shutdown()


def noise_experiments(clients):
    cfg = replace(selected(), clients=10, noise=0.0)
    records = {}
    for b in (0.0, 0.05, 0.1, 0.3, 0.6):
        records[str(b)] = []
        for seed in (42, 43, 44):
            setting = replace(cfg, noise=b, seed=seed)
            result, path = experiment(clients, setting)
            records[str(b)].append({"seed": seed, "state": path.name,
                "run": path.with_suffix(".json").name,
                "train": result["history"][-1]["train"],
                "clean_train": result["history"][-1]["clean_train"],
                "validation": result["history"][-1]["validation"]})
    means = {b: float(np.mean([r["validation"]["accuracy"] for r in values]))
             for b, values in records.items()}
    # State the rule before looking at test results: keep mean validation within 2 pp.
    eligible = [float(b) for b in records if float(b) > 0 and means[b] >= means["0.0"] - 0.02]
    choice = max(eligible) if eligible else 0.05
    summary = {"config": asdict(cfg), "records": records, "means": means,
        "choice": choice, "within_two_points": bool(eligible),
        "selection_rule": "Largest positive b within 2 percentage points of clean mean validation accuracy; if none qualifies, use the smallest tested positive b and report the loss."}
    (ROOT / "noise_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("NOISE CHOICE", json.dumps({"means": means, "choice": choice}), flush=True)


def wait_for_cpus(ray, count):
    for _ in range(150):
        if ray.available_resources().get("CPU", 0) >= count:
            return True
        time.sleep(0.1)
    return False


def stress_experiment(clients,cfg):
    """An expected numerical failure is evidence, not a missing result."""
    path=RUNS/"strong-noise-stress.json"
    source_hash=hashlib.sha256((ROOT/"federated.py").read_bytes()).hexdigest()
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get("config")==asdict(cfg) and previous.get("source_sha256")==source_hash:
            return previous
    net=model(cfg.seed)
    rng=np.random.default_rng(cfg.seed)
    images=noise_images(clients,cfg.noise,cfg.seed+81)
    result={"config":asdict(cfg),"source_sha256":source_hash,"status":"completed","completed_rounds":0,
            "failed_round":None,"model_change_on_rejection":None}
    for r in range(1,cfg.rounds+1):
        ids=rng.permutation(len(clients))[:cfg.clients]
        before=copy_state(net)
        try:
            serial_round(net,clients,ids,cfg,r,images)
        except FloatingPointError as error:
            result.update(status="non_finite_update",failed_round=r,error=str(error),
                          model_change_on_rejection=max_difference(before,copy_state(net)))
            assert result["model_change_on_rejection"]==0
            break
        result["completed_rounds"]=r
    value=evaluate_clients(net,clients,"val",cfg.inputs)
    if not np.isfinite(value["loss"]):
        value["loss"]=None
    result["last_finite_model_validation"]=value
    result["outside_fraction"]=sum(int(((x<0)|(x>1)).sum()) for x in images)/sum(x.numel() for x in images)
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("STRONG NOISE",json.dumps(result),flush=True)
    return result


def validations(clients):
    cfg = selected()
    evidence = {}
    evidence["split_overlap"] = sum(len(np.intersect1d(c["train_indices"], c["val_indices"])) for c in clients)
    assert evidence["split_overlap"] == 0
    net = model(77)
    before = copy_state(net)
    check = replace(cfg, clients=1, epochs=1, rounds=1, seed=77)
    serial_round(net, clients, [0], check, 1)
    expected = local_fit(before, clients[0]["x"], clients[0]["y"], inputs=check.inputs,
                         epochs=1, lr=check.lr, seed=client_seed(77, 1, 0))
    evidence["single_client_error"] = max_difference(copy_state(net), expected[1])
    assert evidence["single_client_error"] < 1e-6
    zero = model(78)
    before = copy_state(zero)
    serial_round(zero, clients, [0, 1], replace(cfg, lr=0.0), 1)
    evidence["zero_lr_error"] = max_difference(before, copy_state(zero))
    assert evidence["zero_lr_error"] < 1e-6
    updates = [local_fit(before, c["x"], c["y"], inputs=cfg.inputs, epochs=1,
                         lr=cfg.lr, seed=client_seed(78, 1, i)) for i,c in enumerate(clients[:2])]
    n0, n1 = updates[0][0], updates[1][0]
    explicit = {k: (n0 * updates[0][1][k] + n1 * updates[1][1][k]) / (n0+n1) for k in before}
    evidence["unequal_client_counts"] = [n0,n1]
    evidence["aggregation_error"] = max_difference(explicit, weighted_mean(updates))
    assert evidence["aggregation_error"] < 1e-6

    tiny_x, tiny_y = clients[0]["x"][:16], clients[0]["y"][:16]
    tiny = model(79)
    evidence["tiny_before"] = evaluate(tiny, tiny_x, tiny_y, cfg.inputs)
    _, tiny_state = local_fit(copy_state(tiny), tiny_x, tiny_y, inputs=cfg.inputs,
                              epochs=500, lr=cfg.lr, seed=79)
    tiny.load_state_dict(tiny_state)
    evidence["tiny_after"] = evaluate(tiny, tiny_x, tiny_y, cfg.inputs)
    assert evidence["tiny_after"]["accuracy"] >= 0.95

    # Break the image/label relation without changing any client's label histogram.
    shuffled = []
    rng = np.random.default_rng(2026)
    for c in clients:
        modified = c.copy()
        modified["y"] = c["y"][torch.from_numpy(rng.permutation(len(c["y"])))].clone()
        assert torch.equal(torch.bincount(modified["y"], minlength=62), torch.bincount(c["y"], minlength=62))
        shuffled.append(modified)
    shuffled_cfg = replace(cfg, rounds=100)
    shuffled_run, path = experiment(shuffled, shuffled_cfg, name="shuffled-label-control")
    selected_run = json.loads((RUNS / json.loads((ROOT / "selected_config.json").read_text())["chosen"]["run"]).read_text())
    evidence["shuffled_label_control"] = {
        "rounds": 100, "validation": shuffled_run["history"][-1]["validation"],
        "real_label_validation": selected_run["history"][99]["validation"], "run": path.with_suffix(".json").name}
    assert (evidence["shuffled_label_control"]["real_label_validation"]["accuracy"]
            > evidence["shuffled_label_control"]["validation"]["accuracy"] + 0.2)

    # Does a model fit its stored noise better than an independent draw on the same images?
    noise_records = json.loads((ROOT/"noise_results.json").read_text())
    chosen_record = json.loads((ROOT/"selected_config.json").read_text())["chosen"]
    if cfg.clients == 10:
        zero_state = restore(RUNS/noise_records["records"]["0.0"][0]["state"])
        chosen_state = restore(RUNS/chosen_record["state"])
        evidence["zero_noise_parameter_error"] = max_difference(copy_state(zero_state),copy_state(chosen_state))
        assert evidence["zero_noise_parameter_error"] < 1e-6
    fresh = {}
    for b, rows in noise_records["records"].items():
        row = next(r for r in rows if r["seed"] == 42)
        fresh_inputs = noise_images(clients,float(b),20261117)
        net = restore(RUNS/row["state"])
        fresh[b] = {"stored_noise_accuracy":row["train"]["accuracy"],
                    "fresh_noise_accuracy":evaluate_clients(net,clients,"train",cfg.inputs,fresh_inputs)["accuracy"],
                    "clean_train_accuracy":row["clean_train"]["accuracy"]}
    evidence["fresh_noise_evaluation"] = fresh

    b = 0.3
    noised = noise_images(clients, b, 123)
    delta = noised[0] - clients[0]["x"]
    evidence["laplace"] = {"b": b, "mean": float(delta.mean()), "mean_absolute": float(delta.abs().mean()),
                           "variance": float(delta.var(unbiased=True)), "expected_variance": 2*b*b}
    assert abs(float(delta.mean())) < 0.02*b
    assert abs(float(delta.abs().mean())-b) < 0.02*b
    assert abs(float(delta.var(unbiased=True))-2*b*b) < 0.04 * (2*b*b)
    zeros = noise_images(clients, 0, 123)
    assert all(torch.equal(c["x"], x) for c,x in zip(clients,zeros))
    rejected = []
    for bad in (-0.1, float("nan"), float("inf")):
        try:
            noise_images(clients,bad,123)
        except ValueError:
            rejected.append(str(bad))
    assert len(rejected) == 3
    evidence["invalid_noise_rejected"] = rejected
    # Only validation accuracy is inspected in this stress experiment.
    stress_cfg = replace(cfg, clients=10, noise=3.0)
    evidence["strong_noise"] = stress_experiment(clients,stress_cfg)
    stable, path = experiment(clients,replace(stress_cfg,lr=0.01),name="strong-noise-lower-rate")
    evidence["strong_noise_lower_rate"]={"lr":0.01,"rounds":cfg.rounds,
        "validation":stable["history"][-1]["validation"],"run":path.with_suffix(".json").name}

    import ray
    ray.init(num_cpus=1,num_gpus=0,include_dashboard=False,logging_level="ERROR")
    try:
        limited = model(85)
        serial = model(85)
        check = replace(cfg, clients=4, seed=85, epochs=1, rounds=1, noise=0.0)
        ids = np.random.default_rng(85).permutation(len(clients))[:4]
        serial_round(serial,clients,ids,check,1)
        batches = []
        ray_round(limited,clients,ids,check,1,audit=batches)
        evidence["one_cpu"] = {"batches": batches, "max_parameter_error": max_difference(copy_state(limited),copy_state(serial)),
                               "cpus_released": wait_for_cpus(ray,1)}
        assert batches == [1,1,1,1] and evidence["one_cpu"]["max_parameter_error"] < 1e-6
        before = copy_state(limited)
        failed = False
        try:
            ray_round(limited,clients,ids,check,2,fail_client=int(ids[1]))
        except ray.exceptions.RayTaskError:
            failed = True
        error = max_difference(before,copy_state(limited))
        released = wait_for_cpus(ray,1)
        assert failed and error == 0 and released
        serial_round(serial,clients,ids,check,2)
        ray_round(limited,clients,ids,check,2)
        recovery = max_difference(copy_state(limited),copy_state(serial))
        assert recovery < 1e-6 and wait_for_cpus(ray,1)
        evidence["actor_failure"] = {"injected_failure_raised": failed, "unchanged_model_error": error,
                                     "cpus_released": released, "successful_retry_error": recovery}
    finally:
        ray.shutdown()
    (ROOT / "validation_results.json").write_text(json.dumps(evidence,indent=2)+"\n")
    print("VALIDATION",json.dumps(evidence),flush=True)


def final_test(clients):
    # Every choice above is frozen on disk before this function opens the test file.
    choice = json.loads((ROOT / "selected_config.json").read_text())
    noise = json.loads((ROOT / "noise_results.json").read_text())
    ray = json.loads((ROOT / "ray_results.json").read_text())
    validation = json.loads((ROOT / "validation_results.json").read_text())
    x,y = load_test(ROOT/"Assignment3-data.zip")
    records = {}
    for b, rows in noise["records"].items():
        seed42 = next(row for row in rows if row["seed"]==42)
        records[b] = evaluate(restore(RUNS/seed42["state"]),x,y,noise["config"]["inputs"])
    main = evaluate(restore(RUNS/choice["chosen"]["state"]),x,y,choice["chosen"]["config"]["inputs"])
    parallel = evaluate(restore(RUNS/ray["state"]),x,y,ray["config"]["inputs"])
    result = {"samples":len(y), "sequential":main,"ray":parallel,"noise":records,
              "chosen_noise":noise["choice"],"selection_frozen_before_test":True,
              "frozen_selection_sha256": hashlib.sha256((ROOT/"selected_config.json").read_bytes()+(ROOT/"noise_results.json").read_bytes()).hexdigest()}
    (ROOT/"test_results.json").write_text(json.dumps(result,indent=2)+"\n")
    print("FINAL TEST",json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["tune", "select", "ray", "noise", "validate", "test", "all"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    clients = load_clients(ROOT / "Assignment3-data.zip")
    if args.stage == "tune":
        tune(clients)
    elif args.stage == "select":
        select(clients)
    elif args.stage == "ray":
        ray_experiments(clients)
    elif args.stage == "noise":
        noise_experiments(clients)
    elif args.stage == "validate":
        validations(clients)
    elif args.stage == "test":
        final_test(clients)
    elif args.stage == "all":
        tune(clients)
        select(clients)
        ray_experiments(clients)
        noise_experiments(clients)
        validations(clients)
        final_test(clients)
        from make_figures import main as make_figures
        make_figures()


if __name__ == "__main__":
    main()
