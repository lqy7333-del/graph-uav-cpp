"""Fixed-code value/return gate: paired short A/B run, resume, closure and GAE checks.

No algorithm change and no model selection: A and B use seed 123 and 5 updates
only to verify that the corrected bootstrap state path is exercised end to end.
The formal 400-update runs are started only after this gate passes.
"""
from __future__ import annotations

import csv, json, math, os, shutil, subprocess, sys, time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coverage_env.config import load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.checkpoint import evaluation_config, parameter_sha256
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.ppo.loss import compute_advantage
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.trainer.distributed import resolve_config
from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice

OUT = REPO / "coverage_env/results/maneuver_time/value_gate"
PYTHON = sys.executable
SEED = 123
UPDATES = 5
ROLLOUT = 128
CHECKPOINT_EVERY = 1
TOLERANCE = 1e-2


def overrides(mode):
    return [
        f"rl.reward.mode={'distance' if mode == 'A' else 'time'}",
        "rl.decoder.candidate_attention=false",
        "ppo.randomize_uav=false",
        "ppo.gamma=1.0",
        "environment.num_fields.min=5",
        "environment.num_fields.max=20",
        "environment.num_fields_strategy=uniform",
        "uav.speed=8.0",
        "uav.coverage_width=10.0",
        "rl.reward.coverage_width=10.0",
        "rl.reward.nominal_work_speed=8.0",
        "rl.reward.time_scale=100.0",
        "rl.time.cruise_speed=8.0",
        "rl.time.tau_pi=3.0",
        "rl.time.exponent=2.0",
        "rl.time.service_time=0.0",
    ]


def config_for(mode):
    args = SimpleNamespace(num_fields=None, set=overrides(mode), ppo_epochs=None,
                           batch_size=None, gae_lambda=None)
    return resolve_config(args)


def initial_hash(mode):
    torch.manual_seed(SEED)
    return parameter_sha256(CoveragePolicy(None, config=config_for(mode)))


def run_command(mode, out_dir, resume=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(REPO / "train_ddp.py"),
        "--num-updates", str(UPDATES), "--rollout-steps", str(ROLLOUT),
        "--seed", str(SEED), "--checkpoint-every", str(CHECKPOINT_EVERY),
        "--checkpoint-dir", str(out_dir),
    ]
    for item in overrides(mode):
        cmd += ["--set", item]
    if resume is not None:
        cmd += ["--resume", str(resume), "--resume-mode", "exact"]
    log = out_dir / "train.log"
    started = time.time()
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            cmd, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(REPO)},
        )
    if completed.returncode:
        raise RuntimeError(f"{mode} short run failed; see {log}")
    return {"command": cmd, "log": str(log), "wall_seconds": time.time() - started}


def read_rows(directory):
    with (directory / "train_log.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finite_rows(directory):
    rows = read_rows(directory)
    bad = []
    for row in rows:
        for key, value in row.items():
            if value in ("", None):
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if not math.isfinite(number):
                bad.append((row.get("update"), key, value))
    return rows, bad


def load_agent(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = evaluation_config(saved)
    environment = InstanceGenerator(config).generate(
        num_fields=5, seed=40000, randomize_each_reset=False, randomize_uav=False)
    actor = CoveragePolicy(environment, config=config)
    agent = PPOAgent(actor, config, device="cpu")
    agent.load_checkpoint(checkpoint, map_location="cpu", restore_rng=False,
                          load_optimizers=False)
    agent.actor.eval(); agent.critic.eval()
    return agent, config


def reward_closure(checkpoint):
    agent, config = load_agent(checkpoint)
    rows = []
    for scale, seed in ((5, 40000), (10, 40030), (20, 40090)):
        environment = InstanceGenerator(config).generate(
            num_fields=scale, seed=seed, randomize_each_reset=False,
            randomize_uav=False)
        state = environment.reset()
        done = False; info = {}; total_reward = 0.0; total_time = 0.0
        total_area = 0.0; failed = 0
        for _ in range(400):
            with torch.no_grad():
                action = CoveragePolicy.greedy_action(
                    agent.actor(state, environment=environment))
            state, reward, done, info = environment.step(action)
            total_reward += float(reward)
            total_time += float(info.get("time_total", 0.0) or 0.0)
            total_area += float(info.get("coverage_gain", 0.0) or 0.0)
            failed += int(bool(info.get("failed")))
            if done:
                break
        expected = -(total_time - total_area / 80.0) / 100.0 - failed
        rows.append({
            "checkpoint": str(checkpoint), "scale": scale, "seed": seed,
            "completed": bool(done and info.get("termination_reason") == "mission_complete"),
            "failed_steps": failed, "sum_reward": total_reward,
            "sum_time": total_time, "sum_area": total_area,
            "expected_reward": expected, "closure_error": total_reward - expected,
        })
    return rows


def independent_gae(rewards, values, next_values, dones, truncated, gamma, lam):
    adv = 0.0; out = [0.0] * len(rewards)
    for i in range(len(rewards) - 1, -1, -1):
        delta = rewards[i] + gamma * next_values[i] * (0.0 if dones[i] else 1.0) - values[i]
        boundary = bool(dones[i] or truncated[i])
        adv = delta + (0.0 if boundary else gamma * lam * adv)
        out[i] = adv
    return out


def gae_check(checkpoint):
    agent, config = load_agent(checkpoint)
    environment = InstanceGenerator(config).generate(
        num_fields=5, seed=40000, randomize_each_reset=False, randomize_uav=False)
    torch.manual_seed(SEED)
    state = environment.reset()
    rewards=[]; values=[]; next_values=[]; dones=[]; truncated=[]; headings=[]
    for _ in range(ROLLOUT):
        with torch.no_grad():
            output = agent.actor(state, environment=environment)
            distribution = output["action_distribution"]
            action = CoveragePolicy.greedy_action(output)
            value = float(agent.value_from_output(output).item())
        next_state, reward, done, info = environment.step(action)
        graph = agent.actor.graph_builder.build_graph(environment)
        next_value = 0.0 if done else agent.value_for_graph(
            graph, environment, state=next_state)
        rewards.append(float(reward)); values.append(value)
        next_values.append(float(next_value)); dones.append(bool(done))
        truncated.append(False)
        headings.append(bool(state["uavs"]["UAV-1"]["heading_valid"]))
        state = environment.reset() if done else next_state
    if not dones[-1]:
        truncated[-1] = True
    gamma=float(agent.gamma); lam=float(agent.gae_lambda)
    adv_ref,ret_ref=compute_advantage(rewards,values,next_values,dones,gamma,lam,
                                      device="cpu",truncated=truncated)
    adv_ref=[float(x) for x in adv_ref]; ret_ref=[float(x) for x in ret_ref]
    adv_ind=independent_gae(rewards,values,next_values,dones,truncated,gamma,lam)
    gaps=[abs(next_values[i]-values[i+1]) for i in range(len(values)-1) if not dones[i] and not truncated[i]]
    return {
        "checkpoint": str(checkpoint), "n": len(rewards),
        "gaE_max_abs_diff": max(abs(a-b) for a,b in zip(adv_ref,adv_ind)),
        "return_max_abs_diff": max(abs(a-b) for a,b in zip(ret_ref,[a+v for a,v in zip(adv_ind,values)])),
        "next_value_gap_max": max(gaps) if gaps else 0.0,
        "terminal_transitions": sum(dones), "truncated_tail": bool(truncated[-1]),
        "heading_valid_fraction": sum(headings)/len(headings),
        "ret_min": min(ret_ref), "ret_max": max(ret_ref),
    }


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    report={"seed":SEED,"updates":UPDATES,"rollout_steps":ROLLOUT,
            "tolerance":TOLERANCE,"runs":{},"initial_hashes":{}}
    for mode in ("A","B"):
        report["initial_hashes"][mode]=initial_hash(mode)
    if report["initial_hashes"]["A"] != report["initial_hashes"]["B"]:
        raise SystemExit("paired A/B initial hashes differ")
    for mode in ("A","B"):
        directory=OUT/f"{mode}_seed{SEED}"
        report["runs"][mode]=run_command(mode,directory)
        rows,bad=finite_rows(directory)
        if len(rows)!=UPDATES: raise SystemExit(f"{mode}: expected {UPDATES} rows")
        gaps=[float(r["value_state_gap_max"]) for r in rows]
        report["runs"][mode].update({"rows":len(rows),"nonfinite":bad,
            "value_state_gap_max":max(gaps),"value_state_gap_values":gaps,
            "checkpoints":sorted(p.name for p in directory.glob("ckpt_update_*.pt"))})
        if bad: raise SystemExit(f"{mode}: nonfinite metrics")
        if max(gaps)>TOLERANCE: raise SystemExit(f"{mode}: value gap {max(gaps)} > {TOLERANCE}")
    # exact resume of A from update 2 to 5
    resume_dir=OUT/"A_resume_from_2"
    report["runs"]["A_resume"]=run_command("A",resume_dir,OUT/f"A_seed{SEED}/ckpt_update_2.pt")
    rows,bad=finite_rows(resume_dir)
    gaps=[float(r["value_state_gap_max"]) for r in rows]
    config_a=json.loads((OUT/f"A_seed{SEED}/resolved_config_update_1.json").read_text())["resolved_config"]
    config_r=json.loads((resume_dir/"resolved_config_update_3.json").read_text())["resolved_config"]
    if config_a!=config_r: raise SystemExit("resume changed resolved config")
    if [int(r["update"]) for r in rows] != [3,4,5]: raise SystemExit("resume did not start at 3")
    report["runs"]["A_resume"].update({"rows":len(rows),"nonfinite":bad,
        "value_state_gap_max":max(gaps),"updates":[int(r["update"]) for r in rows],
        "config_equal":True})
    if bad: raise SystemExit("resume: nonfinite metrics")
    if max(gaps)>TOLERANCE: raise SystemExit(f"resume: value gap {max(gaps)} > {TOLERANCE}")
    # Time-mode reward closure on B; A uses the distance objective and is not
    # tested with the time formula.
    closure=reward_closure(OUT/f"B_seed{SEED}/ckpt_update_{UPDATES}.pt")
    report["reward_closure"]=closure
    max_closure=max(abs(r["closure_error"]) for r in closure)
    if max_closure>1e-6: raise SystemExit(f"reward closure error {max_closure}")
    gae=[]
    for mode in ("A","B"):
        gae.append(gae_check(OUT/f"{mode}_seed{SEED}/ckpt_update_{UPDATES}.pt"))
    report["gae"]=gae
    for item in gae:
        if item["gaE_max_abs_diff"]>2e-4 or item["return_max_abs_diff"]>2e-4:
            raise SystemExit(f"GAE recompute mismatch: {item}")
        if item["next_value_gap_max"]>TOLERANCE:
            raise SystemExit(f"GAE next-value gap: {item}")
    report["passed"]=True
    (OUT/"value_gate_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    lines=["# Corrected value/return gate","",
           f"- seed {SEED}, {UPDATES} updates × {ROLLOUT} steps, paired A/B; no NN comparison",
           f"- initial A/B hash: `{report['initial_hashes']['A']}`",
           "",
           "| run | rows | max value_state_gap | checkpoints |",
           "|---|---:|---:|---:|"]
    for mode in ("A","B"):
        r=report["runs"][mode]
        lines.append(f"| {mode} | {r['rows']} | {r['value_state_gap_max']:.3e} | {len(r['checkpoints'])} |")
    r=report["runs"]["A_resume"]
    lines.append(f"| A resume 2→5 | {r['rows']} | {r['value_state_gap_max']:.3e} | updates {r['updates']} |")
    lines += ["", "B time-mode reward closure:", "",
              "| checkpoint | scale | completed | failed | sum reward | expected | error |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for row in closure:
        lines.append(f"| {Path(row['checkpoint']).parent.name} | {row['scale']} | {row['completed']} | "
                     f"{row['failed_steps']} | {row['sum_reward']:.6f} | {row['expected_reward']:.6f} | "
                     f"{row['closure_error']:.2e} |")
    lines += ["", "| checkpoint | GAE max diff | return max diff | next-value gap max | terminal | truncated tail |",
              "|---|---:|---:|---:|---:|---:|"]
    for item in gae:
        lines.append(f"| {Path(item['checkpoint']).parent.name} | {item['gaE_max_abs_diff']:.2e} | "
                     f"{item['return_max_abs_diff']:.2e} | {item['next_value_gap_max']:.2e} | "
                     f"{item['terminal_transitions']} | {item['truncated_tail']} |")
    lines += ["", "Gate passed: **True**. No training branch beyond the paired short gate was started."]
    (OUT/"value_gate_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"passed":True,"gap_A":report["runs"]["A"]["value_state_gap_max"],
                      "gap_B":report["runs"]["B"]["value_state_gap_max"],
                      "resume_gap":r["value_state_gap_max"],
                      "closure_max":max_closure,
                      "gae":[{k:v for k,v in x.items() if k not in ("checkpoint",)} for x in gae]},indent=1))

if __name__=="__main__": main()
