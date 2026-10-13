"""Run 30 complete heuristic-baseline episodes at 5/10/20 fields."""

from __future__ import annotations

import json
from math import isclose
from pathlib import Path
import shutil
import unittest

import matplotlib

matplotlib.use("Agg")

from coverage_env.baseline.nearest_neighbor import NearestNeighborAgent
from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.visualization.episode_visualizer import EpisodeVisualizer


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=repr)


def _transition_failure(before, action, result, after, reason) -> str:
    return (
        f"{reason}\nbefore state:\n{_dump(before)}\n"
        f"action:\n{_dump(action)}\nresult:\n{_dump(result)}\n"
        f"after state:\n{_dump(after)}"
    )


class NearestNeighborEpisodeTests(unittest.TestCase):
    """Integration-level task completion and state/feedback validation."""

    def test_5_10_20_fields_x_10_episodes(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        generator = InstanceGenerator(config)
        output_dir = Path(__file__).parents[1] / config["plot"]["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        all_case_metrics = {}
        failures: list[str] = []

        for field_count in (5, 10, 20):
            episode_metrics = []
            case_visual_environment = None
            for episode_index in range(10):
                seed = int(config["environment"]["random_seed"])
                seed += field_count * 100 + episode_index
                environment = generator.generate(
                    num_fields=field_count, seed=seed
                )
                initial_state = environment.reset()
                area_config = config["field_generator"]["field_size"]
                geometry_report = [
                    {
                        "field_id": field.field_id,
                        "area": field.area,
                        "valid": field.xy_polygon.is_valid,
                        "grid_points": len(field.grid_points),
                    }
                    for field in environment.fields
                ]
                geometry_ok = all(
                    field.xy_polygon.is_valid
                    and float(area_config["min_area"]) <= field.area
                        <= float(area_config["max_area"])
                    and bool(field.grid_points)
                    for field in environment.fields
                )
                self.assertTrue(
                    geometry_ok,
                    _transition_failure(
                        initial_state, {"validation": "generated field geometry"},
                        geometry_report, initial_state,
                        "Generated field is invalid, out of configured area bounds, or undiscretized.",
                    ),
                )
                agent = NearestNeighborAgent(environment, config)
                state = initial_state
                failed_configuration_count = 0
                invalid_configuration_count = 0
                episode_error = None
                step_limit = field_count * int(
                    config["environment"].get("max_episode_steps_per_field", 12)
                )
                step_number = 0

                while any(
                    field_state["uncovered_points"] > 0
                    for field_state in state["fields"].values()
                ) and step_number < step_limit:
                    action = agent.select_action(state)
                    if action is None:
                        failed_configuration_count += 1
                        episode_error = agent.last_failure or "Agent returned no action before completion."
                        break

                    before = state
                    result = None
                    try:
                        result = environment.step_configuration(action)
                    except Exception as exc:  # captured as a failed configuration with full state
                        failed_configuration_count += 1
                        after = environment.get_state()
                        episode_error = _transition_failure(
                            before, action,
                            f"{type(exc).__name__}: {exc}", after,
                            "Configuration execution raised an exception.",
                        )
                        break

                    after = environment.get_state()
                    step_number += 1
                    field = environment.get_field(action.field_id)
                    path = result["path"]
                    expected_distance = sum(
                        ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
                        for start, end in zip(path[:-1], path[1:])
                    )
                    action_valid = (
                        action.field_id in after["fields"]
                        and any(b.id == action.entry_boundary_id for b in field.boundaries)
                        and any(b.id == action.exit_boundary_id for b in field.boundaries)
                        and 0.0 < action.workload_ratio <= 1.0
                    )
                    state_updated = (
                        result["state"] == after
                        and after["uavs"][environment.uavs[0].uav_id]["position"]
                            == result["exit_point"]
                        and after["fields"][field.field_id]["covered_area"]
                            > before["fields"][field.field_id]["covered_area"]
                    )
                    energy_valid = (
                        0.0 <= result["energy"]
                            <= before["uavs"][environment.uavs[0].uav_id]["remaining_energy"]
                        and after["uavs"][environment.uavs[0].uav_id]["remaining_energy"] >= 0.0
                        and isclose(
                            result["energy"],
                            environment.uavs[0].alpha * result["distance"]
                            + environment.uavs[0].beta * result["coverage"],
                            rel_tol=1e-9,
                            abs_tol=1e-7,
                        )
                    )
                    path_valid = (
                        len(path) >= 2
                        and path[0] == before["uavs"][environment.uavs[0].uav_id]["position"]
                        and path[-1] == result["exit_point"]
                        and all(
                            len(point) == 2
                            and all(abs(float(coordinate)) < float("inf") for coordinate in point)
                            for point in path
                        )
                        and isclose(expected_distance, result["distance"], abs_tol=1e-6)
                    )
                    if not action_valid:
                        invalid_configuration_count += 1
                    if not (action_valid and state_updated and energy_valid and path_valid):
                        episode_error = _transition_failure(
                            before, action, result, after,
                            "Action validity, state update, energy, or path invariant failed.",
                        )
                        break
                    state = after

                environment.finalize_episode()

                summary = environment.get_summary()
                if environment.episode_logger.transitions:
                    log_path = output_dir / "logs" / (
                        f"episode_{field_count:02d}_fields_{episode_index + 1:02d}.jsonl"
                    )
                    environment.episode_logger.save_jsonl(log_path)
                completed_fields = [
                    field.field_id for field in environment.fields
                    if not field.get_uncovered_points()
                ]
                complete = len(completed_fields) == field_count
                if step_number >= step_limit and not complete:
                    episode_error = f"Episode exceeded step limit {step_limit}."
                    failed_configuration_count += 1
                if not complete and episode_error is None:
                    episode_error = "Episode stopped before every field was covered."
                    failed_configuration_count += 1

                metric = {
                    "seed": seed,
                    "steps": step_number,
                    "completed_fields": len(completed_fields),
                    "total_distance": summary["total_distance"],
                    "total_energy": summary["total_energy"],
                    "coverage_ratio": summary["coverage_ratio"],
                    "remaining_energy": summary["remaining_energy"][environment.uavs[0].uav_id],
                    "failed_configuration_count": failed_configuration_count,
                    "invalid_configuration_count": invalid_configuration_count,
                    "complete": complete,
                }
                episode_metrics.append(metric)
                if episode_error:
                    failures.append(
                        f"field_count={field_count}, episode={episode_index + 1}: "
                        f"{episode_error}"
                    )
                    print(f"FAIL field_count={field_count}, episode={episode_index + 1}\n{episode_error}")
                else:
                    print(
                        f"PASS episode={episode_index + 1}, fields={field_count}, "
                        f"completed={len(completed_fields)}/{field_count}, "
                        f"distance={summary['total_distance']:.1f} m, "
                        f"energy={summary['total_energy']:.1f} Wh, "
                        f"coverage={summary['coverage_ratio']:.1%}, "
                        f"remaining_energy={metric['remaining_energy']:.1f} Wh, "
                        f"failed_configurations={failed_configuration_count}, "
                        f"invalid_configurations={invalid_configuration_count}"
                    )
                if episode_index == 9:
                    case_visual_environment = environment if complete else None

            all_case_metrics[field_count] = episode_metrics
            successful = sum(metric["complete"] for metric in episode_metrics)
            self.assertEqual(
                successful, 10,
                f"Case {field_count} fields completed {successful}/10 episodes. "
                f"Errors: {failures[-10:]}",
            )
            self.assertTrue(all(
                metric["failed_configuration_count"] == 0
                and metric["invalid_configuration_count"] == 0
                and metric["remaining_energy"] >= 0.0
                for metric in episode_metrics
            ), f"Constraint failures in {field_count}-field case: {episode_metrics}")

            if case_visual_environment is not None and config["plot"]["save_result"]:
                case_path = output_dir / f"episode_result_{field_count}_fields.png"
                EpisodeVisualizer().plot(
                    case_visual_environment,
                    show=bool(config["plot"].get("show", False)),
                    save_path=case_path,
                )
                if field_count == 20:
                    shutil.copyfile(case_path, output_dir / "episode_result.png")

            distances = [metric["total_distance"] for metric in episode_metrics]
            energies = [metric["total_energy"] for metric in episode_metrics]
            coverages = [metric["coverage_ratio"] for metric in episode_metrics]
            print(
                f"CASE PASS fields={field_count}, episodes=10/10, "
                f"mean_steps={sum(metric['steps'] for metric in episode_metrics) / 10:.1f}, "
                f"mean_distance={sum(distances) / 10:.1f} m, "
                f"mean_energy={sum(energies) / 10:.1f} Wh, "
                f"mean_coverage={sum(coverages) / 10:.1%}, "
                f"failed_configurations={sum(m['failed_configuration_count'] for m in episode_metrics)}, "
                f"invalid_configurations={sum(m['invalid_configuration_count'] for m in episode_metrics)}"
            )

        self.assertFalse(failures, "One or more heuristic episodes failed:\n" + "\n".join(failures))
        total_episodes = sum(len(metrics) for metrics in all_case_metrics.values())
        print(
            f"\nMISSION VALIDATION PASS: {total_episodes}/{total_episodes} episodes; "
            "all fields completed; no invalid actions or energy violations."
        )


if __name__ == "__main__":
    unittest.main()
