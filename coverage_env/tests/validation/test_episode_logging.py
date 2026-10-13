"""Validate transition snapshots and JSONL episode export."""

import json
import tempfile
from pathlib import Path

from coverage_env.tests.validation.support import (
    ValidationTestCase, capture_step, make_action, make_environment,
)


class EpisodeLoggingTests(ValidationTestCase):
    def test_log_contains_transition_and_can_be_exported(self):
        environment = make_environment()
        action = make_action()
        before, result, after = capture_step(environment, action)
        transitions = environment.episode_logger.transitions
        logged = transitions[0] if transitions else None
        json_before = json.loads(json.dumps(before))
        json_after = json.loads(json.dumps(after))
        self.assert_transition(
            logged is not None
            and logged["step"] == 0
            and logged["before_state"] == json_before
            and logged["action"]["field_id"] == action.field_id
            and logged["result"]["distance"] == result["distance"]
            and logged["after_state"] == json_after,
            before=before, action=action, result=result, after=after,
            message="Episode log does not contain the expected transition snapshot.",
        )

        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[2]) as temporary_directory:
            output = Path(temporary_directory) / "episode.jsonl"
            environment.episode_logger.save_jsonl(output)
            lines = output.read_text(encoding="utf-8").splitlines()
            parsed = json.loads(lines[0]) if lines else None
            self.assert_transition(
                parsed == logged,
                before=before, action=action, result=result, after=after,
                message="JSONL episode export differs from the in-memory transition.",
            )

        environment.reset()
        self.assertEqual(environment.episode_logger.transitions, ())
