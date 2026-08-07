from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.report_reproduction import (
    ReproductionDependencies,
    ReproductionNode,
    _dry_run_exit_code,
    _execute_single_node,
    _print_gate_action_instruction,
    plan_reproduction_nodes,
    run_report_reproduction,
)
from buffalo_weight.reproduction_config import load_report_contract
from tests.fake_report_provenance import FixedReportProvenance
from tests.report_inputs_fixture import CuratedInputsFixture


class ReportReproductionUnitTest(unittest.TestCase):
    def test_reproduction_node_fields(self) -> None:
        node = ReproductionNode("inputs", "stage", "reusable", "snapshot intact")
        self.assertEqual(node.name, "inputs")
        self.assertEqual(node.node_type, "stage")
        self.assertEqual(node.action, "reusable")
        self.assertEqual(node.reason, "snapshot intact")

    def test_dry_run_exit_code_returns_one_when_any_node_blocked(self) -> None:
        nodes = [
            ReproductionNode("inputs", "stage", "reusable", "ok"),
            ReproductionNode("confirm-features", "gate", "blocked", "missing confirmation"),
        ]
        stdout = io.StringIO()
        code = _dry_run_exit_code(nodes, stdout)
        self.assertEqual(code, 1)
        self.assertIn("reproduction: blocked by 1 node(s)", stdout.getvalue())

    def test_dry_run_exit_code_returns_zero_when_no_node_blocked(self) -> None:
        nodes = [
            ReproductionNode("inputs", "stage", "rebuild", "absent"),
            ReproductionNode("feature-selection", "stage", "rebuild", "upstream rebuild"),
            ReproductionNode("confirm-features", "gate", "released", "ok"),
        ]
        stdout = io.StringIO()
        code = _dry_run_exit_code(nodes, stdout)
        self.assertEqual(code, 0)
        self.assertIn("reproduction: plan valid", stdout.getvalue())

    def test_gate_action_instruction_formatting(self) -> None:
        for gate_name in ("confirm-features", "confirm-approach", "confirm-diagnostics", "unknown"):
            with self.subTest(gate_name=gate_name):
                stdout = io.StringIO()
                _print_gate_action_instruction(gate_name, stdout)
                self.assertIn("Action required:", stdout.getvalue())

    def test_execute_single_node_gate_blocked_returns_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            contract = load_report_contract(fixture.config_path)
            node = ReproductionNode("confirm-features", "gate", "blocked", "missing confirmation")
            stdout = io.StringIO()
            code = _execute_single_node(contract, node, ReproductionDependencies(), stdout)
            self.assertEqual(code, 1)
            self.assertIn("blocked: confirm-features -> missing confirmation", stdout.getvalue())

    def test_execute_single_node_stage_reusable_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CuratedInputsFixture(Path(directory))
            contract = load_report_contract(fixture.config_path)
            node = ReproductionNode("inputs", "stage", "reusable", "intact")
            stdout = io.StringIO()
            code = _execute_single_node(contract, node, ReproductionDependencies(), stdout)
            self.assertEqual(code, 0)
            self.assertIn("inputs: reusable", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
