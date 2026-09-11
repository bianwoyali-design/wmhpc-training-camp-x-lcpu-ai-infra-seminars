"""CPU regression tests for report comparisons and the command-line entry."""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from report import best_table, render


class Reports(unittest.TestCase):
    def test_best_candidate_does_not_use_official_or_other_stage(self):
        rows = []
        for stage, c, method, us in [
            ("gram", 16, "wmma_w1", 10),
            ("gram", 16, "official", 1),
            ("gram", 32, "wmma_w4", 40),
            ("projection", 16, "wmma_w1", 100),
        ]:
            rows.append(
                dict(
                    workload="single",
                    chunk=str(c),
                    stage=stage,
                    method=method,
                    median_us=str(us),
                )
            )
        text = "\n".join(best_table(rows, stage=True))
        self.assertIn("4.00×", text)
        self.assertNotIn("| official |", text)

    def test_complete_result(self):
        text = render(ROOT / "results/benchmark")
        for suite in ("range", "inverse", "mma"):
            self.assertIn(f"## {suite}", text)
        self.assertIn("2032", text)

    def test_cli_without_torch(self):
        spec = importlib.util.spec_from_file_location("experiment_cli", ROOT / "run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module.parser().parse_args(
            ["reproduce", "--suite", "inverse", "--warps", "8,1,4", "--quick"]
        )
        self.assertEqual(args.warps, [1, 4, 8])
        self.assertTrue(module.settings(args)["quick"])


if __name__ == "__main__":
    unittest.main()
