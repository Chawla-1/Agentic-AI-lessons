import os
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v15_parallel_execution import (
    find_ready_wavefront,
    calculator,
    MAX_CONCURRENT_WORKERS,
)


class TestV15ParallelExecution(unittest.TestCase):
    """
    Automated regression tests for V15 parallel tool execution and Wavefront DAG scheduling.
    """

    def test_find_ready_wavefront_ordering(self):
        """Verify topological discovery discovers independent steps and respects dependencies."""
        steps = [
            {"id": 1, "task": "Task A", "status": "PENDING", "depends_on": []},
            {"id": 2, "task": "Task B", "status": "PENDING", "depends_on": []},
            {"id": 3, "task": "Task C", "status": "PENDING", "depends_on": [1, 2]},
            {"id": 4, "task": "Task D", "status": "PENDING", "depends_on": [3]},
        ]

        # Wave 1: Steps 1 & 2 have no dependencies
        completed_ids = set()
        wave1 = find_ready_wavefront(steps, completed_ids)
        self.assertEqual([s["id"] for s in wave1], [1, 2])

        # After step 1 completes, step 3 is NOT ready yet because step 2 is pending
        completed_ids.add(1)
        steps[0]["status"] = "COMPLETED"
        wave1_partial = find_ready_wavefront(steps, completed_ids)
        self.assertEqual([s["id"] for s in wave1_partial], [2])

        # After step 2 completes, step 3 becomes ready
        completed_ids.add(2)
        steps[1]["status"] = "COMPLETED"
        wave2 = find_ready_wavefront(steps, completed_ids)
        self.assertEqual([s["id"] for s in wave2], [3])

        # After step 3 completes, step 4 becomes ready
        completed_ids.add(3)
        steps[2]["status"] = "COMPLETED"
        wave3 = find_ready_wavefront(steps, completed_ids)
        self.assertEqual([s["id"] for s in wave3], [4])

    def test_parallel_speedup(self):
        """Verify concurrent execution of independent workloads achieves measurable speedup."""
        def simulated_task(duration: float):
            time.sleep(duration)
            return "done"

        delay = 0.2
        num_tasks = 3

        # Concurrent execution
        start_concurrent = time.perf_counter()
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
            futures = [executor.submit(simulated_task, delay) for _ in range(num_tasks)]
            results = [f.result() for f in futures]
        concurrent_elapsed = time.perf_counter() - start_concurrent

        self.assertEqual(len(results), num_tasks)
        # Running 3 x 0.2s concurrently should finish in ~0.25s, well under 3 * 0.2 = 0.6s
        self.assertLess(concurrent_elapsed, 0.45, f"Expected concurrency speedup, took {concurrent_elapsed:.2f}s")

    def test_ast_calculator_safety(self):
        """Verify AST evaluator computes arithmetic correctly and strictly blocks malicious inputs."""
        res = calculator("20 * 40 + 60 - 20 / 5")
        self.assertEqual(res, "856.0")

        # Unsupported / malicious expressions return an error string
        res_bad1 = calculator("__import__('os').system('dir')")
        self.assertTrue(res_bad1.startswith("ERROR:"), f"Expected ERROR, got {res_bad1}")

        res_bad2 = calculator("[x for x in (1, 2)]")
        self.assertTrue(res_bad2.startswith("ERROR:"), f"Expected ERROR, got {res_bad2}")


if __name__ == "__main__":
    unittest.main()
