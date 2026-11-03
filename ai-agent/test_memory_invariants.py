import os
import sys
import unittest
from pydantic import ValidationError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v15_parallel_execution import (
    FactExtractionItem,
    FactExtractionPayload,
    ConflictDecision,
    MEMORY_LABELS,
)


class TestMemoryInvariants(unittest.TestCase):
    """
    Automated regression tests for memory category invariants and schema contracts.
    """

    def test_category_normalization_and_aliases(self):
        """Verify profile categories and aliases normalize to canonical MEMORY_LABELS."""
        item1 = FactExtractionItem(category="hobby", item="chess")
        self.assertEqual(item1.category, "hobbies")

        item2 = FactExtractionItem(category="interest", item="guitar")
        self.assertEqual(item2.category, "hobbies")

        item3 = FactExtractionItem(category="city", item="Bangalore")
        self.assertEqual(item3.category, "location")

        item4 = FactExtractionItem(category="languages", item="Python")
        self.assertEqual(item4.category, "programming_languages")

    def test_rejection_of_non_biographical_categories(self):
        """Verify non-biographical conversational items raise validation errors."""
        forbidden_categories = ["greeting", "inquiry", "math", "calculation", "systemaction", "summary"]
        for cat in forbidden_categories:
            with self.assertRaises(ValidationError, msg=f"Category '{cat}' should be rejected"):
                FactExtractionItem(category=cat, item="test")

    def test_rejection_of_empty_items(self):
        """Verify empty atomic values are rejected by minimum length constraints."""
        with self.assertRaises(ValidationError):
            FactExtractionItem(category="hobbies", item="")

    def test_conflict_decision_schema(self):
        """Verify conflict resolution decision contracts enforce valid actions."""
        d1 = ConflictDecision(action="REPLACE", reason="User moved to a new city")
        self.assertEqual(d1.action, "REPLACE")

        d2 = ConflictDecision(action="KEEP_BOTH", reason="Distinct projects")
        self.assertEqual(d2.action, "KEEP_BOTH")

        with self.assertRaises(ValidationError):
            ConflictDecision(action="INVALID_ACTION", reason="Test")


if __name__ == "__main__":
    unittest.main()
