import sys
import types
import unittest


try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_stub.device = lambda value: value
    sys.modules["torch"] = torch_stub

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModelForCausalLM = object
    transformers_stub.AutoTokenizer = object
    transformers_stub.GenerationConfig = object
    sys.modules["transformers"] = transformers_stub

from src.llm.validation import validate_generated_arguments


class YelpArgumentContractRegressionTests(unittest.TestCase):
    def setUp(self):
        self.example = {
            "user_id": "U1",
            "history": [{"name": "History Place"}],
            "target_item": {"name": "Target Place"},
        }

    def _payload(self, evidence_count: int = 1):
        arguments = []
        for index in range(4):
            arg_type = "support" if index < 2 else "attack"
            arguments.append(
                {
                    "id": f"A{index + 1}",
                    "type": arg_type,
                    "text": f"Argument {index + 1}",
                    "used_aspects": ["service"],
                    "aspect_effect": (
                        "present_preferred"
                        if arg_type == "support"
                        else "present_disliked"
                    ),
                    "evidence": [
                        f"Target Place | evidence {item + 1}"
                        for item in range(evidence_count)
                    ],
                }
            )
        return {"arguments": arguments}

    def test_balanced_four_argument_contract(self):
        validation = validate_generated_arguments(
            self.example,
            self._payload(),
            argument_mode="balanced",
        )
        self.assertTrue(validation["is_valid"])

    def test_current_contract_allows_three_evidence_items(self):
        validation = validate_generated_arguments(
            self.example,
            self._payload(evidence_count=3),
            argument_mode="balanced",
        )
        self.assertTrue(validation["is_valid"])

    def test_current_contract_rejects_four_evidence_items(self):
        validation = validate_generated_arguments(
            self.example,
            self._payload(evidence_count=4),
            argument_mode="balanced",
        )
        self.assertIn(
            "too_many_evidence_items",
            validation["error_counts"],
        )


if __name__ == "__main__":
    unittest.main()
