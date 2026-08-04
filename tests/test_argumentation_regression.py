import sys
import types
import unittest


# The current package initializers eagerly import optional local-LLM dependencies
# even when only the dependency-free DF-QuAD core is needed. Keep this regression
# suite runnable in a minimal environment without changing the Yelp package yet.
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

from src.argumentation.dfquad import aggregate_strength, evaluate_root_dfquad
from src.argumentation.graph_builder import build_argument_graph
from src.argumentation.schema import Argument
from src.llm.validation import validate_generated_arguments


class DFQuADRegressionTests(unittest.TestCase):
    def test_dfquad_aggregation_and_combination(self):
        arguments = [
            Argument(
                id="A1",
                arg_type="support",
                text="Strong support",
                evidence=["Target | support"],
                target_item_name="Target",
                combined_score=0.8,
            ),
            Argument(
                id="A2",
                arg_type="attack",
                text="Moderate attack",
                evidence=["Target | attack"],
                target_item_name="Target",
                combined_score=0.3,
            ),
        ]
        graph = build_argument_graph(arguments, root_base_score=0.5)
        result = evaluate_root_dfquad(graph)

        self.assertAlmostEqual(aggregate_strength([0.2, 0.3]), 0.44)
        self.assertAlmostEqual(result.aggregated_support, 0.8)
        self.assertAlmostEqual(result.aggregated_attack, 0.3)
        self.assertAlmostEqual(result.final_score, 0.75)

    def test_graph_score_precedence_is_preserved(self):
        argument = Argument(
            id="A1",
            arg_type="support",
            text="Support",
            evidence=["Target | evidence"],
            target_item_name="Target",
            llm_score=0.9,
            mf_score=0.8,
            combined_score=0.7,
        )
        graph = build_argument_graph([argument])
        self.assertEqual(graph.nodes["A1"].base_score, 0.7)


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
