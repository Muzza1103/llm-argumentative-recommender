import unittest

from src.argumentation.dfquad import aggregate_strength, evaluate_root_dfquad
from src.argumentation.graph_builder import build_argument_graph
from src.argumentation.schema import Argument


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


if __name__ == "__main__":
    unittest.main()
