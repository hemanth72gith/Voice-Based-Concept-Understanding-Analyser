import unittest

import numpy as np

from modules.semantic_analysis import calculate_semantic_similarity


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0], [0.6, 0.8]])


class SemanticAnalysisTests(unittest.TestCase):
    def test_cosine_similarity_is_returned_as_percent(self):
        result = calculate_semantic_similarity(
            "A transcript", "A reference", model=FakeEmbeddingModel()
        )
        self.assertAlmostEqual(result["cosine_similarity"], 0.6)
        self.assertAlmostEqual(result["similarity_percent"], 60.0)

    def test_empty_reference_is_rejected_before_model_call(self):
        with self.assertRaises(ValueError):
            calculate_semantic_similarity("A transcript", "  ", model=FakeEmbeddingModel())


if __name__ == "__main__":
    unittest.main()
