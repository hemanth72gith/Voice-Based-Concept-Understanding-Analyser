import unittest

from modules.scoring import calculate_score


class ScoringTests(unittest.TestCase):
    def test_default_weighted_score_and_component_formulas(self):
        result = calculate_score(80, 5, 25)
        self.assertEqual(result["components"]["semantic"], 80)
        self.assertEqual(result["components"]["fillers"], 80)
        self.assertEqual(result["components"]["pauses"], 80)
        self.assertAlmostEqual(result["overall_score"], 80)

    def test_user_weights_are_normalized(self):
        result = calculate_score(100, 0, 15, weights={"semantic": 2, "fillers": 1, "pauses": 1})
        self.assertEqual(result["weights"], {"semantic": 0.5, "fillers": 0.25, "pauses": 0.25})
        self.assertEqual(result["overall_score"], 100)

    def test_all_zero_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            calculate_score(50, 10, 20, weights={"semantic": 0, "fillers": 0, "pauses": 0})

    def test_out_of_range_measure_is_rejected(self):
        with self.assertRaises(ValueError):
            calculate_score(101, 0, 0)


if __name__ == "__main__":
    unittest.main()
