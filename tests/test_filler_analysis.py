import unittest

from modules.filler_analysis import analyze_fillers


class FillerAnalysisTests(unittest.TestCase):
    def test_counts_single_and_multiword_fillers_case_insensitively(self):
        result = analyze_fillers("Um, you know, photosynthesis is basically plant food, uh.")
        self.assertEqual(result["filler_count"], 4)
        self.assertEqual(result["filler_tokens"], 5)
        self.assertAlmostEqual(
            result["filler_ratio_percent"], 5 / result["total_words"] * 100
        )
        self.assertEqual(
            {entry["phrase"]: entry["count"] for entry in result["occurrences"]},
            {"you know": 1, "basically": 1, "um": 1, "uh": 1},
        )

    def test_no_words_returns_zero_ratio(self):
        result = analyze_fillers("... !!!")
        self.assertEqual(result["total_words"], 0)
        self.assertEqual(result["filler_ratio_percent"], 0)

    def test_word_boundaries_avoid_partial_matches(self):
        result = analyze_fillers("The umbrella is useful and the human knows.")
        self.assertEqual(result["filler_count"], 0)


if __name__ == "__main__":
    unittest.main()
