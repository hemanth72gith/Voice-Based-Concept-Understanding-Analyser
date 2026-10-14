import unittest

from modules.report_generation import generate_pdf_report


class ReportGenerationTests(unittest.TestCase):
    def test_report_is_a_pdf_with_analysis_fields(self):
        result = {
            "concept_name": "Gravity",
            "audio_filename": "practice.wav",
            "reference": "Gravity attracts objects with mass.",
            "transcript": "Gravity pulls objects toward Earth.",
            "audio_metrics": {
                "duration_seconds": 3.0,
                "pause_ratio_percent": 10.0,
                "rms_energy": 0.12,
            },
            "filler_analysis": {
                "total_words": 6,
                "filler_count": 0,
                "filler_ratio_percent": 0.0,
            },
            "semantic": {"similarity_percent": 75.0},
            "score": {
                "overall_score": 80.0,
                "weights": {"semantic": 0.7, "fillers": 0.2, "pauses": 0.1},
            },
            "feedback": {"headline": "Good base to build on", "strengths": [], "suggestions": []},
        }
        pdf = generate_pdf_report(result)
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertGreater(len(pdf), 1000)


if __name__ == "__main__":
    unittest.main()
