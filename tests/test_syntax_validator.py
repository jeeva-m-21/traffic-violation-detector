import unittest
from src.anpr.syntax_validator import clean_ocr_text, format_indian_plate
from src.anpr.temporal_voter import TemporalPlateVoter

class TestANPRModules(unittest.TestCase):

    def test_clean_ocr_text(self):
        self.assertEqual(clean_ocr_text("MH 12 DE 1433"), "MH12DE1433")
        self.assertEqual(clean_ocr_text("tn-09/ab-1234"), "TN09AB1234")
        self.assertEqual(clean_ocr_text(""), "")

    def test_format_indian_plate_standard(self):
        self.assertEqual(format_indian_plate("MH 12 DE 1433"), "MH12DE1433")
        self.assertEqual(format_indian_plate("tn 09 ab 1234"), "TN09AB1234")
        self.assertEqual(format_indian_plate("DL 1C AA 1111"), "DL1CAA1111")
        self.assertEqual(format_indian_plate("KA 01 MG 8899"), "KA01MG8899")

    def test_positional_disambiguation(self):
        # 'O' in number position -> '0'
        self.assertEqual(format_indian_plate("MH 12 DE 143O"), "MH12DE1430")
        # 'I' in number position -> '1'
        self.assertEqual(format_indian_plate("MH 12 DE I433"), "MH12DE1433")
        # '0' in state letter position -> 'O'
        self.assertEqual(format_indian_plate("0D 02 AB 1234"), "OD02AB1234")

    def test_temporal_voter(self):
        voter = TemporalPlateVoter(min_confidence=0.4)
        voter.add_observation(track_id=1, plate_string="MH12DE1433", confidence=0.85)
        voter.add_observation(track_id=1, plate_string="MH12DE1433", confidence=0.90)
        voter.add_observation(track_id=1, plate_string="MH12DEI433", confidence=0.45)

        resolved, conf, count = voter.resolve_plate(track_id=1)
        self.assertEqual(resolved, "MH12DE1433")
        self.assertEqual(count, 2)
        self.assertAlmostEqual(conf, 0.875, places=2)

if __name__ == "__main__":
    unittest.main()
