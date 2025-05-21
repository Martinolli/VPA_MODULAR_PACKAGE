import unittest
from vpa_modular.vpa_training_data_generator import VPATrainingDataGenerator
from vpa_modular.vpa_facade import VPAFacade
import os
import json

class TestVPATrainingDataGenerator(unittest.TestCase):
    def setUp(self):
        self.vpa_facade = VPAFacade()
        self.generator = VPATrainingDataGenerator(self.vpa_facade)

    def test_load_historical_data(self):
        ticker = "AAPL"
        start_date = "2024-01-01"
        end_date = "2025-01-31"
        timeframes = ["1d", "1h"]
        data = self.generator._load_historical_data(ticker, start_date, end_date, timeframes)
        
        self.assertIsNotNone(data)
        self.assertEqual(len(data), len(timeframes))
        for tf in timeframes:
            self.assertIn(tf, data)
            self.assertFalse(data[tf].empty)

    def test_generate_training_data(self):
        ticker = "AAPL"
        start_date = "2024-01-01"
        end_date = "2025-01-31"
        primary_tf = "1d"
        secondary_tfs = ["1h"]
        
        self.generator.generate_training_data(ticker, start_date, end_date, primary_timeframe=primary_tf, other_timeframes=secondary_tfs)
        
        # Check if the output file was created
        output_file = os.path.join(self.generator.output_dir, f"{ticker}_vpa_training_data.jsonl")
        self.assertTrue(os.path.exists(output_file), f"Output file {output_file} was not created")
            
        # Check if the file contains valid JSON lines
        with open(output_file, 'r') as f:
            lines = f.readlines()
            self.assertTrue(len(lines) > 0, "Output file is empty")
            for line in lines:
                data = json.loads(line)
                self.assertIn("input", data)
                self.assertIn("output", data)

if __name__ == '__main__':
    unittest.main()