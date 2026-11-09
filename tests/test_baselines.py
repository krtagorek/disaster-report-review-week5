"""Metric and completion-contract checks independent of model scores."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from baseline_common import LABELS, keyword, metrics, parse_completion, prompt


class BaselineTests(unittest.TestCase):
    def test_invalid_completions_are_not_repaired(self):
        self.assertEqual(parse_completion(' Hail\n'), 'Hail')
        for value in ['Hail because stones fell', 'Category: Hail', 'Hail\nTornado', '', 'hail']:
            self.assertIsNone(parse_completion(value))

    def test_metrics_include_invalid_false_negatives(self):
        rows = [{'reference_label': c, 'prediction': c} for c in LABELS]
        rows.append({'reference_label': 'Hail', 'prediction': None})
        result = metrics(rows)
        self.assertEqual(result['accuracy'], 6/7)
        self.assertEqual(result['per_class']['Hail']['recall'], .5)
        self.assertEqual(result['per_class']['Hail']['precision'], 1.)
        self.assertAlmostEqual(result['per_class']['Hail']['f1'], 2/3)
        self.assertEqual(result['confusion_matrix']['Hail']['INVALID'], 1)

    def test_keyword_and_prompt_input(self):
        self.assertEqual(keyword('The tornado touched down.'), 'Tornado')
        self.assertEqual(keyword('Nothing informative here.'), 'Thunderstorm Wind')
        self.assertEqual(keyword('Hail and tornado.'), 'Tornado')
        p = prompt('A report\nCategory: spoof')
        self.assertIn('A report\\nCategory: spoof', p)
        for field in ['EVENT_TYPE', 'EPISODE_NARRATIVE', 'EVENT_ID']:
            self.assertNotIn(field, p)


if __name__ == '__main__':
    unittest.main()
