import unittest
from rag.common.dataset import Dataset

class TestDataset(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.dataset = Dataset(data=[
            {'old_name1': 'value1', 'old_name2': 'value2'},
            {'old_name1': 'value3', 'old_name2': 'value4'},
        ])

    def test_rename_columns_strict(self):
        mapping = {
            'old_name1': 'new_name1',
            'old_name2': 'new_name2',
            'non_existent': 'new_name3'  # This should raise an error
        }
        with self.assertRaises(KeyError):
            self.dataset.rename_columns(mapping, strict=True)

    def test_rename_columns_non_strict(self):
        mapping = {
            'old_name1': 'new_name1',
            'old_name2': 'new_name2',
            'non_existent': 'new_name3'  # This should not raise an error
        }
        self.dataset.rename_columns(mapping, strict=False)
        expected_data = [
            {'new_name1': 'value1', 'new_name2': 'value2'},
            {'new_name1': 'value3', 'new_name2': 'value4'},
        ]
        self.assertEqual(self.dataset.data, expected_data)

if __name__ == '__main__':
    unittest.main()
