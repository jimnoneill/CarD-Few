import unittest
from card_few.data_processing import load_tsv_dataset, prepare_dataset
from card_few.model import CarDFewClassifier

class TestCarDFewClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Class method to set up environment once for all tests."""
        # Adjust the path as necessary to point to the correct location of your dataset
        cls.dataset_path = 'data/fewshot_trainset3.tsv'
        cls.model = CarDFewClassifier()

    def test_dataset_loading(self):
        """Test loading of the TSV dataset."""
        dataset = load_tsv_dataset(self.dataset_path)
        self.assertIsNotNone(dataset)
        self.assertTrue(len(dataset) > 0, "Dataset should not be empty.")

    def test_prediction(self):
        """Test model prediction on a portion of the dataset."""
        # Load the dataset
        dataset = load_tsv_dataset(self.dataset_path)
        # Prepare the dataset - note: adjust this to your actual implementation
        prepared_dataset = prepare_dataset(dataset, self.model.tokenizer, max_length=128)  # Example

        # Select a small subset for testing to save time
        test_subset = prepared_dataset.select(range(10))

        # Run prediction - adjust according to your model's method signatures
        predictions = self.model.predict(test_subset['sentence'])

        # Basic check to ensure predictions are returned
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), 10)  # Ensure we got a prediction for each input

if __name__ == '__main__':
    unittest.main()
