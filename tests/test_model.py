import unittest
from card_few.data_processing import load_tsv_dataset, prepare_dataset
from card_few.model import CarDFewClassifier

import unittest
from datasets import load_dataset

class TestCarDFewClassifier(unittest.TestCase):
    def setUp(self):
        self.dataset_path = 'path/to/your/dataset'  # change to your dataset path
        self.model = CarDFewClassifier(model_id='your-pretrained-model')  # change to your model ID

    def test_prediction(self):
        # Load the dataset
        dataset = load_dataset(self.dataset_path)

        # Prepare the dataset - example code, adapt as necessary
        texts = [record['text'] for record in dataset['test']]  # change 'test' to the appropriate split if necessary

        # Run prediction
        predictions = self.model.predict(texts[:10])  # just the first 10 for quick testing

        # Perform your assertions here
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(isinstance(pred, int) for pred in predictions))


    def test_dataset_loading(self):
        """Test loading of the TSV dataset."""
        dataset = load_tsv_dataset(self.dataset_path)
        self.assertIsNotNone(dataset)
        self.assertTrue(len(dataset) > 0, "Dataset should not be empty.")

if __name__ == '__main__':
    unittest.main()
