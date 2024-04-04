import unittest
from card_few.data_processing import load_tsv_dataset, prepare_dataset
from card_few.model import CarDFewClassifier
card_few
import unittest
from datasets import load_dataset
import pytest
from card_few.train_model import train_model
from setfit import SetFitModel

class TestCarDFewClassifier(unittest.TestCase):
    #def setUp(self):
    @classmethod
    def setUpClass(cls):
        #self.dataset_path = 'data/fewshot_trainset3.tsv'
        #self.model = CarDFewClassifier(model_id='model')
        cls.dataset_path = 'data/fewshot_trainset3.tsv'
        #cls.model = CarDFewClassifier()
        model_save_path = 'data/trained_model'
        train_model(dataset_path, model_save_path)
        cls.model = SetFitModel.from_pretrained(model_save_path)
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
