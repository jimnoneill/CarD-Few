from transformers import AutoModelForSequenceClassification, AutoTokenizer
from setfit import SetFitModel, SetFitTrainer
import torch

from setfit import SetFitModel

class CarDFewClassifier:
    def __init__(self, model_id):
        self.model = SetFitModel.from_pretrained(
            model_id,
            multi_target_strategy="one-vs-rest"  # or whatever strategy your model requires
        )

    def predict(self, texts):
        """
        Runs predictions on a list of texts.
        :param texts: a list of texts to classify.
        :return: a list of predictions.
        """
        # Prepare the texts as expected by SetFitModel
        prepped_texts = [{'text': text} for text in texts]

        # Run the model and get predictions
        predictions = self.model.predict(prepped_texts)
        return predictions


    def save_model(self, output_path):
        """
        Saves the model and tokenizer to the specified output path.
        :param output_path: the path where to save the model and tokenizer.
        """
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
