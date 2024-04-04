from transformers import AutoModelForSequenceClassification, AutoTokenizer
from setfit import SetFitModel, SetFitTrainer
import torch

class CarDFewClassifier:
    def __init__(self, model_name='thenlper/gte-large'):
        """
        Initializes the classifier with a pre-trained model.
        :param model_name: the name of the pre-trained model to load.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = SetFitModel.from_pretrained(model_name)

    def predict(self, sentences):
        """
        Runs predictions on a list of sentences.
        :param sentences: a list of sentences to classify.
        :return: a list of predictions.
        """
        # Tokenize the input sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Run the model and get the logits
        with torch.no_grad():
            output = self.model(**encoded_input)

        # Convert logits to probabilities and then to labels
        probabilities = torch.nn.functional.softmax(output.logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

        # Convert predictions to a list of labels to return
        return predictions.tolist()

    def save_model(self, output_path):
        """
        Saves the model and tokenizer to the specified output path.
        :param output_path: the path where to save the model and tokenizer.
        """
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
