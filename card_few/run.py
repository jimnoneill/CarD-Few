import argparse
from .model import CarDFewClassifier
from .data_processing import load_user_dataset
#Usage: python -m card_few.run path/to/your/dataset.csv

def main():
    # Set up an argument parser
    parser = argparse.ArgumentParser(description="Run CarD-Few model on your dataset.")
    parser.add_argument("dataset_path", type=str, help="The path to your dataset file.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Load the dataset
    user_dataset = load_user_dataset(args.dataset_path)

    # Initialize the classifier
    classifier = CarDFewClassifier()

    # Run the classifier on the dataset
    predictions = classifier.predict(user_dataset)

    # Output the predictions
    for sent, pred in zip(user_dataset, predictions):
        print(f"Sentence: {sent} - Prediction: {pred}")

if __name__ == "__main__":
    main()
