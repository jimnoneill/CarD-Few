# CarD-Few: Carcinogenic Context Detection by Few-Shot Learning

CarD-Few is a Python package developed to identify sentences in scientific literature that discuss carcinogens, leveraging the power of few-shot learning with state-of-the-art NLP models.

## Installation

To install CarD-Few, clone this repository and install the package using pip. Ensure you have Python 3.7 or newer.

```bash
git clone https://github.com/jimnoneill/CarD-Few.git
cd CarD-Few
pip install .
```

## Usage
# Running the Classifier on Your Data
CarD-Few allows you to classify sentences to identify mentions of carcinogens easily. Here's how you can use it:

```python
from card_few.model import CarDFewClassifier

# Initialize the classifier
classifier = CarDFewClassifier()

# Classify a list of sentences
sentences = ["This compound has been found to cause cancer in lab rats.",
             "No evidence suggests that this substance is a carcinogen."]
predictions = classifier.predict(sentences)

print(predictions)
```

# Loading and Preparing Your Dataset
You can also load your dataset for classification:

```python
from card_few.data_processing import load_tsv_dataset

# Assuming you have a dataset in TSV format
dataset_path = 'path/to/your/dataset.tsv'
dataset = load_tsv_dataset(dataset_path)

# Now, you can use the classifier to predict this dataset


```
# Running the Model with Command Line
We provide a script to run the model directly from the command line on a dataset:
```bash
python -m card_few.run path/to/your/dataset.tsv
```
This will output the classifications directly to your terminal.

# Validation
Model was trained 80\20 split on 4 contextual categories. 1. Carcinogen implication. 2. Negative conclusion of carcinogenicty. 3. Antineoplasticity implication. 4. Unknown\other classification.
F1 0.926

## Contributing
We welcome contributions to CarD-Few! Whether it's adding new features, improving documentation, or reporting issues, please feel free to reach out.

## License
CarD-Few is licensed under the MIT License. See the LICENSE file for more details.

