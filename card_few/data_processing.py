import pandas as pd
from datasets import Dataset

def load_tsv_dataset(file_path):
    """
    Load a dataset from a TSV file.
    :param file_path: The path to the TSV file containing the dataset.
    :return: A `Dataset` object from the `datasets` library.
    """
    # Assuming the TSV file has two columns: 'sentence' and 'label'
    data = pd.read_csv(file_path, sep='\t', header=None, names=['sentence', 'label'])

    # Convert the DataFrame into a Hugging Face Dataset
    dataset = Dataset.from_pandas(data)
    return dataset

def load_dict_dataset(data_dict):
    """
    Load a dataset from a dictionary with sentences as keys and labels as values.
    :param data_dict: A dictionary with sentences as keys and labels as values.
    :return: A `Dataset` object from the `datasets` library.
    """
    # Prepare data lists for conversion to Dataset
    sentences = list(data_dict.keys())
    labels = list(data_dict.values())

    # Create a DataFrame from the lists
    df = pd.DataFrame({'sentence': sentences, 'label': labels})

    # Convert the DataFrame into a Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    return dataset

def prepare_dataset(dataset, tokenizer, max_length=512):
    """
    Tokenize and format the dataset for the model.
    :param dataset: A `Dataset` object to prepare.
    :param tokenizer: The tokenizer to use for encoding the texts.
    :param max_length: The maximum sequence length for the model.
    :return: The pre-processed and tokenized dataset.
    """
    # Define the tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=max_length)

    # Apply tokenization
    dataset = dataset.map(tokenize_function, batched=True)
    return dataset
