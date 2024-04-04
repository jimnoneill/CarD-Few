import pandas as pd
from datasets import Dataset

def load_tsv_dataset(file_path):
    """
    Load a dataset from a TSV file.
    :param file_path: The path to the TSV file containing the dataset.
    :return: A `Dataset` object from the `datasets` library.
    """
    # Assuming the TSV file has two columns: 'sentence' and 'label'
    #data = pd.read_csv(file_path, sep='\t', header=None, names=['sentence', 'label'])
    dataset = {}
    with open(file_path,'r') as fin:
        for line in fin:
            line = line.rstrip("\n")
            label = line[-1]
            sent = line[:-1].rstrip()
            if label in ["0","1","2","3","4"]:
                if label == "3":
                    label = "-1"
                if label == "4":
                    label = "-2"
                dataset[sent] = label
        # Convert the DataFrame into a Hugging Face Dataset
    return dataset
carc_training = {}
with open('/home/joneill/vaults/jmind/nlp/CarDBERT/training/fewshot_labels/fewshot_trainset3.tsv','r') as fin:
    for line in fin:
        line = line.rstrip("\n")
        label = line[-1]
        sent = line[:-1].rstrip()
        if label in ["0","1","2","3","4"]:
            if label == "3":
                label = "-1"
            if label == "4":
                label = "-2"
            carc_training[sent] = label
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

def prepare_dataset(dataset):
    """
    Tokenize, format, and split the dataset for the model.
    :param dataset: A `Dataset` object to prepare.
    :param tokenizer: The tokenizer to use for encoding the texts.
    :param max_length: The maximum sequence length for the model.
    :param split_ratio: The ratio to split the dataset into training and evaluation sets.
    :return: A tuple containing the pre-processed and tokenized training and evaluation datasets.
    """
    # Tokenize the dataset
    def dataset_maker(input_data):

        idx_list, sentence_list, label_list = [], [], []
        for i, (k, v) in enumerate(input_data.items()):
            idx_list.append(i)
            sentence_list.append(k)
            label_list.append(int(v))

        data_dict = {'idx': idx_list, 'sentence': sentence_list, 'label': label_list}

        custom_dataset = Dataset.from_dict(data_dict)
        return custom_dataset


    ds_dataset = dataset_maker(dataset)


    ds_shuffle = ds_dataset.shuffle(seed=42)

    train_dataset = ds_shuffle.select([i for i in range(int(len(ds_shuffle)*.8))])
    eval_ds = ds_shuffle.select([i for i in range(len(ds_shuffle)) if i >= int(len(ds_shuffle)*.2)])
    return train_dataset, eval_ds
