from datasets import load_dataset
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from card_few.data_processing import prepare_dataset, load_tsv_dataset  # Make sure this is the correct import

def train_model(dataset_path, model_save_path):
    # Load and prepare the dataset
    dataset = load_tsv_dataset(dataset_path)  # Make sure this function is defined or imported correctly
    train_ds, eval_ds = prepare_dataset(dataset)   # Make sure this function is defined or imported correctly

    # Initialize the SetFitModel
    model = SetFitModel.from_pretrained('thenlper/gte-large',average="micro")

    # Initialize the trainer
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss_class=CosineSimilarityLoss,
        metric="precision",
        batch_size=8,
        num_iterations=20,
        seed=42,
        num_epochs=5,
        column_mapping={"sentence": "text", "label": "label"}
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(model_save_path)

    return model  # Return the trained model for use in testing if needed immediately after training


if __name__ == "__main__":
    dataset_path = 'data/fewshot_trainset3.tsv'
    model_save_path = 'data/trained_model'
    train_model(dataset_path, model_save_path)
