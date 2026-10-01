from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from data import train_tokenized, test_tokenized


# MODEL AND TRAINING ARGUMENTS
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-5
)


# TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized
)



trainer.train()



