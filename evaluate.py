import numpy as np
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from data import train_tokenized, test_tokenized


model_path = "./results/checkpoint-3000"  # Path to the saved model checkpoint


model = AutoModelForSequenceClassification.from_pretrained(model_path)

training_args = TrainingArguments(
    output_dir="./evaluation"
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_tokenized
)


predictions = trainer.predict(test_tokenized)
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.4f}")

# results = trainer.evaluate()
# print("Evaluation results:", results)