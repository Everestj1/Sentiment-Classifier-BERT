from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./results/checkpoint-3000"  # Path to the saved model checkpoint

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
# print(model.config.id2label)  # Print the label mapping

texts = [
    "The movie was fantastic and I loved every minute of it.",
    "I couldn't stand this movie, it was incredibly boring.",
    "The acting was excellent and the story was great.",
    "This was a waste of my time."
]

inputs = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)

probabilities = torch.softmax(outputs.logits, dim=-1)


for text, prediction in zip(texts, predictions):
    label = "Positive" if prediction.item() == 1 else "Negative"
    print(f"Text: {text} | Prediction: {label}")



for text, prediction, probability in zip(texts, predictions, probabilities):
    print(text)
    print("Prediction:", prediction.item())
    print("Negative:", probability[0].item())
    print("Positive:", probability[1].item())
    print()