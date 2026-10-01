from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter

dataset = load_dataset("stanfordnlp/imdb")

train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))  # Select a subset of the training data for faster processing
test_dataset = dataset["test"].shuffle(seed=42).select(range(200))  # Select a subset of the test data for faster processing



tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"], 
        padding="max_length", 
        truncation=True,
        max_length=128
    )

train_tokenized = train_dataset.map(tokenize, batched=True)
test_tokenized = test_dataset.map(tokenize, batched=True)

# print(Counter(test_tokenized["label"]))
# print(train_dataset["label"][:20])
# print(train_dataset["text"][:5])


# print(train_dataset)
# print(test_dataset)
# print(train_tokenized)
# print(test_tokenized)




# DATASET AND TOKENIZER
# data = {
#     "text": [
#         "I love this product",
#         "This is amazing",
#         "I am very happy with it",
#         "Excellent experience",
#         "I hate this product",
#         "This is terrible",
#         "I am very disappointed",
#         "Worst experience ever"
#     ],
#     "label": [
#         1, 1, 1, 1,
#         0, 0, 0, 0
#     ]
# }
# dataset = Dataset.from_dict(data)
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# def tokenize(batch):
#     return tokenizer(
#         batch["text"], 
#         padding="max_length", 
#         truncation=True,
#         max_length=128
#     )

# # TOKENIZED DATASET
# tokenized_dataset = dataset.map(tokenize, batched=True)







