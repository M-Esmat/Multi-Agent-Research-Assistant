from datasets import load_dataset

dataset = load_dataset("jamescalam/ai-arxiv-chunked")

print(dataset)
print(dataset["train"][0])

print(dataset["train"][0].keys())