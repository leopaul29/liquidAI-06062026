from datasets import load_dataset

# Load your dataset
dataset = load_dataset("RikkaBotan/nyan")

# See what columns and splits (train, test, etc.) it has
print(dataset)

# Look at the very first row of data
print(dataset['train'][0])