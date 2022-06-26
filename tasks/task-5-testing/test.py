import torch
from model import SentClf
from transformers import AutoTokenizer
import pandas as pd

# Helper function to get predictions
def predict(input_text):
    tokenized_inputs = tokenizer(
        input_text,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    ).to(device)

    # Model returns the raw logtits
    logits = model(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'])
    
    # Transform logits into actual probabilities
    prob = torch.sigmoid(logits).cpu().detach().numpy()[0]

    # A single probability is returned. Threshold the probability into ether 0 or 1
    pred_index = int(prob > 0.5)
    
    # Get label associated to index
    pred_label = args.labels[pred_index]
    
    # Calculate confidence
    confidence = prob if pred_index == 1 else 1-prob
    
    return pred_label, confidence

# Load model

args = torch.load("/content/NYU/tasks/task-4-language-transformer-models/workflow/model_artifacts/args.pt")
model = SentClf(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("/content/NYU/tasks/task-4-language-transformer-models/workflow/model_artifacts/model_best.pt", map_location=device))
model = model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased", use_fast=True)

# Load MFT-Neutral test data

print("-"*20)
print("MFT-Neutral test")
print("-"*20, "\n")

data = pd.read_csv('Behavioral Testing - MFT-Neutral.csv', skiprows=1)['Tamil']
incorrect_count = 0

for i,example in enumerate(data):
    print(f"Example {i+1}/{data.shape[0]}")
    pred, prob = predict(example)
    if pred != 'Non-Hate-Speech':
        incorrect_count += 1
print(f"\nError rate: {incorrect_count/data.shape[0]}")

# Load MFT-Adversarial test data

print("\n"+"-"*20)
print("\nMFT-Adversarial test")
print("-"*20, "\n")

data = pd.read_csv('Behavioral Testing - MFT-Adversarial.csv', skiprows=1)['Tamil']
incorrect_count = 0

for i,example in enumerate(data):
    print(f"Example {i+1}/{data.shape[0]}")
    pred, prob = predict(example)
    if pred != 'Non-Hate-Speech':
        incorrect_count += 1
print(f"\nError rate: {incorrect_count/data.shape[0]}")

# Load MFT-Script test data

print("\n"+"-"*20)
print("\nMFT-Script test")
print("-"*20, "\n")

data = pd.read_csv('Behavioral Testing - MFT-Script.csv', skiprows=1)[:6]
total_error_rate = 0
column_names = data.columns.to_list()[:4]
error_by_type = {
    0: 0,
    1: 0,
    2: 0,
    3: 0
}

for i, row in data.iterrows():
    print(f"Example {i+1}/{data.shape[0]}")
    correct_label = row[4]
    example_error_count = 0
    for j, example in enumerate(row[:4]):
        pred, prob = predict(example)
        if pred != correct_label:
            example_error_count += 1
            error_by_type[j] += 1

    total_error_rate += (example_error_count/4)

print("")
for script_type, error_count in error_by_type.items():
    print(f"{column_names[script_type]} error rate: {error_count/data.shape[0]}")

print(f"\nError rate: {total_error_rate/data.shape[0]}")

# Load INV-Typos test data

print("\n"+"-"*20)
print("\nINV-Typos test")
print("-"*20, "\n")

data = pd.read_csv('Behavioral Testing - INV-Typos.csv', skiprows=1)
data_groups = data.groupby('Index')
total_error_rate = 0

for name, group in data_groups:
    true_label = group['Label'].iloc[0]
    variations = group['Tamil'].to_list()
    predicted_labels = []

    for example in variations:
        pred, prob = predict(example)
        predicted_labels.append(pred)
    
    correct_prediction_count = predicted_labels.count(true_label)
    error_rate = 1.0 - (correct_prediction_count/len(variations))
    total_error_rate += error_rate
    print(f"Example {name} with label {true_label: <16}. Error rate: {error_rate}")

print(f"\nError rate: {total_error_rate/len(data_groups)}")
