import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

with open("data/train_data.json", "r") as f:
    train_data = json.load(f)

train_data = pd.DataFrame(train_data)

prompt = "You are an expert linguist identifying native language backgrounds of non-native English speakers. Given their written English text, predict their native language.\n"
train_data['text'] = prompt + train_data['text']

label_encoder = LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['label'])

model_name = "qwen-2.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

train_encodings = tokenizer(list(train_data['text']), truncation=True, padding=True, max_length=512, return_tensors='pt')
train_dataset = {'input_ids': train_encodings['input_ids'], 'labels': list(train_data['label'])}

# Evaluate before fine-tuning
trainer = Trainer(model=model)
outputs = trainer.predict(train_dataset)
before_accuracy = accuracy_score(train_data['label'], outputs.predictions.argmax(axis=1))
print(f"Accuracy before fine-tuning (Qwen): {before_accuracy}")

# Evaluate after fine-tuning
fine_tuned_model_path = "./results_qwen/checkpoint-best"
model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path)
trainer = Trainer(model=model)
outputs = trainer.predict(train_dataset)
after_accuracy = accuracy_score(train_data['label'], outputs.predictions.argmax(axis=1))
print(f"Accuracy after fine-tuning (Qwen): {after_accuracy}")
