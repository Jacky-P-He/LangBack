import json
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

with open("data/train_data.json", "r") as f:
    train_data = json.load(f)

train_data = pd.DataFrame(train_data)

label_encoder = LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['label'])

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

train_encodings = tokenizer(list(train_data['text']), truncation=True, padding=True, max_length=512, return_tensors='pt')
train_dataset = {'input_ids': train_encodings['input_ids'], 'labels': list(train_data['label'])}

# Evaluate before fine-tuning
trainer = Trainer(
    model=model
)
outputs = trainer.predict(train_dataset)
before_accuracy = accuracy_score(train_data['label'], outputs.predictions.argmax(axis=1))
print(f"Accuracy before fine-tuning (BERT): {before_accuracy}")

# Fine-tuning hyperparameters
training_args = TrainingArguments(
    output_dir="./results_bert",
    evaluation_strategy="epoch",
    logging_dir="./logs_bert",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    weight_decay=0.01,
    seed=42,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# Evaluate after fine-tuning
outputs = trainer.predict(train_dataset)
after_accuracy = accuracy_score(train_data['label'], outputs.predictions.argmax(axis=1))
print(f"Accuracy after fine-tuning (BERT): {after_accuracy}")
