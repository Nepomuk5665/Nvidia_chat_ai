from transformers import GPTNeoForCausalLM, GPT2TokenizerFast, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch

class CustomTextDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer

        self.encodings = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    encoding = self.tokenizer.encode(line)
                    self.encodings.append(encoding)

    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx], dtype=torch.long)

    def __len__(self):
        return len(self.encodings)

# Step 1: Load and tokenize the dataset
print("Loading and tokenizing the dataset...")
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
model.resize_token_embeddings(len(tokenizer))

file_path = 'database.txt'
dataset = CustomTextDataset(file_path, tokenizer)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir='./model',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_total_limit=2,
    learning_rate=1e-5,  # Adjust the learning rate as needed
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# Step 2: Start the training
print("Starting the training...")
trainer.train()

# Step 3: Save the trained model and tokenizer
print("Saving the trained model and tokenizer...")
model_dir = 'nepomuk'
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)

print("Training complete!")
