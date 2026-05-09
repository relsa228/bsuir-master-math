import numpy as np
import torch_directml
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def main():
    device = torch_directml.device()
    print(f"Device: {device}")

    model_name = "distilbert/distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)

    # print("Data loading...")
    # dataset = load_dataset("EleutherAI/twitter-sentiment")
    # small_train = dataset["train"].shuffle(seed=42).select(range(10000))
    # split = small_train.train_test_split(test_size=0.1)
    # split = dataset["train"].train_test_split(test_size=0.1)

    print("Data loading...")
    raw_dataset = load_dataset("EleutherAI/twitter-sentiment")

    total_samples = min(135000, len(raw_dataset["train"]))
    full_train_sample = (
        raw_dataset["train"].shuffle(seed=42).select(range(total_samples))
    )

    split = full_train_sample.train_test_split(test_size=0.1, seed=42)

    def tokenize_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=128)

    tokenized_ds = split.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": (predictions == labels).mean()}

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        fp16=False,
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training start...")
    trainer.train()

    print("Weight save...")
    model.save_pretrained("./fine_tuned_roberta")
    tokenizer.save_pretrained("./fine_tuned_roberta")
    print("Model saved in ./fine_tuned_roberta")


if __name__ == "__main__":
    main()
