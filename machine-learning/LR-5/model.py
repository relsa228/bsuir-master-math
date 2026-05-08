import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Загружаем датасет
raw_datasets = load_dataset("EleutherAI/twitter-sentiment")

model_name = "facebook/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Функция токенизации
def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=128
    )


# Токенизируем весь датасет
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols="label",
    shuffle=True,
    batch_size=16,
)

val_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols="label",
    shuffle=False,
    batch_size=16,
)

# Загружаем модель для бинарной классификации
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Оптимизатор с очень низким learning rate (стандарт для BERT-подобных)
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

# Обучение (обычно хватает 2-3 эпох для твитов)
model.fit(train_dataset, validation_data=val_dataset, epochs=2)

# Сохраняем модель
model.save_pretrained("./my_roberta_apple_model")

# ОЧЕНЬ ВАЖНО: Сохраните и токенизатор тоже!
# Без него вы не сможете правильно подготовить текст для предсказания.
tokenizer.save_pretrained("./my_roberta_apple_model")
