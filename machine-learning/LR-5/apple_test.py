import numpy as np
import pandas as pd

# Загружаем ваш файл с Kaggle
apple_df = pd.read_csv("apple_tweets.csv")  # Путь к файлу


def get_sentiment(text):
    inputs = tokenizer(
        text, return_tensors="tf", padding=True, truncation=True, max_length=128
    )
    outputs = model(inputs)
    # Получаем вероятности
    probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
    # 0 - Negative, 1 - Positive
    label_idx = np.argmax(probs)
    confidence = probs[label_idx]

    return label_idx, confidence


# Пример тестирования на Apple
test_tweet = "Apple's new M3 chip is absolutely mind-blowing! Best laptop ever."
label, conf = get_sentiment(test_tweet)
print(f"Result: {'Positive' if label == 1 else 'Negative'} (Confidence: {conf:.2f})")
