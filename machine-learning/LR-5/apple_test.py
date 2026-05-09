import pandas as pd
import torch_directml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def run_apple_test():
    model_path = "./fine_tuned_roberta"

    device = torch_directml.device()
    print(f"Device: {device}")

    print("Model loading")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    labels_map = {"LABEL_0": -1, "LABEL_1": 1}

    df = pd.read_csv("apple-twitter-sentiment-texts.csv")
    data_pairs = list(zip(df["text"], df["sentiment"]))

    true_guesses, false_guesses = 0, 0
    confidences = list()

    for text, label in data_pairs:
        if label == 0:
            continue
        result = classifier(text)[0]
        label_id, confidence = result["label"], result["score"]
        confidences.append(confidence)
        status = labels_map.get(label_id, label_id)

        if status == label:
            true_guesses += 1
        else:
            false_guesses += 1

    print(f"Analysis accuracy: {(true_guesses / (true_guesses + false_guesses)):.2%}")
    print(f"Average prediction accuracy: {(sum(confidences) / len(confidences)):.2%}")


if __name__ == "__main__":
    run_apple_test()
