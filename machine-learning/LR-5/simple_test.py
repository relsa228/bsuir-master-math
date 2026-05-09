import torch_directml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def run_test():
    model_path = "./fine_tuned_roberta"

    device = torch_directml.device()
    print(f"Device: {device}")

    print("Model loading")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    labels_map = {"LABEL_0": "(-) Negative", "LABEL_1": "(+) Positive"}
    test_phrases = [
        "I love how this brand cares about the environment!",
        "Just bought their new sneakers. Best purchase ever! 🔥",
        "The new update is absolutely terrible, everything is broken.",
        "Worst customer service experience in my life.",
        "The delivery was a bit late, but the product is high quality.",
        "I'm not sure if I like it, but it's definitely not bad.",
        "Disappointed with the packaging, it arrived damaged.",
        "Simply amazing service, highly recommend to everyone!",
        "The product arrived today. It's okay, I guess.",
    ]

    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print("=" * 50)

    for phrase in test_phrases:
        result = classifier(phrase)[0]
        label_id = result["label"]
        status = labels_map.get(label_id, label_id)
        confidence = result["score"]

        print(f"Sample: {phrase}")
        print(f"Result: {status} | Confidence: {confidence:.2%}")
        print("-" * 50)


if __name__ == "__main__":
    run_test()
