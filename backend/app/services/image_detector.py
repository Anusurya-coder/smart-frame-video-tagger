from pathlib import Path

from groundingdino.util.inference import load_model, load_image, predict


CONFIG_PATH = Path(
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)

WEIGHTS_PATH = Path(
    "models/grounding_dino/groundingdino_swint_ogc.pth"
)


def detect_image(image_path: str, prompt: str):
    """
    Run Grounding DINO detection on one image.
    """

    model = load_model(
        str(CONFIG_PATH),
        str(WEIGHTS_PATH),
    )

    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=prompt,
        box_threshold=0.30,
        text_threshold=0.25,
        device="cpu",
    )

    results = []

    for box, confidence, phrase in zip(boxes, logits, phrases):
        results.append(
            {
                "label": phrase,
                "confidence": float(confidence),
                "box": box.tolist(),
            }
        )

    return results


def main():
    image_path = "test_data/test.jpg"
    prompt = "person"

    print("Running image detection...")

    results = detect_image(
        image_path=image_path,
        prompt=prompt,
    )

    print("\nDetection Results")
    print("-----------------")

    for result in results:
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Box: {result['box']}")
        print()


if __name__ == "__main__":
    main()