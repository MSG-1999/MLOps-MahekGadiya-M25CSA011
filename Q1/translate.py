from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

MODEL_NAME = "Helsinki-NLP/opus-mt-bn-en"
INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.txt"
BATCH_SIZE = 8


def load_model():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def translate_lines(lines, tokenizer, model, batch_size=BATCH_SIZE):
    translations = []
    for i in tqdm(range(0, len(lines), batch_size), desc="Translating"):
        batch = lines[i : i + batch_size]
        # Tokenize
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        # Generate translations
        translated = model.generate(**inputs)
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
        translations.extend(decoded)
    return translations


def main():
    # Load model
    tokenizer, model = load_model()

    # Read input file
    print(f"Reading input from: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Total sentences to translate: {len(lines)}")

    # Translate
    translations = translate_lines(lines, tokenizer, model)

    # Save output
    print(f"Saving translations to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for t in translations:
            f.write(t + "\n")

    # Print first translation
    print("\n--- First sentence translation ---")
    print(f"Input : {lines[0]}")
    print(f"Output: {translations[0]}")


if __name__ == "__main__":
    main()