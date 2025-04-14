import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json

# === Config ===
TINT_URL = "http://localhost:8012/tint?pipeline=tint"
MAX_WORKERS = 240

# === Lemmatization Functions ===
def lemmatize_with_tint(text):
    try:
        response = requests.post(
            TINT_URL,
            data=text.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
            timeout=10
        )
        if response.status_code != 200:
            raise RuntimeError(f"Status {response.status_code}")
        data = response.json()

        results = []
        for sentence in data.get("sentences", []):
            for token in sentence.get("tokens", []):
                word = token["word"].lower()
                lemma = token["lemma"].lower()
                ner = token.get("ner", "O")
                pos = token.get("ud_pos", "")

                if (
                        pos != "PUNCT" and
                        word.isalpha() and
                        " " not in lemma and
                        lemma
                ):
                    results.append((word, lemma, ner))
        return results

    except Exception as e:
        print(f"Errore da Tint su: {text[:30]}... -> {e}")
        return []

def lemmatize_batch(texts):
    results = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(lemmatize_with_tint, text): i for i, text in enumerate(texts)}
        for future in tqdm(as_completed(futures), total=len(texts), desc="Lemmatizing"):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                print(f"Errore su testo {i}: {e}")
                results[i] = []
    return results

def main():
    filename = "../data/660k_italian_words.txt"
    with open(filename, 'r', encoding='utf-8') as file:
        words = set([line.strip() for line in file if line.strip()])

    print(f"Loaded {len(words)} italian words.")

    results = lemmatize_batch(words)

    lemmas_set = set()

    with open("../data/vocabulary_generation_summary.tsv", "w", encoding="utf-8") as f:
        for word, lemmas in zip(words, results):
            if lemmas:
                for original, lemma, ner in lemmas:
                    lemmas_set.add(lemma)
                for original, lemma, ner in lemmas:
                    f.write(f"{original}\t{lemma}\t{ner}\n")
            else:
                f.write(f"{word}\t{word}\tO\n")

    with open("../data/italian_vocabulary.txt", "w", encoding="utf-8") as f:
        for lemma in sorted(lemmas_set):
            f.write(f"{lemma}\n")

if __name__ == "__main__":
    main()
