import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from datasets import load_dataset
from nltk import download
from requests import post
from torch import stack
# from dataset_maker.dataset import load_dataset
from nltk import FreqDist
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

download('stopwords')
# === Config ===
SUBSET_SIZE = 5000
SIMILARITY_THRESHOLD = 0.85
LANG = "italian"
MAX_WORKERS = 8
NER_CATEGORIES = {"PER", "ORG", "GPE", "LOC", "PRODUCT", "EVENT"}
CACHE_DIR = '../data/.cache'
# dataset = load_dataset("gsarti/clean_mc4_it", name="tiny", split="validation", streaming=False, cache_dir=CACHE_DIR)
with(open('../data/clean_mc4_it/clean_mc4_it.jsonl', "r", encoding="utf-8")) as f:
    dataset = [json.loads(line) for line in f.readlines()]

# === Paths ===
TINT_URL = "http://localhost:8012/tint?pipeline=tint"
DISTRIBUTION_PATH = "../data/output/subset/lemma_distribution-clean_mc4_it_tiny_validation.json"
VOCAB_PATH = "../data/output/subset/italian_vocabulary.txt"
OUTPUT_BASE_DIR = "../data/output/subset"

# === Setup ===
stopwords_it = set(stopwords.words(LANG))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Load Italian vocabulary ===
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    italian_vocab = set(word.strip().lower() for word in f.readlines())


def load_lemma_distribution(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_lemma_distribution(path):
    if os.path.exists(path):
        print(f"Loading lemma distribution from {path}...")
        return load_lemma_distribution(path)
    else:
        print(f"Lemma distribution file not found. Calculating...")
        distribution = calculate_lemma_distribution()
        save_lemma_distribution(distribution, path)
        return distribution


def process_text_with_tint(text):
    try:
        response = post(
            TINT_URL,
            data=text.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
            timeout=20
        )
        if response.status_code != 200:
            raise RuntimeError(f"Status {response.status_code}")
        data = response.json()

        sentence_data = []

        for sentence in data.get("sentences", []):
            sentence_text = sentence["text"]

            lemma_list = []
            for token in sentence.get("tokens", []):
                word = token["word"].lower()
                lemma = token["lemma"].lower()
                ner = token.get("ner", "O")
                pos = token.get("ud_pos", "")

                if (
                        pos != "PUNCT" and
                        word.isalpha() and
                        word not in stopwords_it and
                        " " not in lemma and
                        lemma
                ):
                    lemma_list.append((word, lemma, ner))

            sentence_data.append((sentence_text, lemma_list))

        return sentence_data

    except Exception as e:
        print(f"Errore da Tint su: {text[:30]}... -> {e}")
        return []


def process_texts_in_parallel(texts):
    all_sentence_data = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_text_with_tint, text): i for i, text in enumerate(texts)}

        for future in tqdm(as_completed(futures), total=len(texts), desc="Processing texts"):
            try:
                sentence_data = future.result()
                all_sentence_data.extend(sentence_data)
            except Exception as e:
                print(f"Errore nell'elaborazione: {e}")

    return all_sentence_data


def calculate_lemma_distribution():
    print("Calculating lemma distribution...")

    texts = [entry["text"].lower() for entry in dataset]

    print("Processing texts to extract sentences and lemmas...")
    all_sentence_data = process_texts_in_parallel(texts)

    print(f"Total sentences extracted: {len(all_sentence_data)}")

    distribution = FreqDist()
    for _, lemma_list in all_sentence_data:
        for word, lemma, ent in lemma_list:
            if lemma in italian_vocab or ent in NER_CATEGORIES:
                distribution[lemma] += 1

    return dict(distribution)


def save_lemma_distribution(distribution, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(distribution, f, ensure_ascii=False)
    print(f"Lemma distribution saved to {path}")


def save_run_outputs(run_id, selected_sentences, subset_lemmas, all_lemmas, subset_distribution, excluded_lemmas):
    run_dir = os.path.join(OUTPUT_BASE_DIR, f"{run_id}_{SUBSET_SIZE}_{SIMILARITY_THRESHOLD}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, f"subset_{SUBSET_SIZE}_sentences.json"), "w", encoding="utf-8") as f:
        json.dump(selected_sentences, f, ensure_ascii=False, indent=2)

    with open(os.path.join(run_dir, "subset_distribution.json"), "w", encoding="utf-8") as f:
        subset_distribution = dict(sorted(subset_distribution.items(), key=lambda x: x[1], reverse=True))
        json.dump(subset_distribution, f, ensure_ascii=False)

    uncovered_lemmas = sorted(set(all_lemmas) - set(subset_lemmas))
    with open(os.path.join(run_dir, "uncovered_lemmas.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(uncovered_lemmas))

    with open(os.path.join(run_dir, "excluded_lemmas.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(excluded_lemmas)))

    summary = {
        "total_lemmas_in_full_distribution": len(all_lemmas),
        "lemmas_covered_in_subset": len(subset_lemmas),
        "coverage_ratio": len(subset_lemmas) / len(all_lemmas),
        "uncovered_lemmas_count": len(uncovered_lemmas),
        "total_sentences": len(selected_sentences)
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nRun results: {run_dir}")
    print(
        f"Lemmas coverage: {summary['lemmas_covered_in_subset']} / {summary['total_lemmas_in_full_distribution']} ({summary['coverage_ratio']:.2%})")
    print(f"Total sentences selected: {len(selected_sentences)}")


def select_subset():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Dataset loading...")
    texts = [entry["text"].lower() for entry in dataset]

    print("Processing texts to extract sentences and lemmas...")
    all_sentence_data = process_texts_in_parallel(texts)
    print(f"Total sentences extracted: {len(all_sentence_data)}")

    print("Lemma distribution loading...")
    lemma_distribution = get_lemma_distribution(DISTRIBUTION_PATH)
    all_lemmas_set = set(lemma_distribution.keys())

    lemmatized_data = []
    excluded_lemmas = set()

    for sentence, lemma_list in all_sentence_data:
        final_lemmas = set()
        for word, lemma, ent in lemma_list:
            if lemma in italian_vocab or ent in NER_CATEGORIES:
                final_lemmas.add(lemma)
            else:
                excluded_lemmas.add(lemma)
        lemmatized_data.append((sentence, final_lemmas))

    print("Scoring all sentences by rarity gain...")
    rarity = {lemma: 1 / (lemma_distribution.get(lemma, 1) + 1) for lemma in lemma_distribution}
    covered_lemmas = set()
    subset_distribution = FreqDist()
    selected_indices = []
    selected_embeddings = []
    scored_data = []

    for i, (sentence, lemmas) in enumerate(lemmatized_data):
        new_lemmas = lemmas - covered_lemmas
        if not new_lemmas:
            continue
        score = sum(rarity.get(lemma, 1) for lemma in new_lemmas)
        scored_data.append((i, sentence, lemmas, score))

        covered_lemmas.update(new_lemmas)

    scored_data.sort(key=lambda x: x[3], reverse=True)
    covered_lemmas.clear()

    print("Subset extraction (greedy + rarity + diversity)...")
    for i, sentence, lemmas, score in tqdm(scored_data):
        if len(selected_indices) >= SUBSET_SIZE:
            break

        new_lemmas = lemmas - covered_lemmas
        if not new_lemmas:
            continue

        emb = embedder.encode(sentence, convert_to_tensor=True)
        if selected_embeddings:
            embeddings_tensor = stack(selected_embeddings)
            sims = util.cos_sim(emb, embeddings_tensor)[0]
            if sims.max().item() > SIMILARITY_THRESHOLD:
                continue

        selected_indices.append(i)
        covered_lemmas.update(lemmas)
        selected_embeddings.append(emb)
        subset_distribution.update(lemmas)

    print(f"\nSelected sentences: {len(selected_indices)}")
    print(f"Covered lemmas: {len(covered_lemmas)}")

    selected_sentences = [lemmatized_data[i][0] for i in selected_indices]

    save_run_outputs(run_id, selected_sentences, covered_lemmas, all_lemmas_set, subset_distribution, excluded_lemmas)


if __name__ == "__main__":
    select_subset()
