# Here we use an LLM (like Command-A from Cohere AI) to generate a silver version of the dataset.

# We may use the methodology from the paper "LexC-Gen: Generating Data for Extremely Low-Resource Languages
# with Large Language Models and Bilingual Lexicons" (https://arxiv.org/pdf/2402.14086) [our dictionary would be the
# bilingual lexicon constraint].

# --- Imports ---
import os
import random
import json
import re

import pandas as pd

# --- Run Configuration ---
EXTRACT_SUBSET = True
PRE_PROCESS_DICTIONARY = True

# --- Paths ---
COMMONS_DATA_DIR = '../data2/commons/'
RAW_DICTIONARY = COMMONS_DATA_DIR + 'scraped_dictionary.jsonl'
T5_DATA_DIR = '../data2/t5_finetune/'
TATOEBA = T5_DATA_DIR + 'tatoeba_it.tsv'

def extract_subset():
    tat = pd.read_csv(TATOEBA, sep="\t", names=["id", "lang", "text"])

    # Filter for Italian sentences long between 5 and 25 words, excluding URLs
    ita = tat[tat.lang == "ita"]
    mask = ita.text.str.split().str.len().between(5, 25) & ~ita.text.str.contains("http|www", regex=True)
    ita = ita[mask]

    # Shuffle with seed for reproducibility, extract 15k samples
    sample = ita.sample(15_000, random_state=42)["text"].str.replace(r"\s+", " ", regex=True)
    sample.to_csv(f"{T5_DATA_DIR}tatoeba_it_15k.tsv", sep="\t", index=False, header=False)

def pre_process_dictionary():
    rows = [json.loads(l) for l in open(RAW_DICTIONARY)]
    records = []
    for r in rows:
        contesto = r["contesto"].strip()
        dial = r["parola"].strip().lower()
        m = re.match(r"([A-Za-zÀ-ÿ' ]+)", contesto)
        if not m:
            continue
        it_lemma = m.group(1).strip().lower()
        note = contesto[len(m.group(0)):].strip(" ,;:-").strip()
        records.append((it_lemma, dial, note))
    df = pd.DataFrame(records, columns=["it", "dial", "notes"]) \
        .drop_duplicates(subset=["it", "dial"])
    df.sort_values(by="it", inplace=True)
    df.to_json(f"{COMMONS_DATA_DIR}silver_dictionary.jsonl", orient="records", lines=True, force_ascii=False)


if __name__ == '__main__':
    if EXTRACT_SUBSET:
        if not os.path.exists(f"{T5_DATA_DIR}tatoeba_it_15k.tsv"):
            print("[+] Extracting subset from Tatoeba...")
            extract_subset()
        else:
            print("[-] Subset already exists, skipping extraction.")
    if PRE_PROCESS_DICTIONARY:
        if not os.path.exists(f"{COMMONS_DATA_DIR}silver_dictionary.jsonl"):
            print("[+] Pre-processing dictionary...")
            pre_process_dictionary()
        else:
            print("[-] Dictionary already exists, skipping pre-processing.")
