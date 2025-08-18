# ----------------------------------------------------------------------
#  Imports & logging (unchanged)
# ----------------------------------------------------------------------
import logging
import os
import sys
from json import loads as json_loads
import streamlit as st
from dataset_maker.database import DatabaseSor, DatabaseIta, VALIDATION
from logging import getLogger
# import pythonmonkey

# jsonrepair = pythonmonkey.require('jsonrepair').jsonrepair

def get_logger():
    logger = getLogger(__name__)
    if logger.hasHandlers():
        return logger
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(sh)
    return logger

logger = get_logger()
# jsonrepair = pythonmonkey.require("jsonrepair").jsonrepair

db_sor_path = os.getenv("DB_SOR")
db_ita_path = os.getenv("DB_ITA")

# reset timestamps in db
DatabaseSor(db_sor_path).reset_timestamps().commit()

# ----------------------------------------------------------------------
#  Session‑state defaults
# ----------------------------------------------------------------------
if "current_sentence_id" not in st.session_state:
    st.session_state.current_sentence_id = None   # the id of the source sentence
    st.session_state.translations = []           # list of dicts: {"gen_id": int, "text": str}
    st.session_state.ita_sentence = ""          # the Italian source text


# ----------------------------------------------------------------------
#  Helper – fetch a fresh (still‑locked) sentence from the DB
# ----------------------------------------------------------------------
def get_translations_from_db():
    """
    Return the raw rows that `DatabaseSor.get_translations_for_evaluation`
    gives us.  No caching – we want a fresh view after each DB write.
    """
    return DatabaseSor(db_sor_path).get_translations_for_evaluation(VALIDATION)


def _repair_and_unwrap(sentence_json: str):
    """
    The data stored in the DB is sometimes a JSON string like
    {"translation":"…"} – we want just the plain text.
    """
    try:
        loaded = json_loads(sentence_json)
    except Exception as e:                     # malformed → try to repair
        # sentence_json = jsonrepair(sentence_json)
        # loaded = json_loads(sentence_json)
        print(e)
        loaded = sentence_json

    if isinstance(loaded, dict):
        return loaded.get("translation")
    if isinstance(loaded, str):
        return loaded
    return None


def load_next_sentence():
    """
    Pick the next *un‑evaluated* source sentence (the DB method already
    locks it for 1hour) and store the first three translations in
    `st.session_state`.  Also fetch the Italian source sentence.
    """
    # Pull the raw rows (each row = (sentence_id, gen_id, text, …))
    rows = get_translations_from_db()
    if rows is None:
        # nothing left in the DB
        st.session_state.current_sentence_id = None
        st.session_state.translations = []
        st.session_state.ita_sentence = ""
        return

    # 2️ All rows belong to the same `sentence_id`
    sentence_id = rows[0][0]

    # 3️⃣  Keep only not‑yet‑evaluated translations and cut to three
    #    (the DB query may already return only unevaluated ones,
    #     but we double‑check just to be safe)
    filtered = [
        {"gen_id": r[1], "text": _repair_and_unwrap(r[2])}
        for r in rows
        # if r[4] == 0               # evaluated column = 0  (still unevaluated)
    ][:3]                          # show at most three

    # 4️⃣  Store everything in session_state
    st.session_state.current_sentence_id = sentence_id
    st.session_state.translations = filtered
    # Italian source sentence (just one, not a list)
    ita_row = DatabaseIta(db_ita_path).get_sentence_by_id(sentence_id)[0]
    st.session_state.ita_sentence = ita_row

    logger.debug(
        f"Loaded sentence {sentence_id} – {len(filtered)} translations."
    )


# ----------------------------------------------------------------------
#  Callback – called when a rating button is pressed
# ----------------------------------------------------------------------
def on_evaluate(sentence_id: int, gen_id: int, score: int):
    """
    Write the rating to the DB, remove the rating row from the UI and,
    if the list becomes empty, fetch a brand‑new sentence.
    """
    # 1️⃣  Persist the rating
    DatabaseSor(db_sor_path).add_evaluation(sentence_id, gen_id, score).commit()
    logger.debug(f"evaluated {sentence_id}-{gen_id} with score {score}")

    # 2️⃣  Remove the just‑rated translation from the UI list
    st.session_state.translations = [
        tr for tr in st.session_state.translations if tr["gen_id"] != gen_id
    ]

    # 3️⃣  If nothing left → next source sentence
    if not st.session_state.translations:
        load_next_sentence()


# ----------------------------------------------------------------------
#  First load (only when the app starts or when we exhausted a sentence)
# ----------------------------------------------------------------------
if st.session_state.current_sentence_id is None:
    load_next_sentence()

# ----------------------------------------------------------------------
#  UI rendering
# ----------------------------------------------------------------------
if st.session_state.translations:
    # ------------------- Source (Italian) -------------------
    st.subheader("Frase in Italiano")
    st.write(st.session_state.ita_sentence)

    # ------------------- Generated translations -------------------
    st.subheader("Frasi Generate")
    for tr in st.session_state.translations:
        gen_id = tr["gen_id"]
        gen_text = tr["text"]

        st.markdown(f"**Generated (#{gen_id})**")
        st.write(gen_text)

        # three rating buttons
        col1, col2, col3 = st.columns(3)
        col1.button(
            "Bad",
            key=f"bad_{st.session_state.current_sentence_id}_{gen_id}",
            on_click=on_evaluate,
            args=(st.session_state.current_sentence_id, gen_id, 1),

        )
        col2.button(
            "Acceptable",
            key=f"ok_{st.session_state.current_sentence_id}_{gen_id}",
            on_click=on_evaluate,
            args=(st.session_state.current_sentence_id, gen_id, 2),
        )
        col3.button(
            "Good",
            key=f"good_{st.session_state.current_sentence_id}_{gen_id}",
            on_click=on_evaluate,
            args=(st.session_state.current_sentence_id, gen_id, 3),
        )
else:
    # No translations left – either we ran out of data or the user just
    # finished the current sentence.
    st.success("All translations for this sentence have been evaluated.")
    if st.button("Load next sentence"):
        load_next_sentence()