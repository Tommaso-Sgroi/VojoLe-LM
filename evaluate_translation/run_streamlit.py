import streamlit as st
from dataset_maker.database import DatabaseSor, DatabaseIta  # adjust import path
import os

# ====== DB CONNECTIONS ======
db_sor_path = os.getenv('DB_SOR')
db_ita_path = os.getenv('DB_ITA')

db_sor = DatabaseSor(db_sor_path)
db_ita = DatabaseIta(db_ita_path)


# ====== FUNCTIONS ======
def get_next_sentence():
    """Fetch one sentence_id with all generated translations."""
    translations = db_sor.get_translations(None)
    if not translations:
        return None, None, []

    # translations = (sentence_id, sentence_gen_id, sentence_text, train, evaluated)
    sentence_id = translations[0][0]  # from SorSentence
    ita_sentence = db_ita.get_sentence_by_id(sentence_id)[0]

    all_translations = [ss[2] for ss in translations]
    return sentence_id, ita_sentence, all_translations


def update_evaluation(sentence_id, sentence_gen_id, score):
    db_sor.add_evaluation(sentence_id, sentence_gen_id, score)
    db_sor.commit()


# ====== STREAMLIT UI ======
st.title("Sorianese Sentence Evaluator")

if "current_sentence_id" not in st.session_state:
    st.session_state.current_sentence_id, st.session_state.ita_sentence, st.session_state.translations = get_next_sentence()

if st.session_state.current_sentence_id is None:
    st.write("✅ All sentences have been evaluated!")
else:
    st.subheader("Italian Sentence")
    st.info(st.session_state.ita_sentence)

    st.subheader("Generated Translations")
    for i, trans in enumerate(st.session_state.translations):
        sentence_id = st.session_state.current_sentence_id
        sentence_gen_id = i
        sentence_text = trans

        st.write(f"**Generated Sentence #{sentence_gen_id}**")
        st.write(sentence_text)

        col1, col2, col3 = st.columns(3)
        if col1.button("Bad", key=f"bad_{sentence_id}_{sentence_gen_id}"):
            update_evaluation(sentence_id, sentence_gen_id, 1)
            print('bad')
        if col2.button("Acceptable", key=f"ok_{sentence_id}_{sentence_gen_id}"):
            update_evaluation(sentence_id, sentence_gen_id, 2)
            print('acceptable')
        if col3.button("Good", key=f"good_{sentence_id}_{sentence_gen_id}"):
            update_evaluation(sentence_id, sentence_gen_id, 3)
            print('good')


