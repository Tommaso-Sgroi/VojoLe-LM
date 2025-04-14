import os
import hashlib
import sqlite3
import time
import json
import pandas as pd
from datetime import datetime
import streamlit as st
from streamlit_shortcuts import add_keyboard_shortcuts
import threading

APP_TITLE = "Annotation Interface"
DB_PATH = "./db/annotations.db"
SUBSET_PATH = "./data/output/subset/20250414_175606_5000_0.85/subset_5000_sentences.json"  # Percorso aggiornato al nuovo formato
ANNOTATIONS_PER_USER = 2500

lock = threading.Lock()


def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")  # Write-Ahead Logging per gestire la concorrenza
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    with lock:
        conn = get_db_connection()
        conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        ''')

        conn.execute('''
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY,
            entry_id TEXT NOT NULL,
            original_text TEXT NOT NULL,
            translation TEXT,
            annotator_id INTEGER NOT NULL,
            status TEXT DEFAULT 'pending',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (annotator_id) REFERENCES users(id),
            UNIQUE(entry_id, annotator_id)
        )
        ''')

        conn.execute('''
        CREATE TABLE IF NOT EXISTS assignments (
            id INTEGER PRIMARY KEY,
            entry_id TEXT NOT NULL,
            annotator_id INTEGER NOT NULL,
            UNIQUE(entry_id, annotator_id),
            FOREIGN KEY (annotator_id) REFERENCES users(id)
        )
        ''')

        user1_exists = conn.execute("SELECT 1 FROM users WHERE username = 'annotator1'").fetchone()
        user2_exists = conn.execute("SELECT 1 FROM users WHERE username = 'annotator2'").fetchone()

        if not user1_exists:
            password_hash = hashlib.sha256("password1".encode()).hexdigest()
            conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                         ("annotator1", password_hash))

        if not user2_exists:
            password_hash = hashlib.sha256("password2".encode()).hexdigest()
            conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                         ("annotator2", password_hash))

        conn.commit()
        conn.close()


def normalize_entry(entry, index):
    """Normalizza l'entry per supportare sia il vecchio che il nuovo formato"""
    normalized = {}

    if isinstance(entry, str):
        return {
            "entry_id": f"entry_{index}",
            "text": entry,
            "original_entry": entry
        }

    if "url" in entry:
        normalized["entry_id"] = entry["url"]
    elif "id" in entry:
        normalized["entry_id"] = entry["id"]
    else:
        normalized["entry_id"] = f"entry_{index}"

    if "text" in entry:
        normalized["text"] = entry["text"]
    elif "sentence" in entry:
        normalized["text"] = entry["sentence"]
    elif "original" in entry:
        normalized["text"] = entry["original"]
    else:
        text_keys = [k for k in entry.keys() if isinstance(entry[k], str) and len(entry[k]) > 5]
        if text_keys:
            normalized["text"] = entry[text_keys[0]]
        else:
            normalized["text"] = "Testo non trovato"

    normalized["original_entry"] = entry

    return normalized


@st.cache_data(ttl=3600)
def load_and_normalize_subset():
    """Carica e normalizza le entry del subset"""
    raw_entries = load_subset()
    normalized = [normalize_entry(entry, i) for i, entry in enumerate(raw_entries)]
    return normalized


def load_subset():
    if os.path.exists(SUBSET_PATH):
        with open(SUBSET_PATH, 'r', encoding='utf-8') as f:
            try:
                entries = json.load(f)
                if not isinstance(entries, list):
                    for key in ["sentences", "data", "entries", "texts"]:
                        if key in entries and isinstance(entries[key], list):
                            entries = entries[key]
                            break
                return entries
            except json.JSONDecodeError:
                f.seek(0)
                lines = f.readlines()
                entries = [json.loads(line) for line in lines]
                return entries
    else:
        st.error(f"Subset file not found: {SUBSET_PATH}")
        return []


def assign_entries():
    with lock:
        conn = get_db_connection()
        assignments_exist = conn.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]

        if assignments_exist > 0:
            conn.close()
            return

        entries = load_and_normalize_subset()

        annotators = conn.execute("SELECT id FROM users").fetchall()

        if len(annotators) < 2:
            st.error("Not enough annotators in the database")
            conn.close()
            return

        half = min(ANNOTATIONS_PER_USER, len(entries) // 2)

        for i in range(half):
            conn.execute(
                "INSERT INTO assignments (entry_id, annotator_id) VALUES (?, ?)",
                (entries[i]["entry_id"], annotators[0][0])
            )

        for i in range(half, min(len(entries), half * 2)):
            conn.execute(
                "INSERT INTO assignments (entry_id, annotator_id) VALUES (?, ?)",
                (entries[i]["entry_id"], annotators[1][0])
            )

        conn.commit()
        conn.close()


def authenticate(username, password):
    with lock:
        conn = get_db_connection()
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        user = conn.execute(
            "SELECT id, username FROM users WHERE username = ? AND password_hash = ?",
            (username, password_hash)
        ).fetchone()
        conn.close()
        return user


def get_annotation_stats(user_id):
    with lock:
        conn = get_db_connection()
        total = conn.execute(
            "SELECT COUNT(*) FROM assignments WHERE annotator_id = ?",
            (user_id,)
        ).fetchone()[0]

        completed = conn.execute(
            "SELECT COUNT(*) FROM annotations WHERE annotator_id = ? AND status = 'completed'",
            (user_id,)
        ).fetchone()[0]

        skipped = conn.execute(
            "SELECT COUNT(*) FROM annotations WHERE annotator_id = ? AND status = 'skipped'",
            (user_id,)
        ).fetchone()[0]

        conn.close()

        return {
            "total": total,
            "completed": completed,
            "skipped": skipped,
            "pending": total - completed - skipped
        }


def get_user_entries(user_id):
    """Ottiene tutte le entry assegnate all'utente con il loro stato e traduzione"""
    with lock:
        conn = get_db_connection()
        query = """
        SELECT a.id as assignment_id, a.entry_id, 
               an.translation, an.status
        FROM assignments a
        LEFT JOIN annotations an ON a.entry_id = an.entry_id AND an.annotator_id = a.annotator_id
        WHERE a.annotator_id = ?
        ORDER BY a.id
        """

        entries = conn.execute(query, (user_id,)).fetchall()
        conn.close()

        return [dict(entry) for entry in entries]


def get_entry_by_index(user_id, index):
    """Recupera una entry specifica in base all'indice"""
    entries = get_user_entries(user_id)

    if not entries or index < 0 or index >= len(entries):
        return None, None, None, None

    entry_data = entries[index]
    entry_id = entry_data['entry_id']
    translation = entry_data['translation']
    status = entry_data['status']

    normalized_entries = load_and_normalize_subset()

    entry = next(
        (e for e in normalized_entries if e["entry_id"] == entry_id),
        {"entry_id": entry_id, "text": "Testo non trovato"}
    )

    return entry, entry_id, status, translation


def save_annotation(user_id, entry_id, original_text, translation, status):
    with lock:
        try:
            conn = get_db_connection()
            existing = conn.execute(
                "SELECT id FROM annotations WHERE entry_id = ? AND annotator_id = ?",
                (entry_id, user_id)
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE annotations SET translation = ?, status = ?, timestamp = CURRENT_TIMESTAMP WHERE id = ?",
                    (translation, status, existing[0])
                )
            else:
                conn.execute(
                    "INSERT INTO annotations (entry_id, original_text, translation, annotator_id, status) VALUES (?, ?, ?, ?, ?)",
                    (entry_id, original_text, translation, user_id, status)
                )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            st.error(f"Error saving annotation: {str(e)}")
            return False


def export_annotations():
    with lock:
        conn = get_db_connection()
        annotations = conn.execute("""
            SELECT a.*, u.username 
            FROM annotations a
            JOIN users u ON a.annotator_id = u.id
            WHERE a.status = 'completed'
        """).fetchall()

        conn.close()

        df = pd.DataFrame([dict(a) for a in annotations])
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"annotations_export_{timestamp}.csv"
        df.to_csv(filename, index=False)

        return filename, df


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="✍️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .status-box {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        width: 120px;
    }
    .status-pending {
        background-color: #cce5ff;
        color: #004085;
        border: 2px solid #b8daff;
    }
    .status-completed {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .status-skipped {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeeba;
    }
    .navigation-section {
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

    init_database()
    assign_entries()

    if 'user' not in st.session_state:
        st.session_state.user = None

    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    if st.session_state.user is None:
        login_page()
    else:
        annotation_interface()


def login_page():
    st.title("Annotation Interface Login")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = authenticate(username, password)
            if user:
                st.session_state.user = {
                    "id": user["id"],
                    "username": user["username"]
                }
                st.session_state.current_index = 0
                st.rerun()
            else:
                st.error("Invalid username or password")

    with col2:
        st.subheader("Instructions")
        st.markdown("""
        ### How to use this tool

        1. Login with your provided credentials
        2. Translate the text shown in the annotation interface
        3. Press Ctrl+Enter to submit your translation
        4. Press Esc to skip a difficult entry
        5. Use the navigation buttons to move between entries

        Your progress is automatically saved.
        """)


def annotation_interface():
    shortcuts = {
        "ctrl+enter": lambda: st.session_state.update({"submit_clicked": True}),
        "esc": lambda: st.session_state.update({"skip_clicked": True}),
        "ctrl+left": lambda: st.session_state.update({"prev_clicked": True}),
        "ctrl+right": lambda: st.session_state.update({"next_clicked": True})
    }
    add_keyboard_shortcuts(shortcuts)

    if "submit_clicked" not in st.session_state:
        st.session_state.submit_clicked = False
    if "skip_clicked" not in st.session_state:
        st.session_state.skip_clicked = False
    if "next_clicked" not in st.session_state:
        st.session_state.next_clicked = False
    if "prev_clicked" not in st.session_state:
        st.session_state.prev_clicked = False

    user_entries = get_user_entries(st.session_state.user["id"])

    if st.session_state.next_clicked and st.session_state.current_index < len(user_entries) - 1:
        st.session_state.current_index += 1
        st.session_state.next_clicked = False
    elif st.session_state.prev_clicked and st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.session_state.prev_clicked = False

    with st.sidebar:
        st.subheader(f"Welcome, {st.session_state.user['username']}")

        stats = get_annotation_stats(st.session_state.user["id"])
        st.metric("Total Assignments", stats["total"])
        st.metric("Completed", stats["completed"])
        st.metric("Pending", stats["pending"])
        st.metric("Skipped", stats["skipped"])

        progress = (stats["completed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        st.progress(progress / 100)
        st.text(f"Progress: {progress:.1f}%")

        st.markdown("<div class='navigation-section'></div>", unsafe_allow_html=True)
        st.subheader("Navigation")

        st.markdown(f"Entry **{st.session_state.current_index + 1}** of **{len(user_entries)}**")

        col1, col2 = st.columns(2)

        with col1:
            prev_disabled = st.session_state.current_index <= 0
            if st.button("◀ Previous", disabled=prev_disabled, key="prev_sidebar"):
                if not prev_disabled:
                    st.session_state.prev_clicked = True
                    st.rerun()

        with col2:
            next_disabled = st.session_state.current_index >= len(user_entries) - 1
            if st.button("Next ▶", disabled=next_disabled, key="next_sidebar"):
                if not next_disabled:
                    st.session_state.next_clicked = True
                    st.rerun()

        st.markdown("### Go to specific entry")
        goto_idx = st.number_input("Entry number:", min_value=1, max_value=len(user_entries), step=1,
                                   value=st.session_state.current_index + 1)
        if st.button("Go"):
            st.session_state.current_index = goto_idx - 1
            st.rerun()

        st.markdown("---")

        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()

    st.title("Translation Annotation")

    entry_data = get_entry_by_index(st.session_state.user["id"], st.session_state.current_index)

    if not entry_data[0]:
        st.error("No entries found for your account.")
        return

    entry, entry_id, status, existing_translation = entry_data

    current_status = status if status else "pending"
    status_text = current_status.upper()
    status_class = f"status-{current_status}"

    st.markdown(f"<div class='status-box {status_class}'>{status_text}</div>", unsafe_allow_html=True)

    st.subheader("Original Text")
    if "text" in entry:
        st.markdown(f"**{entry['text']}**")
    else:
        st.error("Testo non disponibile per questa entry")

    st.subheader("Your Translation")
    translation = st.text_area("Translate the text above",
                               value=existing_translation if existing_translation else "",
                               height=150, key=f"translation_{st.session_state.current_index}")

    col1, col2 = st.columns([1, 1])

    with col1:
        submit_button = st.button("Submit (Ctrl+Enter)")
        if submit_button or st.session_state.submit_clicked:
            st.session_state.submit_clicked = False
            if translation.strip():
                if save_annotation(st.session_state.user["id"], entry_id, entry.get('text', ''), translation,
                                   "completed"):
                    st.success("Translation saved!")

                    if st.session_state.current_index < len(user_entries) - 1:
                        st.session_state.current_index += 1

                    time.sleep(0.5)
                    st.rerun()
            else:
                st.warning("Please enter a translation before submitting.")

    with col2:
        skip_button = st.button("Skip (Esc)")
        if skip_button or st.session_state.skip_clicked:
            st.session_state.skip_clicked = False
            save_annotation(st.session_state.user["id"], entry_id, entry.get('text', ''), "", "skipped")

            if st.session_state.current_index < len(user_entries) - 1:
                st.session_state.current_index += 1

            st.info("Entry skipped!")
            time.sleep(0.5)
            st.rerun()

    st.markdown("---")
    st.markdown("**Keyboard Shortcuts:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- **Ctrl+Enter**: Submit translation")
        st.markdown("- **Esc**: Skip current entry")
    with col2:
        st.markdown("- **Ctrl+Left**: Go to previous entry")
        st.markdown("- **Ctrl+Right**: Go to next entry")


if __name__ == "__main__":
    import nltk

    nltk.download('stopwords')
    main()
