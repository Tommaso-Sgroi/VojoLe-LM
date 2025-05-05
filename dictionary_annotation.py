import random

import streamlit as st
import sqlite3
import pandas as pd
import json
import os
import re
import uuid
from datetime import datetime


# Inizializzazione del database
def init_db():
    conn = sqlite3.connect('./db/sorianese_annotations.db')
    c = conn.cursor()

    # Tabella principale delle entries
    c.execute('''
    CREATE TABLE IF NOT EXISTS entries (
        id TEXT PRIMARY KEY,
        sorianese TEXT NOT NULL,
        pos TEXT,
        notes TEXT,
        status TEXT DEFAULT 'pending',
        flagged INTEGER DEFAULT 0,
        created_at TEXT,
        updated_at TEXT,
        source TEXT,
        definition TEXT,
        original_sorianese TEXT
    )
    ''')

    # Tabella delle traduzioni
    c.execute('''
    CREATE TABLE IF NOT EXISTS translations (
        id TEXT PRIMARY KEY,
        entry_id TEXT,
        italiano TEXT NOT NULL,
        context TEXT,
        register TEXT,
        FOREIGN KEY (entry_id) REFERENCES entries (id)
    )
    ''')

    # Tabella delle forme morfologiche
    c.execute('''
    CREATE TABLE IF NOT EXISTS morphology (
        id TEXT PRIMARY KEY,
        entry_id TEXT,
        form TEXT NOT NULL,
        type TEXT,
        FOREIGN KEY (entry_id) REFERENCES entries (id)
    )
    ''')

    # Tabella per le traduzioni morfologiche
    c.execute('''
    CREATE TABLE IF NOT EXISTS morphology_translations (
        id TEXT PRIMARY KEY,
        morphology_id TEXT,
        translation TEXT NOT NULL,
        FOREIGN KEY (morphology_id) REFERENCES morphology (id)
    )
    ''')

    # Tabella per le varianti
    c.execute('''
    CREATE TABLE IF NOT EXISTS variants (
        id TEXT PRIMARY KEY,
        entry_id TEXT,
        variant TEXT NOT NULL,
        FOREIGN KEY (entry_id) REFERENCES entries (id)
    )
    ''')

    # Tabella per gli esempi
    c.execute('''
    CREATE TABLE IF NOT EXISTS examples (
        id TEXT PRIMARY KEY,
        entry_id TEXT,
        sorianese TEXT NOT NULL,
        source TEXT,
        FOREIGN KEY (entry_id) REFERENCES entries (id)
    )
    ''')

    # Tabella per le traduzioni degli esempi
    c.execute('''
    CREATE TABLE IF NOT EXISTS example_translations (
        id TEXT PRIMARY KEY,
        example_id TEXT,
        translation TEXT NOT NULL,
        FOREIGN KEY (example_id) REFERENCES examples (id)
    )
    ''')

    conn.commit()
    return conn


# Parsing del dizionario con gestione duplicati
def parse_dictionary(file_path):
    entries = []
    seen_entries = {}  # Dizionario per tenere traccia delle entry viste

    with open(file_path, 'r', encoding='utf-8') as f:
        # Leggiamo l'intero file come una stringa
        content = f.read()

        # Dividiamo il contenuto in righe
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if not line:  # Saltiamo le righe vuote
                continue

            # Dividiamo la riga al primo ":" trovato
            parts = line.split(':', 1)  # Split solo al primo ":"

            if len(parts) < 2:  # Se non c'è ":", potrebbe essere un'intestazione o altro
                continue

            term = parts[0].strip()
            definition = parts[1].strip()

            # Gestione duplicati
            if term in seen_entries:
                count = seen_entries[term] + 1
                seen_entries[term] = count
                entries.append({
                    'sorianese': f"{term} ({count})",
                    'definition': definition,
                    'source': 'dizionario',
                    'original_sorianese': term
                })
            else:
                seen_entries[term] = 1
                entries.append({
                    'sorianese': term,
                    'definition': definition,
                    'source': 'dizionario',
                    'original_sorianese': term
                })

    return entries


# Parsing delle poesie
def parse_poetry(file_path):
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

        # Divide in righe e processa ogni riga
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.isupper():  # Salta intestazioni e righe vuote
                # Estrae le parole dalla riga
                words = re.findall(r'\b[a-zA-ZàèìòùÀÈÌÒÙ]+\b', line)
                for word in words:
                    if len(word) > 1:  # Salta lettere singole
                        entries.append({
                            'sorianese': word,
                            'definition': '',
                            'source': 'poesia',
                            'original_sorianese': word
                        })

                # Aggiunge anche l'intera riga come potenziale frase
                if len(line) > 5:  # Salta righe molto brevi
                    entries.append({
                        'sorianese': line,
                        'definition': '',
                        'source': 'poesia_riga',
                        'original_sorianese': line
                    })

    # Rimuove i duplicati mantenendo l'ordine
    unique_entries = []
    seen = set()
    for entry in entries:
        sorianese_lower = entry['sorianese'].lower()
        if sorianese_lower not in seen:
            seen.add(sorianese_lower)
            unique_entries.append(entry)

    return unique_entries


# Elaborazione dei file
def process_files(dictionary_path, poetry_path, conn):
    c = conn.cursor()

    # Controlla se ci sono già entries nel database
    c.execute("SELECT COUNT(*) FROM entries")
    count = c.fetchone()[0]

    if count == 0:
        # Parse dei file
        dict_entries = parse_dictionary(dictionary_path)
        poetry_entries = parse_poetry(poetry_path)

        # Combinazione delle entries
        all_entries = dict_entries + poetry_entries

        # Inserimento nel database
        timestamp = datetime.now().isoformat()
        for entry in all_entries:
            entry_id = str(uuid.uuid4())
            c.execute(
                "INSERT INTO entries (id, sorianese, status, created_at, source, definition, original_sorianese) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (entry_id, entry['sorianese'], 'pending', timestamp, entry['source'], entry.get('definition', ''),
                 entry.get('original_sorianese', ''))
            )

        conn.commit()
        return len(all_entries)

    return count


# Recupero di una entry tramite ID
def get_entry_by_id(conn, entry_id):
    c = conn.cursor()

    # Recupera l'entry principale
    c.execute("SELECT * FROM entries WHERE id = ?", (entry_id,))
    entry = c.fetchone()

    if not entry:
        return None

    # Converte in dizionario
    entry_dict = {
        'id': entry[0],
        'sorianese': entry[1],
        'pos': entry[2],
        'notes': entry[3],
        'status': entry[4],
        'flagged': entry[5],
        'created_at': entry[6],
        'updated_at': entry[7],
        'source': entry[8],
        'definition': entry[9],
        'original_sorianese': entry[10],
        'translations': [],
        'morphology': {'forms': []},
        'examples': [],
        'variants': []
    }

    # Recupera le traduzioni
    c.execute("SELECT id, italiano, context, register FROM translations WHERE entry_id = ?", (entry_id,))
    translations = c.fetchall()
    for t in translations:
        entry_dict['translations'].append({
            'id': t[0],
            'italiano': t[1],
            'context': t[2],
            'register': t[3]
        })

    # Recupera le forme morfologiche
    c.execute("SELECT id, form, type FROM morphology WHERE entry_id = ?", (entry_id,))
    forms = c.fetchall()
    for form in forms:
        form_dict = {
            'id': form[0],
            'form': form[1],
            'type': form[2],
            'translations': []
        }

        # Recupera le traduzioni morfologiche
        c.execute("SELECT id, translation FROM morphology_translations WHERE morphology_id = ?", (form[0],))
        morph_translations = c.fetchall()
        for mt in morph_translations:
            form_dict['translations'].append({
                'id': mt[0],
                'translation': mt[1]
            })

        entry_dict['morphology']['forms'].append(form_dict)

    # Recupera le varianti
    c.execute("SELECT id, variant FROM variants WHERE entry_id = ?", (entry_id,))
    variants = c.fetchall()
    for v in variants:
        entry_dict['variants'].append({
            'id': v[0],
            'variant': v[1]
        })

    # Recupera gli esempi
    c.execute("SELECT id, sorianese, source FROM examples WHERE entry_id = ?", (entry_id,))
    examples = c.fetchall()
    for ex in examples:
        example_dict = {
            'id': ex[0],
            'sorianese': ex[1],
            'source': ex[2],
            'translations': []
        }

        # Recupera le traduzioni degli esempi
        c.execute("SELECT id, translation FROM example_translations WHERE example_id = ?", (ex[0],))
        ex_translations = c.fetchall()
        for et in ex_translations:
            example_dict['translations'].append({
                'id': et[0],
                'translation': et[1]
            })

        entry_dict['examples'].append(example_dict)

    return entry_dict


# Recupero della lista di entries filtrata senza limiti
def get_entries_list(conn, status_filter=None, flagged_filter=None, search_query=None):
    c = conn.cursor()

    query = "SELECT id, sorianese, pos, status, flagged, source FROM entries"
    params = []

    conditions = []
    if status_filter and status_filter != "all":
        conditions.append("status = ?")
        params.append(status_filter)

    if flagged_filter is not None:
        conditions.append("flagged = ?")
        params.append(1 if flagged_filter else 0)

    if search_query:
        conditions.append("sorianese LIKE ?")
        params.append(f"%{search_query}%")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY source, sorianese"  # Rimosso il limite per mostrare tutte le entries

    c.execute(query, params)
    entries = c.fetchall()

    result = []
    for entry in entries:
        result.append({
            'id': entry[0],
            'sorianese': entry[1],
            'pos': entry[2],
            'status': entry[3],
            'flagged': entry[4],
            'source': entry[5]
        })

    return result


# Determinazione automatica dello status
def determine_status(entry_data):
    """Determina automaticamente lo status dell'entry in base ai dati inseriti"""
    # Se è stato premuto il pulsante Skippa, impostiamo skipped
    if entry_data.get('skip_pressed', False):
        return 'skipped'

    # Verifichiamo se c'è almeno una traduzione e un POS tag
    has_translation = any(t.get('italiano') for t in entry_data.get('translations', []))
    has_pos = entry_data.get('pos', '')

    # Se abbiamo sia traduzione che POS, è completata
    if has_translation and has_pos:
        return 'done'

    # Altrimenti è ancora pending
    return 'pending'


# Salvataggio di un'entry con status automatico
def save_entry(conn, entry_data):
    c = conn.cursor()
    timestamp = datetime.now().isoformat()

    # Determina lo status automaticamente
    entry_data['status'] = determine_status(entry_data)

    # Aggiorna l'entry principale
    c.execute(
        "UPDATE entries SET sorianese = ?, pos = ?, notes = ?, status = ?, flagged = ?, updated_at = ? WHERE id = ?",
        (entry_data['sorianese'], entry_data['pos'], entry_data['notes'], entry_data['status'],
         1 if entry_data['flagged'] else 0, timestamp, entry_data['id'])
    )

    # Gestione delle traduzioni
    c.execute("DELETE FROM translations WHERE entry_id = ?", (entry_data['id'],))
    for trans in entry_data['translations']:
        if trans.get('italiano'):  # Inserisce solo se c'è contenuto
            trans_id = trans.get('id') or str(uuid.uuid4())
            c.execute(
                "INSERT INTO translations (id, entry_id, italiano, context, register) VALUES (?, ?, ?, ?, ?)",
                (trans_id, entry_data['id'], trans['italiano'], trans.get('context', ''), trans.get('register', ''))
            )

    # Gestione della morfologia
    c.execute("SELECT id FROM morphology WHERE entry_id = ?", (entry_data['id'],))
    existing_forms = c.fetchall()
    for form in existing_forms:
        c.execute("DELETE FROM morphology_translations WHERE morphology_id = ?", (form[0],))

    c.execute("DELETE FROM morphology WHERE entry_id = ?", (entry_data['id'],))

    for form in entry_data['morphology']['forms']:
        if form.get('form'):  # Inserisce solo se c'è contenuto
            form_id = form.get('id') or str(uuid.uuid4())
            c.execute(
                "INSERT INTO morphology (id, entry_id, form, type) VALUES (?, ?, ?, ?)",
                (form_id, entry_data['id'], form['form'], form.get('type', ''))
            )

            # Inserimento traduzioni morfologiche
            for mt in form.get('translations', []):
                if mt.get('translation'):
                    mt_id = mt.get('id') or str(uuid.uuid4())
                    c.execute(
                        "INSERT INTO morphology_translations (id, morphology_id, translation) VALUES (?, ?, ?)",
                        (mt_id, form_id, mt['translation'])
                    )

    # Gestione delle varianti
    c.execute("DELETE FROM variants WHERE entry_id = ?", (entry_data['id'],))
    for variant in entry_data.get('variants', []):
        if variant.get('variant'):
            variant_id = variant.get('id') or str(uuid.uuid4())
            c.execute(
                "INSERT INTO variants (id, entry_id, variant) VALUES (?, ?, ?)",
                (variant_id, entry_data['id'], variant['variant'])
            )

    # Gestione degli esempi
    c.execute("SELECT id FROM examples WHERE entry_id = ?", (entry_data['id'],))
    existing_examples = c.fetchall()
    for example in existing_examples:
        c.execute("DELETE FROM example_translations WHERE example_id = ?", (example[0],))

    c.execute("DELETE FROM examples WHERE entry_id = ?", (entry_data['id'],))

    for example in entry_data.get('examples', []):
        if example.get('sorianese'):
            example_id = example.get('id') or str(uuid.uuid4())
            c.execute(
                "INSERT INTO examples (id, entry_id, sorianese, source) VALUES (?, ?, ?, ?)",
                (example_id, entry_data['id'], example['sorianese'], example.get('source', ''))
            )

            # Inserimento traduzioni di esempi
            for et in example.get('translations', []):
                if et.get('translation'):
                    et_id = et.get('id') or str(uuid.uuid4())
                    c.execute(
                        "INSERT INTO example_translations (id, example_id, translation) VALUES (?, ?, ?)",
                        (et_id, example_id, et['translation'])
                    )

    conn.commit()
    return True


# Funzioni per la navigazione tra entries
def get_next_entry_id(conn, current_id, status_filter=None):
    """Trova l'ID dell'entry successiva dopo quella corrente"""
    c = conn.cursor()

    query = "SELECT id FROM entries WHERE id > ? "
    params = [current_id]

    if status_filter and status_filter != "all":
        query += "AND status = ? "
        params.append(status_filter)

    query += "ORDER BY id LIMIT 1"

    c.execute(query, params)
    result = c.fetchone()

    if result:
        return result[0]

    # Se non abbiamo trovato nessuna entry successiva, torniamo alla prima
    query = "SELECT id FROM entries "
    params = []

    if status_filter and status_filter != "all":
        query += "WHERE status = ? "
        params.append(status_filter)

    query += "ORDER BY id LIMIT 1"

    c.execute(query, params)
    result = c.fetchone()

    if result:
        return result[0]

    return None


def get_prev_entry_id(conn, current_id, status_filter=None):
    """Trova l'ID dell'entry precedente a quella corrente"""
    c = conn.cursor()

    query = "SELECT id FROM entries WHERE id < ? "
    params = [current_id]

    if status_filter and status_filter != "all":
        query += "AND status = ? "
        params.append(status_filter)

    query += "ORDER BY id DESC LIMIT 1"

    c.execute(query, params)
    result = c.fetchone()

    if result:
        return result[0]

    # Se non abbiamo trovato nessuna entry precedente, torniamo all'ultima
    query = "SELECT id FROM entries "
    params = []

    if status_filter and status_filter != "all":
        query += "WHERE status = ? "
        params.append(status_filter)

    query += "ORDER BY id DESC LIMIT 1"

    c.execute(query, params)
    result = c.fetchone()

    if result:
        return result[0]

    return None


# Esportazione in JSON
def export_to_json(conn, output_path):
    c = conn.cursor()

    # Recupera tutte le entry con status 'done'
    c.execute("SELECT id FROM entries WHERE status = 'done'")
    entry_ids = c.fetchall()

    entries_json = {"entries": []}

    for entry_id in entry_ids:
        entry = get_entry_by_id(conn, entry_id[0])

        # Formatta l'entry per l'esportazione JSON
        json_entry = {
            "sorianese": entry['sorianese'],
            "pos": entry['pos'],
            "translations": [],
            "morphology": {"forms": []},
            "examples": [],
            "variants": [v['variant'] for v in entry['variants'] if v.get('variant')],
            "notes": entry['notes']
        }

        # Formatta le traduzioni
        for t in entry['translations']:
            trans = {"italiano": t['italiano']}
            if t['context']:
                trans["context"] = t['context']
            if t['register']:
                trans["register"] = t['register']
            json_entry['translations'].append(trans)

        # Formatta la morfologia
        for form in entry['morphology']['forms']:
            if form.get('form'):
                form_dict = {
                    "form": form['form'],
                    "type": form['type'],
                    "translations": [t['translation'] for t in form['translations'] if t.get('translation')]
                }
                json_entry['morphology']['forms'].append(form_dict)

        # Formatta gli esempi
        for ex in entry['examples']:
            if ex.get('sorianese'):
                ex_dict = {
                    "sorianese": ex['sorianese'],
                    "translations": [t['translation'] for t in ex['translations'] if t.get('translation')]
                }
                if ex.get('source'):
                    ex_dict["source"] = ex['source']
                json_entry['examples'].append(ex_dict)

        entries_json['entries'].append(json_entry)

    # Scrive sul file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entries_json, f, ensure_ascii=False, indent=2)

    return len(entries_json['entries'])


# Ottiene i campi specifici per POS
def get_pos_specific_fields(pos):
    """Restituisce i campi specifici per diverse parti del discorso"""
    if pos == "V":  # Verbo
        return [
            "infinito",
            "presente_1sg", "presente_2sg", "presente_3sg", "presente_1pl", "presente_2pl", "presente_3pl",
            "imperfetto_1sg", "imperfetto_2sg", "imperfetto_3sg", "imperfetto_1pl", "imperfetto_2pl", "imperfetto_3pl",
            "passato_remoto_1sg", "passato_remoto_2sg", "passato_remoto_3sg",
            "passato_remoto_1pl", "passato_remoto_2pl", "passato_remoto_3pl",
            "participio_presente", "participio_passato",
            "gerundio"
        ]
    elif pos == "N":  # Nome
        return ["singolare", "plurale", "diminutivo", "accrescitivo", "vezzeggiativo", "peggiorativo"]
    elif pos == "ADJ":  # Aggettivo
        return [
            "maschile_singolare", "maschile_plurale",
            "femminile_singolare", "femminile_plurale",
            "comparativo", "superlativo"
        ]
    elif pos == "ADV":  # Avverbio
        return ["base", "comparativo", "superlativo"]
    elif pos == "PRON":  # Pronome
        return [
            "soggetto_1sg", "soggetto_2sg", "soggetto_3sg", "soggetto_1pl", "soggetto_2pl", "soggetto_3pl",
            "oggetto_1sg", "oggetto_2sg", "oggetto_3sg", "oggetto_1pl", "oggetto_2pl", "oggetto_3pl"
        ]
    else:
        return ["base"]  # Default per altri POS


# Componenti UI

def sidebar_navigation(conn, status_filter):
    st.sidebar.title("Annotazione Sorianese")

    # Filtro per status
    status_filter = st.sidebar.selectbox(
        "Filtra per Status",
        ["all", "pending", "done", "skipped"],
        index=0,
        key="status_filter_dropdown"  # Assicurati che questa chiave sia unica
    )

    # Filtro per flag
    flagged_options = ["Tutti", "Solo Flaggati", "Non Flaggati"]
    flagged_select = st.sidebar.selectbox(
        "Filtra per Flag",
        flagged_options,
        index=0,
        key="flagged_filter"
    )

    flagged_filter = None
    if flagged_select == "Solo Flaggati":
        flagged_filter = True
    elif flagged_select == "Non Flaggati":
        flagged_filter = False

    # Casella di ricerca
    search_query = st.sidebar.text_input("Cerca Entries", "", key="search_query")

    # Pulsanti per navigazione avanti/indietro
    if 'selected_entry_id' in st.session_state and st.session_state['selected_entry_id']:
        prev_id = get_prev_entry_id(conn, st.session_state['selected_entry_id'],
                                    status_filter if status_filter != "all" else None)
        next_id = get_next_entry_id(conn, st.session_state['selected_entry_id'],
                                    status_filter if status_filter != "all" else None)

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if prev_id and st.button("⬅️ Precedente"):
                st.session_state['nav_target_id'] = prev_id
                st.rerun()

        with col2:
            if next_id and st.button("Successiva ➡️"):
                st.session_state['nav_target_id'] = next_id
                st.rerun()

    return status_filter, flagged_filter, search_query


def display_entry_list(conn, status_filter, flagged_filter, search_query):
    entries = get_entries_list(conn, status_filter, flagged_filter, search_query)

    if not entries:
        st.sidebar.warning("Nessuna entry corrisponde ai filtri.")
        return None

    # Mostra statistiche
    pending = sum(1 for e in entries if e['status'] == 'pending')
    done = sum(1 for e in entries if e['status'] == 'done')
    skipped = sum(1 for e in entries if e['status'] == 'skipped')

    st.sidebar.markdown(
        f"**Statistiche:** {len(entries)} entries | {pending} pending | {done} completate | {skipped} saltate")

    # Box selezione per le entries
    entry_options = [f"{e['sorianese']} ({e['status']}{' 🚩' if e['flagged'] else ''})" for e in entries]
    selected_entry_idx = st.sidebar.selectbox(
        "Seleziona Entry",
        range(len(entry_options)),
        format_func=lambda x: entry_options[x],
        key="entry_dropdown_selector"
    )

    if selected_entry_idx is not None:
        return entries[selected_entry_idx]['id']

    return None


def display_annotation_form(conn, entry_id, status_filter):
    if not entry_id:
        st.info("Seleziona un'entry dalla barra laterale per iniziare l'annotazione.")
        return

    entry = get_entry_by_id(conn, entry_id)

    if not entry:
        st.error(f"Entry con ID {entry_id} non trovata.")
        return

    st.title(f"Annota: {entry['sorianese']}")

    # Campo editabile per il termine sorianese
    entry['sorianese'] = st.text_input("Termine Sorianese", value=entry['sorianese'])

    # Mostra definizione dal dizionario se disponibile
    if entry['source'] == 'dizionario' and entry.get('definition'):
        st.info(f"**Definizione dal dizionario:** {entry['definition']}")

    # Crea tab per diverse sezioni
    main_tab, morphology_tab, examples_tab, export_tab = st.tabs(
        ["Informazioni Principali", "Morfologia", "Esempi", "Esportazione"]
    )

    # --- Tab Informazioni Principali ---
    with main_tab:
        col1, col2 = st.columns(2)

        with col1:
            # Dropdown parte del discorso
            pos_options = [
                "", "N", "V", "ADJ", "ADV", "PRON", "PREP", "CONJ", "INTERJ",
                "DET", "NUM", "S", "EXP", "IDIOM", "OTHER"
            ]
            pos_help = """
            N: Nome, V: Verbo, ADJ: Aggettivo, ADV: Avverbio,
            PRON: Pronome, PREP: Preposizione, CONJ: Congiunzione,
            INTERJ: Interiezione, DET: Determinante, NUM: Numerale,
            S: Frase, EXP: Espressione, IDIOM: Espressione idiomatica
            """
            entry['pos'] = st.selectbox(
                "Parte del Discorso",
                pos_options,
                index=pos_options.index(entry['pos']) if entry['pos'] in pos_options else 0,
                help=pos_help
            )

            # Informazioni sulla fonte (sola lettura)
            st.text_input("Fonte", value=entry['source'], disabled=True)

        with col2:
            # Mostra lo status (sola lettura, gestito automaticamente)
            status_mapped = {"pending": "In attesa", "done": "Completata", "skipped": "Saltata"}
            status_display = status_mapped.get(entry['status'], entry['status'])
            st.text_input("Status", value=status_display, disabled=True,
                          help="Lo status viene gestito automaticamente: 'Completata' se presenti POS e almeno una traduzione, 'Saltata' se premuto il pulsante Skippa, altrimenti 'In attesa'")

            # Checkbox per il flag
            entry['flagged'] = st.checkbox("Flagga questa entry", value=bool(entry['flagged']))

        # Note
        entry['notes'] = st.text_area("Note", value=entry['notes'] or "", height=100)

        # Traduzioni
        st.subheader("Traduzioni")

        # Assicura che ci sia almeno una traduzione
        if not entry['translations']:
            entry['translations'] = [{'italiano': '', 'context': '', 'register': ''}]

        # Pulsante aggiungi traduzione
        if st.button("+ Aggiungi Traduzione"):
            entry['translations'].append({'italiano': '', 'context': '', 'register': ''})

        # Mostra ogni traduzione con pulsante rimuovi
        translations_to_keep = []
        for i, trans in enumerate(entry['translations']):
            col1, col2, col3, col4 = st.columns([3, 3, 2, 1])

            with col1:
                trans['italiano'] = st.text_input(
                    f"Italiano {i + 1}",
                    value=trans.get('italiano', ''),
                    key=f"trans_ita_{i}"
                )

            with col2:
                trans['context'] = st.text_input(
                    f"Contesto {i + 1}",
                    value=trans.get('context', ''),
                    key=f"trans_ctx_{i}"
                )

            with col3:
                register_options = ["", "formale", "informale", "colloquiale", "volgare", "arcaico"]
                trans['register'] = st.selectbox(
                    f"Registro {i + 1}",
                    register_options,
                    index=register_options.index(trans.get('register', '')) if trans.get('register',
                                                                                         '') in register_options else 0,
                    key=f"trans_reg_{i}"
                )

            with col4:
                # Mostra pulsante rimuovi solo se c'è più di una traduzione
                if len(entry['translations']) > 1:
                    remove = st.button("❌", key=f"remove_trans_{i}")
                    if not remove:
                        translations_to_keep.append(trans)
                else:
                    translations_to_keep.append(trans)

        entry['translations'] = translations_to_keep

        # Varianti
        st.subheader("Varianti")

        # Assicura che ci sia almeno uno slot per variante
        if not entry.get('variants'):
            entry['variants'] = [{'variant': ''}]

        # Pulsante aggiungi variante
        if st.button("+ Aggiungi Variante"):
            entry['variants'].append({'variant': ''})

        # Mostra ogni variante con pulsante rimuovi
        variants_to_keep = []
        for i, var in enumerate(entry['variants']):
            col1, col2 = st.columns([5, 1])

            with col1:
                var['variant'] = st.text_input(
                    f"Variante {i + 1}",
                    value=var.get('variant', ''),
                    key=f"variant_{i}"
                )

            with col2:
                if len(entry['variants']) > 1:
                    remove = st.button("❌", key=f"remove_var_{i}")
                    if not remove:
                        variants_to_keep.append(var)
                else:
                    variants_to_keep.append(var)

        entry['variants'] = variants_to_keep

    # --- Tab Morfologia ---
    with morphology_tab:
        st.subheader("Forme Morfologiche")

        # Ottiene i campi in base al POS
        form_types = get_pos_specific_fields(entry['pos'])

        # Inizializza la morfologia se non esiste
        if not entry.get('morphology'):
            entry['morphology'] = {'forms': []}

        # Crea form per ogni campo
        existing_forms = {form['type']: form for form in entry['morphology']['forms'] if 'type' in form}

        # Pre-popola la lista delle forme per includere tutti i tipi previsti per il POS
        new_forms = []
        for form_type in form_types:
            if form_type in existing_forms:
                new_forms.append(existing_forms[form_type])
            else:
                new_forms.append({'form': '', 'type': form_type, 'translations': []})

        # Pulsante aggiungi forma personalizzata
        if st.button("+ Aggiungi Forma Personalizzata"):
            new_forms.append({'form': '', 'type': '', 'translations': []})

        # Mostra le forme
        forms_to_keep = []
        for i, form in enumerate(new_forms):
            st.markdown(f"##### Forma: {form['type'] or 'Personalizzata'}")

            col1, col2 = st.columns([2, 4])

            with col1:
                # Permette un tipo personalizzato se non dalla lista predefinita
                if form['type'] not in form_types:
                    form['type'] = st.text_input(
                        "Tipo",
                        value=form['type'],
                        key=f"form_type_{i}"
                    )
                else:
                    st.text_input("Tipo", value=form['type'], disabled=True, key=f"form_type_disabled_{i}")

                form['form'] = st.text_input(
                    "Forma",
                    value=form.get('form', ''),
                    key=f"form_value_{i}"
                )

            with col2:
                # Traduzioni per questa forma
                if not form.get('translations'):
                    form['translations'] = [{'translation': ''}]

                # Pulsante aggiungi traduzione
                if st.button("+ Aggiungi Traduzione", key=f"add_form_trans_{i}"):
                    form['translations'].append({'translation': ''})

                # Mostra traduzioni
                translations_to_keep = []
                for j, trans in enumerate(form['translations']):
                    col_a, col_b = st.columns([5, 1])

                    with col_a:
                        trans['translation'] = st.text_input(
                            f"Traduzione {j + 1}",
                            value=trans.get('translation', ''),
                            key=f"form_trans_{i}_{j}"
                        )

                    with col_b:
                        if len(form['translations']) > 1:
                            remove = st.button("❌", key=f"remove_form_trans_{i}_{j}")
                            if not remove:
                                translations_to_keep.append(trans)
                        else:
                            translations_to_keep.append(trans)

                form['translations'] = translations_to_keep

            # Mantiene solo le forme con contenuto o campi predefiniti
            if form['form'] or any(t.get('translation') for t in form['translations']):
                forms_to_keep.append(form)
            elif form['type'] in form_types:  # Mantiene campi predefiniti vuoti
                forms_to_keep.append(form)

            st.markdown("---")

        entry['morphology']['forms'] = forms_to_keep

    # --- Tab Esempi ---
    with examples_tab:
        st.subheader("Esempi d'Uso")

        # Assicura che ci sia almeno un esempio
        if not entry.get('examples'):
            entry['examples'] = [{'sorianese': '', 'source': '', 'translations': []}]

        # Pulsante aggiungi esempio
        if st.button("+ Aggiungi Esempio"):
            entry['examples'].append({'sorianese': '', 'source': '', 'translations': []})

        # Mostra ogni esempio
        examples_to_keep = []
        for i, example in enumerate(entry['examples']):
            st.markdown(f"##### Esempio {i + 1}")

            example['sorianese'] = st.text_area(
                "Testo Sorianese",
                value=example.get('sorianese', ''),
                height=80,
                key=f"example_sorianese_{i}"
            )

            source_options = ["", "dizionario", "poesia", "manuale", "altro"]
            example['source'] = st.selectbox(
                "Fonte",
                source_options,
                index=source_options.index(example.get('source', '')) if example.get('source',
                                                                                     '') in source_options else 0,
                key=f"example_source_{i}"
            )

            # Traduzioni per questo esempio
            if not example.get('translations'):
                example['translations'] = [{'translation': ''}]

            # Pulsante aggiungi traduzione
            if st.button("+ Aggiungi Traduzione", key=f"add_example_trans_{i}"):
                example['translations'].append({'translation': ''})

            # Mostra traduzioni
            translations_to_keep = []
            for j, trans in enumerate(example['translations']):
                col1, col2 = st.columns([5, 1])

                with col1:
                    trans['translation'] = st.text_area(
                        f"Traduzione {j + 1}",
                        value=trans.get('translation', ''),
                        height=80,
                        key=f"example_trans_{i}_{j}"
                    )

                with col2:
                    if len(example['translations']) > 1:
                        remove = st.button("❌", key=f"remove_example_trans_{i}_{j}")
                        if not remove:
                            translations_to_keep.append(trans)
                    else:
                        translations_to_keep.append(trans)

            example['translations'] = translations_to_keep

            # Pulsante rimuovi esempio
            if len(entry['examples']) > 1:
                if st.button("Rimuovi Esempio", key=f"remove_example_{i}"):
                    pass  # Filtrato non aggiungendo a examples_to_keep
                else:
                    examples_to_keep.append(example)
            else:
                examples_to_keep.append(example)

            st.markdown("---")

        entry['examples'] = examples_to_keep

    # --- Tab Esportazione ---
    with export_tab:
        st.subheader("Esporta Dati")

        # Esporta entry corrente come JSON
        if st.button("Visualizza Entry Corrente come JSON"):
            # Formatta entry per visualizzazione JSON
            display_entry = {
                "sorianese": entry['sorianese'],
                "pos": entry['pos'],
                "translations": [
                    {k: v for k, v in t.items() if k != 'id' and v}
                    for t in entry['translations'] if t.get('italiano')
                ],
                "morphology": {
                    "forms": [
                        {
                            "form": f['form'],
                            "type": f['type'],
                            "translations": [t['translation'] for t in f['translations'] if t.get('translation')]
                        }
                        for f in entry['morphology']['forms'] if f.get('form')
                    ]
                },
                "examples": [
                    {
                        "sorianese": e['sorianese'],
                        "source": e['source'],
                        "translations": [t['translation'] for t in e['translations'] if t.get('translation')]
                    }
                    for e in entry['examples'] if e.get('sorianese')
                ],
                "variants": [v['variant'] for v in entry['variants'] if v.get('variant')],
                "notes": entry['notes']
            }

            st.json(display_entry)

        # Esporta tutte le entry completate
        export_path = st.text_input("Percorso File di Esportazione", "sorianese_annotations.json")
        if st.button("Esporta Tutte le Entry Completate"):
            count = export_to_json(conn, export_path)
            st.success(f"Esportate con successo {count} entry in {export_path}")

    # Pulsanti in fondo (Salva e Skippa)
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Salva", type="primary"):
            entry['skip_pressed'] = False  # Non è skip
            if save_entry(conn, entry):
                next_id = get_next_entry_id(conn, entry['id'], status_filter if status_filter != "all" else None)
                if next_id:
                    st.session_state['nav_target_id'] = next_id
                    st.rerun()
                st.success("Entry salvata con successo!")
            else:
                st.error("Impossibile salvare l'entry.")

    with col2:
        if st.button("Skippa", type="secondary"):
            entry['skip_pressed'] = True  # Imposta lo skip flag
            if save_entry(conn, entry):
                next_id = get_next_entry_id(conn, entry['id'], status_filter if status_filter != "all" else None)
                if next_id:
                    st.session_state['nav_target_id'] = next_id
                    st.rerun()
                st.info("Entry saltata. Passaggio alla successiva.")
            else:
                st.error("Impossibile saltare l'entry.")


# App principale
def main():
    st.set_page_config(
        page_title="Strumento di Annotazione Sorianese",
        page_icon="📝",
        layout="wide"
    )

    # Inizializza database
    conn = init_db()

    # Sezione caricamento file
    with st.sidebar.expander("Caricamento File"):
        dictionary_file = st.file_uploader("Carica File Dizionario", type="txt")
        poetry_file = st.file_uploader("Carica File Poesie", type="txt")

        if dictionary_file and poetry_file:
            # Salva i file caricati
            with open("./data/annotator/dizionario.txt", "wb") as f:
                f.write(dictionary_file.getvalue())

            with open("./data/annotator/poesie.txt", "wb") as f:
                f.write(poetry_file.getvalue())

            if st.button("Elabora File"):
                count = process_files("./data/annotator/dizionario.txt", "./data/annotator/poesie.txt", conn)
                st.success(f"Elaborate {count} entry!")

    # Inizializza lo stato selezionato se non esiste
    if 'selected_entry_id' not in st.session_state:
        st.session_state['selected_entry_id'] = None

    # Valore default per status_filter
    status_filter = "all"

    # Navigazione sidebar
    status_filter, flagged_filter, search_query = sidebar_navigation(conn, status_filter)

    # Mostra lista entry e ottieni entry selezionata
    selected_entry_id = display_entry_list(conn, status_filter, flagged_filter, search_query)

    # Aggiorna l'ID dell'entry selezionata nello stato della sessione
    if selected_entry_id:
        st.session_state['selected_entry_id'] = selected_entry_id

    if 'nav_target_id' not in st.session_state:  # Nuova variabile per tracciare il target di navigazione
        st.session_state['nav_target_id'] = None

    # Dopo l'inizializzazione, controlla se un target di navigazione è stato impostato
    if st.session_state['nav_target_id']:
        st.session_state['selected_entry_id'] = st.session_state['nav_target_id']
        st.session_state['nav_target_id'] = None  # Pulisci il target dopo l'uso

    # Mostra form di annotazione per l'entry selezionata
    display_annotation_form(conn, st.session_state['selected_entry_id'], status_filter)

    # Chiude connessione database
    conn.close()

if __name__ == "__main__":
    main()

