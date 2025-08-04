import streamlit as st
import PyPDF2
import tempfile
import os
import re
import sqlite3
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import logging
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from itertools import combinations
import seaborn as sns
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
import spacy
from math import log
import uuid
import json
import pandas as pd
import yaml
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download NLTK and spaCy data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
        logger.info("NLTK data already present.")
    except LookupError:
        try:
            logger.info("Downloading NLTK punkt_tab and stopwords...")
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            logger.info("NLTK data downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download NLTK data: {str(e)}")
            st.error(f"Failed to download NLTK data: {str(e)}. Please try again or check your network.")
            return False
    return True

# Download spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy en_core_web_sm model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Download NLTK data at startup
if not download_nltk_data():
    st.stop()

# Load SciBERT NER pipeline
try:
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    scibert_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
except Exception as e:
    logger.error(f"Failed to load SciBERT NER model: {str(e)}")
    st.error(f"Failed to load SciBERT NER model: {str(e)}. Using spaCy NER only.")
    scibert_ner = None

# Default keywords in YAML format for dendrite growth in lithium-based batteries
DEFAULT_KEYWORDS_YAML = """
categories:
  thermodynamics:
    name: Thermodynamics
    keywords:
      - free energy
      - entropy
      - enthalpy
      - gibbs free energy
      - thermodynamic stability
      - phase diagram
      - chemical potential
      - nucleation
      - dendrite growth
      - thermal gradient
      - activation energy
      - boltzmann constant
      - gibbs-duhem equation
      - gibbs-thomson coefficient
      - thermodynamic equilibrium
      - thermodynamic radius
      - thermodynamic suppression regime
      - thermodynamic understanding
      - volumetric free energy
      - energy barrier (double well function)
  mechanics:
    name: Mechanics
    keywords:
      - stress
      - strain
      - mechanical stability
      - volume expansion
      - fracture
      - elastic modulus
      - plastic deformation
      - creep
      - interfacial stress
      - dendrite morphology
      - double shear modulus theory
      - plane strain modulus
      - poisson’s ratio
      - shear modulus
      - stress-induced buckling
      - stress-mediated transport
      - yield strength
      - young’s modulus
  electrochemistry:
    name: Electrochemistry
    keywords:
      - overpotential
      - butler-volmer equation
      - ion transport
      - electrolyte conductivity
      - charge transfer
      - electrodeposition
      - lithium plating
      - sei formation
      - electrochemical stability
      - coulombic efficiency
      - butler-volmer kinetics
      - cathodic transfer coefficient
      - charge continuity
      - charge migration
      - charge-transfer coefficient
      - charge-transfer resistance
      - chazalviel’s electromigration model
      - concentration polarization
      - critical current density
      - critical over-potential
      - critical stripping current
      - electrochemical deposition
      - electrochemical energy
      - electrochemical performance
      - electrochemical reaction barrier
      - electrochemical shielding
      - electrochemical stability window
      - electrochemomechanical model
      - electrode/electrolyte interface
      - electrodeposition rate
      - exchange current density
      - faraday constant
      - interfacial charge distribution
      - interfacial impedance
      - interfacial kinetics
      - interfacial plating rate
      - interfacial resistance
      - interface kinetics
      - interface stability
      - ionic concentration gradient
      - ionic conductivity
      - ionic diffusivity
      - ion mobility
      - local overpotential
      - nucleation overpotential
      - surface charge density
      - surface charge excess
      - tafel expression
      - transference number
  material_properties:
    name: Material Properties
    keywords:
      - lithium metal
      - solid electrolyte interphase
      - electrolyte
      - anode
      - cathode
      - ionic conductivity
      - diffusivity
      - surface energy
      - interface thickness
      - lithium-ion battery
      - amorphous li phase
      - amorphous polymeric interphase
      - anionic mobility
      - anionic transference
      - artificial sei
      - binary alloy
      - carbon current collector
      - cu-based interphase
      - cubic garnet structure
      - diffusion barrier
      - diffusion coefficient
      - diffusion matrix
      - gold current collector
      - grain boundary conductivity
      - grain boundary plane
      - grain size
      - graphite anode
      - h-bn layer
      - homo-lumo levels
      - li10gep2s12 electrolyte
      - li2s-p2s5 electrolyte
      - li6.4la3zr1.4ta0.6o12 (llzto)
      - li6.5la3zr1.5ta0.5o12
      - li6la3zrtao12 electrolyte
      - lialsiox electrolyte
      - lic6 phase
      - lif interface layer
      - lithium hydride
      - lix alloys
      - llzo solid electrolyte
      - llzo thickness
      - molar volume
      - multilayered sei structure
      - non-uniform sei
      - polycrystalline llzo
      - solid electrolyte interface (sei)
      - solid-liquid interface
      - solid-state electrolytes
      - surface capacitance
      - surface dipole
      - surface film coating
      - surface inhomogeneities
      - surface packing density
      - titanium sulfide (tis2)
  battery_behavior:
    name: Battery Behavior
    keywords:
      - dendrite formation
      - short-circuiting
      - capacity fade
      - cycle life
      - battery safety
      - lithium dendrite
      - dendrite suppression
      - cycling stability
      - rate capability
      - self-discharge
      - catastrophic failure
      - complete short-circuit
      - critical radius
      - cross-linked dendrites
      - dead li
      - dendrite coalescence
      - dendrite dewetting
      - dendrite growth rate
      - dendrite growth velocity
      - dendrite kinetics
      - dendrite nucleation
      - dendrite tip kinetics
      - dendrite-like spur
      - dendritic growth
      - dendritic morphology
      - diffusion-limited growth
      - filament-type growth
      - internal short circuit
      - island coalescence
      - island-type deposits
      - li microstructure evolution
      - li nucleation
      - li plating and stripping
      - li-alloy anode
      - li-ion concentration gradient
      - li-ion depletion
      - li-ion diffusivity
      - li-ion mobility
      - lithium fiber growth
      - morphological evolution
      - morphological tortuosity
      - morphological transitions
      - morphology instabilities
      - mossy li deposition
      - mossy structure
      - needle-like morphology
      - needle-like wires
      - net-like structure
      - non-spherical dendrite geometries
      - pitting during dissolution
      - planar morphology
      - pseudo-epitaxial growth
      - reaction-limited growth
      - sei breakdown
      - sei ionic resistance
      - sei nanostructure
      - sei passivation
      - sei repair
      - self-healing dendrite tactics
      - self-induced electrodissolution
      - short-circuit prediction
      - short-circuit time
      - soft short-circuit
      - sporadic bulk-plating mechanism
      - surface instability
      - thermal runaway
      - voltage-dependent growth
  characterization_techniques:
    name: Characterization Techniques
    keywords:
      - 1h mri
      - 4d x-ray tomography
      - 7li nmr
      - afm-etem
      - air-tight transfer chamber
      - archimedes’ method
      - chemical shift imaging (csi)
      - computed tomography
      - confocal raman microspectroscopy (crm)
      - cryo-electron microscopy (cryo-em)
      - cryo-fib-sem
      - cryo-tem
      - dynamic nuclear polarization (dnp)
      - ec-sem liquid cell
      - electron paramagnetic resonance (epr)
      - epr imaging (epri)
      - in-situ afm
      - in-situ eis
      - in-situ holographic interferometry
      - in-situ imaging
      - in-situ nanomechanical device
      - in-situ sem
      - in-situ tem
      - inside-out mri
      - laser scanning confocal microscopy (lscm)
      - magnetic resonance imaging (mri)
      - micro-raman spectroscopy
      - neutron depth profiling (ndp)
      - neutron radiographic imaging (nri)
      - nuclear magnetic resonance (nmr)
      - operando imaging
      - optical microscopy
      - phase contrast imaging
      - scanning electron microscopy (sem)
      - synchrotron x-ray
      - x-ray microtomography
"""

# Function to load keywords from YAML
def load_keywords(yaml_content):
    try:
        data = yaml.safe_load(yaml_content)
        if not isinstance(data, dict):
            raise ValueError("YAML content must be a dictionary")
        if 'categories' in data:
            keywords = {cat: data['categories'][cat]['keywords'] for cat in data['categories']}
        else:
            keywords = data
        for category, terms in keywords.items():
            if not isinstance(terms, list):
                raise ValueError(f"Category '{category}' must contain a list of keywords")
            keywords[category] = [str(term).lower() for term in terms]
        return keywords
    except Exception as e:
        logger.error(f"Error parsing YAML content: {str(e)}")
        return None

# SQLite database setup
def setup_databases():
    metadata_conn = sqlite3.connect("metadata.db")
    text_conn = sqlite3.connect("dendrite_universe.db")
    
    metadata_cursor = metadata_conn.cursor()
    text_cursor = text_conn.cursor()
    
    metadata_cursor.execute("""
        CREATE TABLE IF NOT EXISTS pdf_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            upload_date TEXT,
            page_count INTEGER,
            extracted_text_length INTEGER
        )
    """)
    
    text_cursor.execute("""
        CREATE TABLE IF NOT EXISTS pdf_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            full_text TEXT,
            extracted_text TEXT
        )
    """)
    
    metadata_conn.commit()
    text_conn.commit()
    return metadata_conn, text_conn

# Function to save PDF metadata and text to SQLite
def save_to_databases(file, full_text, extracted_text):
    metadata_conn, text_conn = setup_databases()
    metadata_cursor = metadata_conn.cursor()
    text_cursor = text_conn.cursor()
    
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        page_count = len(pdf_reader.pages)
        upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        metadata_cursor.execute("""
            INSERT INTO pdf_metadata (filename, upload_date, page_count, extracted_text_length)
            VALUES (?, ?, ?, ?)
        """, (file.name, upload_date, page_count, len(extracted_text)))
        
        text_cursor.execute("""
            INSERT INTO pdf_texts (filename, full_text, extracted_text)
            VALUES (?, ?, ?)
        """, (file.name, full_text, extracted_text))
        
        metadata_conn.commit()
        text_conn.commit()
    except Exception as e:
        logger.error(f"Error saving to databases: {str(e)}")
        st.error(f"Error saving to databases: {str(e)}")
    finally:
        metadata_conn.close()
        text_conn.close()

# Load IDF_APPROX (updated with new terms)
IDF_APPROX = {
    "dendrite growth": log(1000 / 50),
    "lithium metal": log(1000 / 100),
    "solid electrolyte interphase": log(1000 / 50),
    "overpotential": log(1000 / 50),
    "lithium-ion battery": log(1000 / 100),
    "electrochemical stability": log(1000 / 50),
    "volume expansion": log(1000 / 50),
    "stress": log(1000 / 50),
    "thermodynamic stability": log(1000 / 50),
    "dendritic morphology": log(1000 / 50),
    "sei formation": log(1000 / 50),
    "lithium dendrite": log(1000 / 50),
    "nucleation": log(1000 / 50),
    "in-situ tem": log(1000 / 50),
    "cryo-electron microscopy": log(1000 / 50)
}
DEFAULT_IDF = log(100000 / 10000)
PHYSICS_CATEGORIES = ["thermodynamics", "mechanics", "material_properties"]

# Visualization options
COLORMAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Greens", "Oranges", "Reds"]
NETWORK_STYLES = ["seaborn-v0_8-white", "ggplot", "bmh", "classic"]
NODE_SHAPES = ['o', 's', '^', 'v']
EDGE_STYLES = ['solid', 'dashed', 'dotted']
COLORS = ['black', 'red', 'blue', 'green']
FONT_FAMILIES = ['Arial', 'Helvetica', 'Times New Roman']
BBOX_COLORS = ['black', 'white', 'gray', 'lightgray']
LAYOUT_ALGORITHMS = ['spring', 'circular', 'kamada_kawai', 'shell']
WORD_ORIENTATIONS = ['horizontal', 'vertical', 'random']

# Initialize session state for custom stopwords
if 'custom_stopwords' not in st.session_state:
    st.session_state.custom_stopwords = "et al,figure,table"

# NER processing with spaCy and SciBERT
def perform_ner(text):
    entities = []
    
    # spaCy NER
    doc = nlp(text)
    spacy_entities = [(ent.text.lower(), ent.label_) for ent in doc.ents]
    entities.extend(spacy_entities)
    
    # SciBERT NER
    if scibert_ner:
        try:
            scibert_results = scibert_ner(text)
            scibert_entities = [(res['word'].lower(), res['entity_group']) for res in scibert_results]
            entities.extend(scibert_entities)
        except Exception as e:
            logger.error(f"Error in SciBERT NER: {str(e)}")
    
    # NLTK-based entity extraction
    sentences = sent_tokenize(text)
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        chunked = nltk.ne_chunk(tagged)
        for subtree in chunked:
            if hasattr(subtree, 'label'):
                entity = " ".join(c[0].lower() for c in subtree.leaves())
                entities.append((entity, subtree.label()))
    
    return entities

# Estimate IDF with NER consideration
def estimate_idf(term, word_freq, total_words, idf_approx, keyword_categories, nlp_model):
    if 'custom_idf' not in st.session_state:
        st.session_state.custom_idf = {}
    if term in st.session_state.custom_idf:
        return st.session_state.custom_idf[term]
    tf = word_freq.get(term, 1) / total_words
    freq_idf = log(1 / max(tf, 1e-6))
    freq_idf = min(freq_idf, 8.517)
    sim_idf = DEFAULT_IDF
    max_similarity = 0.0
    term_doc = nlp_model(term)
    for known_term in idf_approx:
        known_doc = nlp_model(known_term)
        similarity = term_doc.similarity(known_doc)
        if similarity > max_similarity and similarity > 0.7:
            max_similarity = similarity
            sim_idf = idf_approx[known_term]
    cat_idf = DEFAULT_IDF
    for category, keywords in keyword_categories.items():
        if any(k in term or term in k for k in keywords):
            cat_idfs = [idf_approx.get(k, DEFAULT_IDF) for k in keywords if k in idf_approx]
            if cat_idfs:
                cat_idf = sum(cat_idfs) / len(cat_idfs)
                break
    estimated_idf = 0.5 * freq_idf + 0.3 * cat_idf + 0.2 * sim_idf
    estimated_idf = max(2.303, min(8.517, estimated_idf))
    st.session_state.custom_idf[term] = estimated_idf
    return estimated_idf

# Extract candidate keywords with NER integration
def get_candidate_keywords(text, min_freq, min_length, use_stopwords, custom_stopwords, exclude_keywords, top_limit, tfidf_weight, use_nouns_only, include_phrases):
    stop_words = set(stopwords.words('english')) if use_stopwords else set()
    stop_words.update(['introduction', 'conclusion', 'section', 'chapter', 'the', 'a', 'an', 'preprint', 'submitted', 'manuscript'])
    stop_words.update([w.strip().lower() for w in custom_stopwords.split(",") if w.strip()])
    exclude_set = set([w.strip().lower() for w in exclude_keywords.split(",") if w.strip()])
    
    # Perform NER
    entities = perform_ner(text)
    entity_freq = Counter([ent[0] for ent in entities if len(ent[0]) >= min_length and ent[0] not in stop_words and ent[0] not in exclude_set])
    
    words = word_tokenize(text.lower())
    if use_nouns_only:
        doc = nlp(text)
        nouns = {token.text.lower() for token in doc if token.pos_ == "NOUN"}
        filtered_words = [w for w in words if w in nouns and w.isalnum() and len(w) >= min_length and w not in stop_words and w not in exclude_set]
    else:
        filtered_words = [w for w in words if w.isalnum() and len(w) >= min_length and w not in stop_words and w not in exclude_set]
    word_freq = Counter(filtered_words)
    
    # Combine entity and word frequencies
    combined_freq = word_freq + entity_freq
    
    phrases = []
    if include_phrases:
        doc = nlp(text)
        raw_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1 and len(chunk.text) >= min_length]
        phrases = [clean_phrase(phrase, stop_words) for phrase in raw_phrases if clean_phrase(phrase, stop_words)]
        phrases = [p for p in phrases if p not in stop_words and p not in exclude_set]
        phrase_freq = Counter(phrases)
        phrases = [(p, f) for p, f in phrase_freq.items() if f >= min_freq]
    
    total_words = len(word_tokenize(text))
    tfidf_scores = {}
    idf_sources = {}
    for term, freq in combined_freq.items():
        if freq < min_freq:
            continue
        tf = freq / total_words
        if term in IDF_APPROX:
            idf = IDF_APPROX[term]
            source = "JSON"
        else:
            idf = estimate_idf(term, combined_freq, total_words, IDF_APPROX, KEYWORD_CATEGORIES, nlp)
            source = "Estimated"
        tfidf_scores[term] = tf * idf * tfidf_weight
        idf_sources[term] = {"idf": idf, "source": source, "frequency": freq}
    
    for phrase, freq in phrases:
        if freq < min_freq:
            continue
        tf = freq / total_words
        if phrase in IDF_APPROX:
            idf = IDF_APPROX[phrase]
            source = "JSON"
        else:
            idf = estimate_idf(phrase, phrase_freq, total_words, IDF_APPROX, KEYWORD_CATEGORIES, nlp)
            source = "Estimated"
        tfidf_scores[phrase] = tf * idf * tfidf_weight
        idf_sources[phrase] = {"idf": idf, "source": source, "frequency": freq}
    
    for term in tfidf_scores:
        for category, keywords in KEYWORD_CATEGORIES.items():
            if term in keywords and category in PHYSICS_CATEGORIES:
                tfidf_scores[term] *= 1.5
    
    if tfidf_weight > 0:
        ranked_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_limit]
    else:
        ranked_terms = [(w, f) for w, f in combined_freq.most_common(top_limit) if f >= min_freq]
        ranked_terms += phrases[:top_limit - len(ranked_terms)]
    
    categorized_keywords = {cat: [] for cat in KEYWORD_CATEGORIES}
    term_to_category = {}
    for term, score in ranked_terms:
        if term in exclude_set:
            continue
        assigned = False
        for category, keywords in KEYWORD_CATEGORIES.items():
            if term in keywords:
                categorized_keywords[category].append((term, score))
                term_to_category[term] = category
                assigned = True
                break
            elif " " in term:
                if any(k == term or term.startswith(k + " ") or term.endswith(" " + k) for k in keywords):
                    categorized_keywords[category].append((term, score))
                    term_to_category[term] = category
                    assigned = True
                    break
        if not assigned:
            categorized_keywords["material_properties"].append((term, score))
            term_to_category[term] = "material_properties"
    
    return categorized_keywords, combined_freq, phrases, tfidf_scores, term_to_category, idf_sources

# Extract text from PDF
def extract_text_from_pdf(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        pdf_reader = PyPDF2.PdfReader(tmp_file_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        os.unlink(tmp_file_path)
        return text if text.strip() else f"No text extracted from {file.name}."
    except Exception as e:
        logger.error(f"Error extracting text from {file.name}: {str(e)}")
        return f"Error extracting text from {file.name}: {str(e)}"

# Extract text between phrases
def extract_text_between_phrases(text, start_phrase, end_phrase, file_name):
    try:
        start_idx = text.find(start_phrase)
        end_idx = text.find(end_phrase, start_idx + len(start_phrase))
        if start_idx == -1 or end_idx == -1:
            return f"Specified phrases not found in {file_name}."
        return text[start_idx:end_idx + len(end_phrase)]
    except Exception as e:
        logger.error(f"Error extracting text between phrases in {file_name}: {str(e)}")
        return f"Error extracting text between phrases in {file_name}: {str(e)}"

# Clean phrase for processing
def clean_phrase(phrase, stop_words):
    words = phrase.split()
    while words and words[0].lower() in stop_words:
        words = words[1:]
    while words and words[-1].lower() in stop_words:
        words = words[:-1]
    return " ".join(words).strip()

# Generate word cloud
def generate_word_cloud(text, selected_keywords, tfidf_scores, selection_criteria, colormap, title_font_size, caption_font_size, font_step, word_orientation, background_color, contour_width, contour_color):
    try:
        stop_words = set(stopwords.words('english'))
        stop_words.update(['the', 'a', 'an', 'preprint', 'submitted', 'manuscript'])
        stop_words.update([w.strip().lower() for w in st.session_state.custom_stopwords.split(",") if w.strip()])
        processed_text = text.lower()
        keyword_map = {}
        for keyword in selected_keywords:
            internal_key = keyword.replace(" ", "_")
            processed_text = processed_text.replace(keyword, internal_key)
            keyword_map[internal_key] = keyword
        words = word_tokenize(processed_text)
        filtered_words = [keyword_map.get(word, word) for word in words if keyword_map.get(word, word) in selected_keywords]
        if not filtered_words:
            return None, "No valid words or phrases found for word cloud after filtering."
        frequencies = {word: tfidf_scores.get(word, 1.0) for word in filtered_words}
        max_freq = max(frequencies.values(), default=1.0)
        frequencies = {word: freq / max_freq for word, freq in frequencies.items()}
        wordcloud = WordCloud(
            width=1600, height=800,
            background_color=background_color,
            min_font_size=8,
            max_font_size=200,
            font_step=font_step,
            prefer_horizontal=1.0 if word_orientation == 'horizontal' else 0.0 if word_orientation == 'vertical' else 0.5,
            colormap=colormap,
            contour_width=contour_width,
            contour_color=contour_color,
            margin=10
        ).generate_from_frequencies(frequencies)
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(16, 8), dpi=400)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Word Cloud of Dendrite Growth Keywords", fontsize=title_font_size, pad=20, fontweight='bold')
        caption = f"Word Cloud generated with: {selection_criteria}"
        plt.figtext(0.5, 0.02, caption, ha="center", fontsize=caption_font_size, wrap=True, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        return fig, None
    except Exception as e:
        logger.error(f"Error generating word cloud: {str(e)}")
        return None, f"Error generating word cloud: {str(e)}"

# Generate bibliometric network with NER relationships
def generate_bibliometric_network(text, selected_keywords, tfidf_scores, label_font_size, selection_criteria, node_colormap, edge_colormap, network_style, line_thickness, node_alpha, edge_alpha, title_font_size, caption_font_size, node_size_scale, node_shape, node_linewidth, node_edgecolor, edge_style, label_font_color, label_font_family, label_bbox_facecolor, label_bbox_alpha, layout_algorithm, label_rotation, label_offset):
    try:
        stop_words = set(stopwords.words('english'))
        stop_words.update(['the', 'a', 'an', 'preprint', 'submitted', 'manuscript'])
        stop_words.update([w.strip().lower() for w in st.session_state.custom_stopwords.split(",") if w.strip()])
        processed_text = text.lower()
        keyword_map = {}
        for keyword in selected_keywords:
            internal_key = keyword.replace(" ", "_")
            processed_text = processed_text.replace(keyword, internal_key)
            keyword_map[internal_key] = keyword
        words = word_tokenize(processed_text)
        filtered_words = [keyword_map.get(word, word) for word in words if keyword_map.get(word, word) in selected_keywords]
        word_freq = Counter(filtered_words)
        if not word_freq:
            return None, "No valid words or phrases found for bibliometric network."
        top_words = [word for word, freq in word_freq.most_common(20)]
        
        # Build co-occurrence network with NER relationships
        sentences = sent_tokenize(text.lower())
        co_occurrences = Counter()
        entities = perform_ner(text)
        entity_set = set([ent[0] for ent in entities if ent[0] in selected_keywords])
        for sentence in sentences:
            processed_sentence = sentence
            for keyword in selected_keywords:
                processed_sentence = processed_sentence.replace(keyword, keyword.replace(" ", "_"))
            words_in_sentence = [keyword_map.get(word, word) for word in word_tokenize(processed_sentence) if keyword_map.get(word, word) in top_words]
            sentence_entities = [ent[0] for ent in entities if ent[0] in sentence.lower() and ent[0] in selected_keywords]
            for pair in combinations(set(words_in_sentence + sentence_entities), 2):
                co_occurrences[tuple(sorted(pair))] += 1
        
        G = nx.Graph()
        for word, freq in word_freq.most_common(20):
            G.add_node(word, size=freq)
        for (word1, word2), weight in co_occurrences.items():
            if word1 in top_words and word2 in top_words:
                G.add_edge(word1, word2, weight=weight)
        
        communities = greedy_modularity_communities(G)
        node_colors = {}
        cmap = plt.cm.get_cmap(node_colormap)
        palette = cmap(np.linspace(0.2, 0.8, max(1, len(communities))))
        for i, community in enumerate(communities):
            for node in community:
                node_colors[node] = palette[i]
        
        edge_weights = [G.edges[edge]['weight'] for edge in G.edges]
        max_weight = max(edge_weights, default=1)
        edge_widths = [line_thickness * (1 + 2 * np.log1p(weight / max_weight)) for weight in edge_weights]
        edge_cmap = plt.cm.get_cmap(edge_colormap)
        edge_colors = [edge_cmap(weight / max_weight) for weight in edge_weights]
        
        pos = nx.spring_layout(G, k=0.8, seed=42) if layout_algorithm == 'spring' else nx.circular_layout(G)
        
        plt.style.use(network_style)
        plt.rcParams['font.family'] = label_font_family
        fig, ax = plt.subplots(figsize=(16, 12), dpi=400)
        node_sizes = [G.nodes[node]['size'] * node_size_scale * 20 for node in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=[node_colors[node] for node in G.nodes], node_shape=node_shape, edgecolors=node_edgecolor, linewidths=node_linewidth, alpha=node_alpha, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, style=edge_style, alpha=edge_alpha, ax=ax)
        label_pos = {node: (pos[node][0] + label_offset * np.cos(np.radians(label_rotation)), pos[node][1] + label_offset * np.sin(np.radians(label_rotation))) for node in G.nodes}
        nx.draw_networkx_labels(G, label_pos, font_size=label_font_size, font_color=label_font_color, font_family=label_font_family, font_weight='bold', bbox=dict(facecolor=label_bbox_facecolor, alpha=label_bbox_alpha, edgecolor='none', boxstyle='round,pad=0.3'), ax=ax)
        ax.set_title("Knowledge Graph of Dendrite Growth in Lithium Batteries", fontsize=title_font_size, pad=20, fontweight='bold')
        caption = f"Network generated with: {selection_criteria}"
        plt.figtext(0.5, 0.02, caption, ha="center", fontsize=caption_font_size, wrap=True, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        ax.set_facecolor('#fafafa')
        return fig, None
    except Exception as e:
        logger.error(f"Error generating bibliometric network: {str(e)}")
        return None, f"Error generating bibliometric network: {str(e)}"

# Generate radar chart
def generate_radar_chart(selected_keywords, values, title, selection_criteria, colormap, max_keywords, label_font_size, line_thickness, fill_alpha, title_font_size, caption_font_size, label_rotation, label_offset, grid_color, grid_style, grid_thickness):
    try:
        if len(selected_keywords) < 3:
            return None, "At least 3 keywords/phrases are required for a radar chart."
        keyword_values = [(k, values.get(k, 0)) for k in selected_keywords if k in values]
        if not keyword_values:
            return None, "No valid keywords/phrases with values for radar chart."
        keyword_values = sorted(keyword_values, key=lambda x: x[1], reverse=True)[:max_keywords]
        labels, vals = zip(*keyword_values)
        num_vars = len(labels)
        max_val = max(vals, default=1)
        vals = [v / max_val for v in vals] if max_val > 0 else vals
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        vals = list(vals) + [vals[0]]
        angles += angles[:1]
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(10, 10), dpi=400, subplot_kw=dict(polar=True))
        cmap = plt.cm.get_cmap(colormap)
        line_color = cmap(0.9)
        fill_color = cmap(0.5)
        ax.plot(angles, vals, color=line_color, linewidth=line_thickness, linestyle='solid')
        ax.fill(angles, vals, color=fill_color, alpha=fill_alpha)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=label_font_size, rotation=label_rotation)
        for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
            x, y = label.get_position()
            lab = ax.text(angle, 1.1 + label_offset, label.get_text(), transform=ax.get_transform(), ha='center', va='center', fontsize=label_font_size, color='black')
            lab.set_rotation(angle * 180 / np.pi + label_rotation)
        ax.set_rlabel_position(0)
        ax.yaxis.grid(True, color=grid_color, linestyle=grid_style, linewidth=grid_thickness, alpha=0.7)
        ax.xaxis.grid(True, color=grid_color, linestyle=grid_style, linewidth=grid_thickness, alpha=0.7)
        ax.set_title(title, fontsize=title_font_size, pad=30, fontweight='bold')
        caption = f"{title} generated with: {selection_criteria}"
        plt.figtext(0.5, 0.02, caption, ha="center", fontsize=caption_font_size, wrap=True, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        ax.set_facecolor('#fafafa')
        return fig, None
    except Exception as e:
        logger.error(f"Error generating radar chart: {str(e)}")
        return None, f"Error generating radar chart: {str(e)}"

# Save figure
def save_figure(fig, filename):
    try:
        fig.savefig(filename + ".png", dpi=400, bbox_inches='tight', format='png')
        fig.savefig(filename + ".svg", bbox_inches='tight', format='svg')
        return True
    except Exception as e:
        logger.error(f"Error saving figure: {str(e)}")
        return False

# Clear selections
if 'clear_selections' not in st.session_state:
    st.session_state.clear_selections = False

def clear_selections():
    st.session_state.clear_selections = True
    for key in list(st.session_state.keys()):
        if key.startswith("multiselect_"):
            del st.session_state[key]

# Streamlit app
st.set_page_config(page_title="Dendrite Growth Knowledge Graph Visualizer", layout="wide")
st.title("Dendrite Growth in Lithium-Based Batteries Visualizer")
st.markdown("""
Upload one or more PDF files to extract text, perform NER, and generate visualizations 
(word cloud, knowledge graph, radar charts) for thermodynamics, mechanics, electrochemistry, 
material properties, battery behavior, and characterization techniques of dendrite growth in lithium-based batteries. 
Optionally upload a YAML file to define custom keyword categories.
""")

# YAML file uploader
yaml_file = st.file_uploader("Upload a YAML file with keyword categories (optional)", type="yaml")

# Load keywords
if yaml_file:
    yaml_content = yaml_file.read().decode("utf-8")
    KEYWORD_CATEGORIES = load_keywords(yaml_content)
    if KEYWORD_CATEGORIES is None:
        st.error("Invalid YAML file. Using default keywords.")
        KEYWORD_CATEGORIES = load_keywords(DEFAULT_KEYWORDS_YAML)
else:
    KEYWORD_CATEGORIES = load_keywords(DEFAULT_KEYWORDS_YAML)

# Multiple PDF file uploader
uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

# Input fields
start_phrase = st.text_input("Enter the desired initial phrase", "Introduction")
end_phrase = st.text_input("Enter the desired final phrase", "Conclusion")
custom_stopwords_input = st.text_input("Custom stopwords (comma-separated)", value=st.session_state.custom_stopwords)
exclude_keywords_input = st.text_input("Exclude keywords/phrases (comma-separated)", "preprint,submitted,manuscript")
st.session_state.custom_stopwords = custom_stopwords_input

if uploaded_files:
    with st.spinner("Processing PDFs and saving to databases..."):
        combined_text = ""
        extraction_errors = []
        for uploaded_file in uploaded_files:
            full_text = extract_text_from_pdf(uploaded_file)
            if "Error" in full_text:
                extraction_errors.append(full_text)
                continue
            selected_text = extract_text_between_phrases(full_text, start_phrase, end_phrase, uploaded_file.name)
            if "Error" in selected_text or "not found" in selected_text:
                extraction_errors.append(selected_text)
            else:
                combined_text += f"\n\n--- Text from {uploaded_file.name} ---\n\n{selected_text}"
                save_to_databases(uploaded_file, full_text, selected_text)
        
        if extraction_errors:
            for error in extraction_errors:
                st.error(error)
        
        if not combined_text.strip():
            st.error("No valid text extracted from any PDF. Please check the files or phrases.")
            st.stop()
        
        st.subheader("Extracted Text Between Phrases (All PDFs)")
        st.text_area("Combined Selected Text", combined_text, height=200)
        
        st.subheader("Configure Keyword Selection Criteria")
        min_freq = st.slider("Minimum frequency", min_value=1, max_value=10, value=1)
        min_length = st.slider("Minimum length", min_value=3, max_value=30, value=10)
        use_stopwords = st.checkbox("Use stopword filtering", value=True)
        top_limit = st.slider("Top limit (max keywords)", min_value=10, max_value=100, value=50, step=10)
        tfidf_weight = st.slider("TF-IDF weighting", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
        use_nouns_only = st.checkbox("Filter for nouns only", value=False)
        include_phrases = st.checkbox("Include multi-word phrases", value=True, disabled=True)
        criteria_parts = [
            f"frequency ≥ {min_freq}",
            f"length ≥ {min_length}",
            "stopwords " + ("enabled" if use_stopwords else "disabled"),
            f"custom stopwords: {custom_stopwords_input}" if custom_stopwords_input.strip() else "no custom stopwords",
            f"excluded keywords: {exclude_keywords_input}" if exclude_keywords_input.strip() else "no excluded keywords",
            f"top {top_limit} keywords",
            f"TF-IDF weight: {tfidf_weight}",
            "nouns only" if use_nouns_only else "all parts of speech",
            "multi-word phrases included"
        ]
        st.subheader("Select Keywords and Phrases by Category")
        if st.button("Clear All Selections"):
            clear_selections()
        try:
            categorized_keywords, word_freq, phrases, tfidf_scores, term_to_category, idf_sources = get_candidate_keywords(
                combined_text, min_freq, min_length, use_stopwords, custom_stopwords_input, exclude_keywords_input,
                top_limit, tfidf_weight, use_nouns_only, include_phrases
            )
        except Exception as e:
            st.error(f"Error processing keywords: {str(e)}")
            logger.error(f"Error in get_candidate_keywords: {str(e)}")
            st.stop()
        selected_keywords = []
        for category in KEYWORD_CATEGORIES:
            keywords = [term for term, _ in categorized_keywords.get(category, [])]
            with st.expander(f"{category} ({len(keywords)} keywords/phrases)"):
                if keywords:
                    selected = st.multiselect(
                        f"Select keywords from {category}",
                        options=keywords,
                        default=[] if st.session_state.clear_selections else keywords[:min(5, len(keywords))],
                        key=f"multiselect_{category}_{uuid.uuid4()}"
                    )
                    selected_keywords.extend(selected)
                else:
                    st.write("No keywords or phrases found for this category.")
        st.session_state.clear_selections = False
        with st.expander("Debug Information"):
            if word_freq:
                st.write("Single Words and Entities (Top 20):", word_freq.most_common(20))
            if phrases:
                st.write("Extracted Phrases (Top 20):", phrases[:20])
            if categorized_keywords:
                st.write("Categorized Keywords:", {k: [t[0] for t in v] for k, v in categorized_keywords.items()})
        with st.expander("IDF Source Details"):
            if idf_sources:
                idf_data = [
                    {
                        "Term": term,
                        "Frequency": idf_sources[term]["frequency"],
                        "TF-IDF Score": round(tfidf_scores.get(term, 0), 3),
                        "IDF Value": round(idf_sources[term]["idf"], 3),
                        "Source": idf_sources[term]["source"]
                    }
                    for term in tfidf_scores
                ]
                idf_df = pd.DataFrame(idf_data).sort_values(by=["Source", "TF-IDF Score"], ascending=[True, False])
                def highlight_json(row):
                    return ["font-weight: bold" if row["Source"] == "JSON" else "" for _ in row]
                source_filter = st.selectbox("Filter by IDF Source", ["All", "JSON", "Estimated"])
                if source_filter != "All":
                    idf_df = idf_df[idf_df["Source"] == source_filter]
                styled_df = idf_df.style.apply(highlight_json, axis=1).format({"TF-IDF Score": "{:.3f}", "IDF Value": "{:.3f}"})
                st.dataframe(styled_df, use_container_width=True)
                st.download_button(
                    label="Download IDF Sources (JSON)",
                    data=json.dumps(idf_data, indent=4),
                    file_name="idf_sources.json",
                    mime="application/json"
                )
        if not selected_keywords:
            st.error("Please select at least one keyword or phrase.")
            st.stop()
        st.subheader("Visualization Settings")
        st.markdown("### General Visualization Settings")
        label_font_size = st.slider("Label font size", min_value=8, max_value=24, value=12, step=1)
        line_thickness = st.slider("Line thickness", min_value=0.5, max_value=6.0, value=2.5, step=0.5)
        title_font_size = st.slider("Title font size", min_value=10, max_value=24, value=16, step=1)
        caption_font_size = st.slider("Caption font size", min_value=8, max_value=16, value=10, step=1)
        transparency = st.slider("Transparency (nodes, edges, fills)", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
        label_rotation = st.slider("Label rotation (degrees)", min_value=0, max_value=90, value=0, step=5)
        label_offset = st.slider("Label offset", min_value=0.0, max_value=0.1, value=0.02, step=0.01)
        criteria_parts.extend([
            f"label font size: {label_font_size}",
            f"line thickness: {line_thickness}",
            f"title font size: {title_font_size}",
            f"caption font size: {caption_font_size}",
            f"transparency: {transparency}",
            f"label rotation: {label_rotation}°",
            f"label offset: {label_offset}"
        ])
        st.markdown("### Word Cloud Settings")
        wordcloud_colormap = st.selectbox("Select colormap for word cloud", options=COLORMAPS, index=0)
        word_orientation = st.selectbox("Word orientation", options=WORD_ORIENTATIONS, index=0)
        font_step = st.slider("Font size step", min_value=1, max_value=10, value=2, step=1)
        background_color = st.selectbox("Background color", options=['white', 'black', 'lightgray'], index=0)
        contour_width = st.slider("Contour width", min_value=0.0, max_value=5.0, value=0.0, step=0.5)
        contour_color = st.selectbox("Contour color", options=COLORS, index=0)
        criteria_parts.extend([
            f"word cloud colormap: {wordcloud_colormap}",
            f"word orientation: {word_orientation}",
            f"font step: {font_step}",
            f"background color: {background_color}",
            f"contour width: {contour_width}",
            f"contour color: {contour_color}"
        ])
        st.markdown("### Network Settings")
        network_style = st.selectbox("Select style for network", options=NETWORK_STYLES, index=0)
        node_colormap = st.selectbox("Select colormap for network nodes", options=COLORMAPS, index=0)
        edge_colormap = st.selectbox("Select colormap for network edges", options=COLORMAPS, index=7)
        layout_algorithm = st.selectbox("Select layout algorithm", options=LAYOUT_ALGORITHMS, index=0)
        node_size_scale = st.slider("Node size scale", min_value=10, max_value=100, value=50, step=5)
        node_shape = st.selectbox("Node shape", options=NODE_SHAPES, index=0)
        node_linewidth = st.slider("Node border thickness", min_value=0.5, max_value=5.0, value=1.5, step=0.5)
        node_edgecolor = st.selectbox("Node border color", options=COLORS, index=0)
        edge_style = st.selectbox("Edge style", options=EDGE_STYLES, index=0)
        label_font_color = st.selectbox("Label font color", options=COLORS, index=0)
        label_font_family = st.selectbox("Label font family", options=FONT_FAMILIES, index=0)
        label_bbox_facecolor = st.selectbox("Label background color", options=BBOX_COLORS, index=0)
        label_bbox_alpha = st.slider("Label background transparency", min_value=0.1, max_value=1.0, value=0.6, step=0.1)
        criteria_parts.extend([
            f"network style: {network_style}",
            f"node colormap: {node_colormap}",
            f"edge colormap: {edge_colormap}",
            f"layout: {layout_algorithm}",
            f"node size scale: {node_size_scale}",
            f"node shape: {node_shape}",
            f"node border thickness: {node_linewidth}",
            f"node border color: {node_edgecolor}",
            f"edge style: {edge_style}",
            f"label font color: {label_font_color}",
            f"label font family: {label_font_family}",
            f"label background color: {label_bbox_facecolor}",
            f"label background transparency: {label_bbox_alpha}"
        ])
        st.markdown("### Radar Chart Settings")
        radar_max_keywords = st.slider("Number of keywords for radar charts", min_value=3, max_value=12, value=6, step=1)
        freq_radar_colormap = st.selectbox("Colormap for frequency radar chart", options=COLORMAPS, index=0)
        tfidf_radar_colormap = st.selectbox("Colormap for TF-IDF radar chart", options=COLORMAPS, index=0)
        grid_color = st.selectbox("Grid color", options=COLORS, index=0)
        grid_style = st.selectbox("Grid style", options=EDGE_STYLES, index=0)
        grid_thickness = st.slider("Grid thickness", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
        criteria_parts.extend([
            f"radar max keywords: {radar_max_keywords}",
            f"frequency radar colormap: {freq_radar_colormap}",
            f"tfidf radar colormap: {tfidf_radar_colormap}",
            f"grid color: {grid_color}",
            f"grid style: {grid_style}",
            f"grid thickness: {grid_thickness}"
        ])
        selection_criteria = ", ".join(criteria_parts)
        st.subheader("Word Cloud")
        wordcloud_fig, wordcloud_error = generate_word_cloud(
            combined_text, selected_keywords, tfidf_scores, selection_criteria,
            wordcloud_colormap, title_font_size, caption_font_size, font_step,
            word_orientation, background_color, contour_width, contour_color
        )
        if wordcloud_error:
            st.error(wordcloud_error)
        elif wordcloud_fig:
            st.pyplot(wordcloud_fig)
            if save_figure(wordcloud_fig, "wordcloud"):
                st.download_button(
                    label="Download Word Cloud (PNG)",
                    data=open("wordcloud.png", "rb").read(),
                    file_name="wordcloud.png",
                    mime="image/png"
                )
                st.download_button(
                    label="Download Word Cloud (SVG)",
                    data=open("wordcloud.svg", "rb").read(),
                    file_name="wordcloud.svg",
                    mime="image/svg+xml"
                )
        st.subheader("Knowledge Graph")
        network_fig, network_error = generate_bibliometric_network(
            combined_text, selected_keywords, tfidf_scores, label_font_size, selection_criteria,
            node_colormap, edge_colormap, network_style, line_thickness, transparency, transparency,
            title_font_size, caption_font_size, node_size_scale, node_shape, node_linewidth,
            node_edgecolor, edge_style, label_font_color, label_font_family, label_bbox_facecolor,
            label_bbox_alpha, layout_algorithm, label_rotation, label_offset
        )
        if network_error:
            st.error(network_error)
        elif network_fig:
            st.pyplot(network_fig)
            if save_figure(network_fig, "network"):
                st.download_button(
                    label="Download Network (PNG)",
                    data=open("network.png", "rb").read(),
                    file_name="network.png",
                    mime="image/png"
                )
                st.download_button(
                    label="Download Network (SVG)",
                    data=open("network.svg", "rb").read(),
                    file_name="network.svg",
                    mime="image/svg+xml"
                )
        st.subheader("Frequency Radar Chart")
        freq_radar_fig, freq_radar_error = generate_radar_chart(
            selected_keywords, word_freq, "Keyword/Phrase Frequency Comparison",
            selection_criteria, freq_radar_colormap, radar_max_keywords,
            label_font_size, line_thickness, transparency, title_font_size, caption_font_size,
            label_rotation, label_offset, grid_color, grid_style, grid_thickness
        )
        if freq_radar_error:
            st.error(freq_radar_error)
        elif freq_radar_fig:
            st.pyplot(freq_radar_fig)
            if save_figure(freq_radar_fig, "freq_radar"):
                st.download_button(
                    label="Download Frequency Radar (PNG)",
                    data=open("freq_radar.png", "rb").read(),
                    file_name="freq_radar.png",
                    mime="image/png"
                )
                st.download_button(
                    label="Download Frequency Radar (SVG)",
                    data=open("freq_radar.svg", "rb").read(),
                    file_name="freq_radar.svg",
                    mime="image/svg+xml"
                )
        st.subheader("TF-IDF Radar Chart")
        tfidf_radar_fig, tfidf_radar_error = generate_radar_chart(
            selected_keywords, tfidf_scores, "Keyword/Phrase TF-IDF Comparison",
            selection_criteria, tfidf_radar_colormap, radar_max_keywords,
            label_font_size, line_thickness, transparency, title_font_size, caption_font_size,
            label_rotation, label_offset, grid_color, grid_style, grid_thickness
        )
        if tfidf_radar_error:
            st.error(tfidf_radar_error)
        elif tfidf_radar_fig:
            st.pyplot(tfidf_radar_fig)
            if save_figure(tfidf_radar_fig, "tfidf_radar"):
                st.download_button(
                    label="Download TF-IDF Radar (PNG)",
                    data=open("tfidf_radar.png", "rb").read(),
                    file_name="tfidf_radar.png",
                    mime="image/png"
                )
                st.download_button(
                    label="Download TF-IDF Radar (SVG)",
                    data=open("tfidf_radar.svg", "rb").read(),
                    file_name="tfidf_radar.svg",
                    mime="image/svg+xml"
                )
        st.markdown("---")
        st.markdown("Enhanced with Streamlit, PyPDF2, WordCloud, NetworkX, NLTK, spaCy, Transformers, Matplotlib, Seaborn, PyYAML, and SQLite for dendrite growth knowledge graph analysis.")