import streamlit as st
import arxiv
import math
import json
import time
import logging
from collections import Counter
import pandas as pd
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Data-driven study of keywords-keyphrases",
    layout="wide"
)

# Title and description
st.title("Inverse Document Frequency Calculation for Selected Keywords-Keyphrases of Lithium-Ion Batteries and Phase Field Modeling")
st.markdown("""
This Streamlit app computes Inverse Document Frequency (IDF) values for terms and phrases
from the arXiv database, tailored for lithium-ion battery and phase field modeling research. 
It queries arXiv for document counts, calculates IDFs, and generates a downloadable `idf_approx.json` 
file to replace the hardcoded `IDF_APPROX` table in your main app. The app focuses on terms relevant 
to lithium-ion batteries, phase field methods, dendrite growth, and related electrochemical phenomena.
Previous computation results are displayed during new computations, with warnings for new or uncomputed keywords.
""")

# Cache function to load previous JSON
@st.cache_data
def load_previous_json(cache_path="previous_idf_approx.json"):
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"Error loading cached JSON from {cache_path}: {str(e)}")
        return None

# Define terms from IDF_APPROX (initial set of core terms)
IDF_APPROX_TERMS = [
    "study", "analysis", "results", "method", "experiment",
    "electrolyte", "anode", "cathode", "dendrite", "phase field",
    "lithium-ion battery", "solid electrolyte interphase", "electrode",
    "charge-discharge", "capacity fading", "electrochemical modeling",
    "lithium plating", "phase field method", "dendrite growth",
    "ionic conductivity", "battery degradation", "solid-state battery",
    "electrolyte decomposition", "interface stability", "lithium diffusion"
]

# Define terms from KEYWORD_CATEGORIES (organized by category, deduplicated)
KEYWORD_CATEGORIES_TERMS = [
    # Materials
    "lithium", "electrolyte", "anode", "cathode", "separator",
    "solid electrolyte", "liquid electrolyte", "polymer electrolyte",
    "lithium cobalt oxide", "lithium iron phosphate", "graphite anode",
    "silicon anode", "lithium metal", "solid-state electrolyte",
    "lithium titanate", "nickel manganese cobalt", "lithium sulfur",
    "garnet electrolyte", "sulfide electrolyte", "oxide electrolyte",
    "lithium-ion battery", "solid electrolyte interphase", "sei layer",
    # Methods
    "phase field method", "electrochemical modeling", "molecular dynamics",
    "density functional theory", "finite element analysis", "spectroscopy",
    "impedance spectroscopy", "cyclic voltammetry", "galvanostatic cycling",
    "operando imaging", "x-ray diffraction", "scanning electron microscopy",
    "transmission electron microscopy", "electrochemical impedance",
    "in-situ characterization", "computational modeling", "continuum modeling",
    "phase field simulation", "lattice boltzmann method", "monte carlo simulation",
    # Physical Phenomena
    "dendrite growth", "lithium plating", "electrolyte decomposition",
    "interface stability", "lithium diffusion", "ionic conductivity",
    "charge transfer", "electrode polarization", "sei formation",
    "thermal runaway", "electrochemical reaction", "ion transport",
    "concentration gradient", "overpotential", "capacity fading",
    "battery degradation", "side reaction", "electrode cracking",
    "lithium dendrite", "electrolyte oxidation",
    # Properties
    "capacity", "energy density", "power density", "cycle life",
    "ionic conductivity", "electronic conductivity", "thermal stability",
    "mechanical stability", "electrochemical stability", "specific capacity",
    "coulombic efficiency", "voltage hysteresis", "rate capability",
    "capacity retention", "mechanical stress", "interface resistance",
    "electrolyte conductivity", "diffusion coefficient", "charge-discharge rate",
    "sei thickness",
    # Other
    "battery safety", "fast charging", "high-energy battery",
    "electrode design", "electrolyte optimization", "battery management",
    "sustainable battery", "recyclable battery", "solid-state battery",
    "phase field modeling", "dendrite suppression", "electrode-electrolyte interface",
    "battery performance", "thermal management", "electrochemical kinetics"
]

# Combine and deduplicate terms
terms = list(set(IDF_APPROX_TERMS + KEYWORD_CATEGORIES_TERMS))
logger.info(f"Total unique terms to query: {len(terms)}")
st.write(f"**Total unique terms to query**: {len(terms)}")

# Corpus size (estimated for lithium-ion battery and related research in arXiv)
N = 50000  # Approx. battery and electrochemistry-related papers
st.write(f"**Estimated corpus size (N)**: {N} documents (electrochemistry and materials science)")

# Initialize IDF dictionary
IDF_APPROX = {}

# Load previous JSON to identify new keywords
previous_json = load_previous_json()
previous_terms = set(previous_json.keys()) if previous_json else set()
new_terms = [term for term in terms if term not in previous_terms]

# Create initial JSON with warnings for new or uncomputed terms
display_json = {}
for term in terms:
    if term in new_terms or not previous_json:
        display_json[term] = {
            "idf": round(math.log(N / 10000), 3),
            "status": "warning: Using default IDF (not yet computed)"
        }
    else:
        display_json[term] = {
            "idf": round(previous_json[term], 3),
            "status": "computed: IDF from previous run"
        }

# Display previous JSON (or initial JSON with warnings) before computation
st.subheader("Previous or Initial Computation Results")
st.json(display_json)
st.markdown("*Note: New or uncomputed keywords are marked with a warning and use a default IDF. These will be updated during computation.*")

# Button to start computation
if st.button("Compute IDFs from arXiv"):
    with st.spinner("Querying arXiv and computing IDFs..."):
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []

        # Query arXiv
        client = arxiv.Client()
        for i, term in enumerate(terms):
            try:
                # Format query (wrap phrases in quotes)
                query_term = f'"{term}"' if " " in term else term
                # Search in materials science, physics, and chemistry
                query = f"{query_term} cat:cond-mat.mtrl-sci OR cat:physics.chem-ph OR cat:physics.app-ph"
                search = arxiv.Search(
                    query=query,
                    max_results=1000,  # Limit to avoid timeout
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                # Count results
                doc_count = sum(1 for _ in search.results())
                doc_count = max(doc_count, 1)  # Avoid division by zero
                idf = math.log(N / doc_count)
                IDF_APPROX[term] = idf
                display_json[term] = {
                    "idf": round(idf, 3),
                    "status": "computed: IDF from current run"
                }
                results.append({"Term": term, "Document Count": doc_count, "IDF": round(idf, 3)})
                logger.info(f"Term: {term}, Documents: {doc_count}, IDF: {idf:.3f}")
            except Exception as e:
                logger.error(f"Error for '{term}': {str(e)}")
                IDF_APPROX[term] = math.log(N / 10000)
                display_json[term] = {
                    "idf": round(math.log(N / 10000), 3),
                    "status": "warning: Using default IDF (computation failed)"
                }
                results.append({"Term": term, "Document Count": "Error", "IDF": round(math.log(N / 10000), 3)})

            # Update progress
            progress = (i + 1) / len(terms)
            progress_bar.progress(progress)
            status_text.text(f"Processed {i + 1}/{len(terms)} terms: {term}")

            # Update JSON display on the fly
            st.subheader("Current Computation Progress (JSON)")
            st.json(display_json)
            time.sleep(1)  # Respect arXiv rate limits (~1 request/second)

        # Save to JSON
        output_path = "idf_approx.json"
        try:
            with open(output_path, "w") as f:
                json.dump(IDF_APPROX, f, indent=4)
            logger.info(f"IDF_APPROX saved to {output_path}")
            st.success(f"IDF_APPROX saved to `{output_path}`")
        except Exception as e:
            logger.error(f"Error saving IDF_APPROX: {str(e)}")
            st.error(f"Error saving IDF_APPROX: {str(e)}")

        # Save to cache file for previous results
        cache_path = "previous_idf_approx.json"
        try:
            with open(cache_path, "w") as f:
                json.dump(IDF_APPROX, f, indent=4)
            logger.info(f"Previous IDF_APPROX saved to {cache_path}")
        except Exception as e:
            logger.error(f"Error saving previous IDF_APPROX: {str(e)}")
            st.error(f"Error saving previous IDF_APPROX: {str(e)}")

        # Clear cache to update with new results
        load_previous_json.clear()

        # Display final JSON file contents
        st.subheader("Final Computation Results (idf_approx.json)")
        try:
            with open(output_path, "r") as f:
                json_data = json.load(f)
            st.json(json_data)
        except Exception as e:
            st.error(f"Error reading idf_approx.json: {str(e)}")
            logger.error(f"Error reading idf_approx.json: {str(e)}")

        # Display results
        st.subheader("IDF Results")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        # Download button
        try:
            with open(output_path, "r") as f:
                st.download_button(
                    label="Download idf_approx.json",
                    data=f,
                    file_name="idf_approx.json",
                    mime="application/json"
                )
        except Exception as e:
            st.error(f"Error accessing idf_approx.json for download: {str(e)}")
            logger.error(f"Error accessing idf_approx.json for download: {str(e)}")

# Instructions for integration
st.subheader("Integration Instructions")
st.markdown("""
1. **Download `idf_approx.json`** after computation.
2. Place it in your main app's directory (e.g., `/home/kindness/workstation/.../corpus_data/`).
3. Update your main app's code to load `idf_approx.json` instead of the hardcoded `IDF_APPROX`:
```python
import json
import math
try:
    with open("idf_approx.json", "r") as f:
        IDF_APPROX = json.load(f)
    logger.info("Loaded arXiv-derived IDF_APPROX from idf_approx.json")
except FileNotFoundError:
    logger.warning("idf_approx.json not found, using default IDF_APPROX")
    IDF_APPROX = {
        "study": math.log(50000 / 40000), "analysis": math.log(50000 / 35000),
        "lithium-ion battery": math.log(50000 / 10000), "dendrite": math.log(50000 / 5000),
        # ... (add more defaults as needed)
    }
DEFAULT_IDF = math.log(50000 / 10000)  # Updated to match arXiv corpus
```
4. Run your main app (`streamlit run your_app.py`) and verify that TF-IDF scores reflect the new IDFs in visualizations.
""")

# Footer
st.markdown("---")
st.markdown("Understanding how common or rare is a word or phrase in lithium-ion battery and phase field modeling research.")
