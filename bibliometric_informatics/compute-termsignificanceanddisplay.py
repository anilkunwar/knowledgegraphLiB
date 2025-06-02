import streamlit as st
import arxiv
import math
import json
import time
import logging
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

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
If new keywords are added, warnings are shown with default IDFs, and recomputation is suggested.
Download the updated JSON and manually update the GitHub repository's `idf_approx.json`.
""")

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

# Define terms from KEYWORD_CATEGORIES (organized by category)
KEYWORD_CATEGORIES = {
    "Materials": [
        "lithium", "electrolyte", "anode", "cathode", "separator",
        "solid electrolyte", "liquid electrolyte", "polymer electrolyte",
        "lithium cobalt oxide", "lithium iron phosphate", "graphite anode",
        "silicon anode", "lithium metal", "solid-state electrolyte",
        "lithium titanate", "nickel manganese cobalt", "lithium sulfur",
        "garnet electrolyte", "sulfide electrolyte", "oxide electrolyte",
        "lithium-ion battery", "solid electrolyte interphase", "sei layer"
    ],
    "Methods": [
        "phase field method", "electrochemical modeling", "molecular dynamics",
        "density functional theory", "finite element analysis", "spectroscopy",
        "impedance spectroscopy", "cyclic voltammetry", "galvanostatic cycling",
        "operando imaging", "x-ray diffraction", "scanning electron microscopy",
        "transmission electron microscopy", "electrochemical impedance",
        "in-situ characterization", "computational modeling", "continuum modeling",
        "phase field simulation", "lattice boltzmann method", "monte carlo simulation"
    ],
    "Physical Phenomena": [
        "dendrite growth", "lithium plating", "electrolyte decomposition",
        "interface stability", "lithium diffusion", "ionic conductivity",
        "charge transfer", "electrode polarization", "sei formation",
        "thermal runaway", "electrochemical reaction", "ion transport",
        "concentration gradient", "overpotential", "capacity fading",
        "battery degradation", "side reaction", "electrode cracking",
        "lithium dendrite", "electrolyte oxidation"
    ],
    "Properties": [
        "capacity", "energy density", "power density", "cycle life",
        "ionic conductivity", "electronic conductivity", "thermal stability",
        "mechanical stability", "electrochemical stability", "specific capacity",
        "coulombic efficiency", "voltage hysteresis", "rate capability",
        "capacity retention", "mechanical stress", "interface resistance",
        "electrolyte conductivity", "diffusion coefficient", "charge-discharge rate",
        "sei thickness"
    ],
    "Other": [
        "battery safety", "fast charging", "high-energy battery",
        "electrode design", "electrolyte optimization", "battery management",
        "sustainable battery", "recyclable battery", "solid-state battery",
        "phase field modeling", "dendrite suppression", "electrode-electrolyte interface",
        "battery performance", "thermal management", "electrochemical kinetics"
    ]
}

# Flatten KEYWORD_CATEGORIES for deduplication
KEYWORD_CATEGORIES_TERMS = [term for sublist in KEYWORD_CATEGORIES.values() for term in sublist]

# Combine and deduplicate terms
terms = list(set(IDF_APPROX_TERMS + KEYWORD_CATEGORIES_TERMS))
logger.info(f"Total unique terms to query: {len(terms)}")
st.write(f"**Total unique terms to query**: {len(terms)}")

# Corpus size (estimated for lithium-ion battery and related research in arXiv)
N = 50000  # Approx. battery and electrochemistry-related papers
st.write(f"**Estimated corpus size (N)**: {N} documents (electrochemistry and materials science)")

# Initialize IDF dictionary
IDF_APPROX = {}

# Load idf_approx.json from repository
repo_json_path = "idf_approx.json"
if os.path.exists(repo_json_path):
    try:
        with open(repo_json_path, "r") as f:
            IDF_APPROX = json.load(f)
        logger.info("Loaded idf_approx.json from repository")
    except Exception as e:
        logger.error(f"Error loading idf_approx.json from repository: {str(e)}")
        st.error(f"Error loading idf_approx.json from repository: {str(e)}")
else:
    logger.warning("idf_approx.json not found in repository, using default IDFs")
    st.warning("idf_approx.json not found in repository. New keywords will use default IDFs until computed.")

# Cache function to load previous JSON (temporary persistence)
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

# Load previous JSON for caching
previous_json = load_previous_json()
previous_terms = set(previous_json.keys()) if previous_json else set()

# Identify new keywords (not in idf_approx.json)
repo_terms = set(IDF_APPROX.keys()) if IDF_APPROX else set()
new_terms = [term for term in terms if term not in repo_terms]

# Create initial JSON with warnings for new or uncomputed terms
display_json = {}
for term in terms:
    if term in new_terms:
        display_json[term] = {
            "idf": round(math.log(N / 10000), 3),
            "status": "warning: New term, using default IDF, recompute suggested"
        }
    elif term in IDF_APPROX:
        display_json[term] = {
            "idf": round(IDF_APPROX[term], 3),
            "status": "computed: Loaded from repository idf_approx.json"
        }
    else:
        display_json[term] = {
            "idf": round(math.log(N / 10000), 3),
            "status": "warning: Missing in idf_approx.json, using default IDF"
        }

# Display discrepancy warning if new terms exist
if new_terms:
    st.warning(
        f"New keywords detected not in idf_approx.json: {', '.join(new_terms)}. "
        "These use default IDF values. Recompute to update IDFs for these terms or all terms."
    )

# Function to determine data type
def get_data_type(term):
    data_types = []
    if term in IDF_APPROX_TERMS:
        data_types.append("Core Term")
    for category, terms_list in KEYWORD_CATEGORIES.items():
        if term in terms_list:
            data_types.append(category)
    return ", ".join(data_types) if data_types else "Unknown"

# Button to start computation
st.subheader("Compute IDF Values")
recompute_all = st.checkbox("Recompute all terms (otherwise only new terms are computed)")
if st.button("Compute IDFs from arXiv"):
    with st.spinner("Querying arXiv and computing IDFs..."):
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []

        # Determine terms to compute
        terms_to_compute = terms if recompute_all else new_terms
        logger.info(f"Computing IDFs for {len(terms_to_compute)} terms: {terms_to_compute}")

        # Create a placeholder for JSON display
        json_placeholder = st.empty()

        # Query arXiv
        client = arxiv.Client()
        for i, term in enumerate(terms_to_compute):
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
                results.append({
                    "Term": term,
                    "Document Count": doc_count,
                    "IDF": round(idf, 3),
                    "Data Type": get_data_type(term)
                })
                logger.info(f"Term: {term}, Documents: {doc_count}, IDF: {idf:.3f}")
            except Exception as e:
                logger.error(f"Error for '{term}': {str(e)}")
                IDF_APPROX[term] = math.log(N / 10000)
                display_json[term] = {
                    "idf": round(math.log(N / 10000), 3),
                    "status": "warning: Using default IDF (computation failed)"
                }
                results.append({
                    "Term": term,
                    "Document Count": "Error",
                    "IDF": round(math.log(N / 10000), 3),
                    "Data Type": get_data_type(term)
                })

            # Update progress
            progress = (i + 1) / len(terms_to_compute)
            progress_bar.progress(progress)
            status_text.text(f"Processed {i + 1}/{len(terms_to_compute)} terms: {term}")

            # Update JSON display in placeholder
            with json_placeholder.container():
                st.subheader("Current Computation Progress (JSON)")
                st.json(display_json)
            time.sleep(1)  # Respect arXiv rate limits (~1 request/second)

        # Add cached or repository results for terms not recomputed
        if not recompute_all:
            for term in terms:
                if term not in terms_to_compute:
                    if term in IDF_APPROX:
                        results.append({
                            "Term": term,
                            "Document Count": "From repository",
                            "IDF": round(IDF_APPROX[term], 3),
                            "Data Type": get_data_type(term)
                        })
                    elif term in previous_json:
                        results.append({
                            "Term": term,
                            "Document Count": "Cached",
                            "IDF": round(previous_json[term], 3),
                            "Data Type": get_data_type(term)
                        })

        # Save to JSON (temporary)
        output_path = "idf_approx.json"
        try:
            with open(output_path, "w") as f:
                json.dump(IDF_APPROX, f, indent=4)
            logger.info(f"IDF_APPROX saved to {output_path}")
            st.success(f"IDF_APPROX saved to `{output_path}`. Download and update the repository manually.")
        except Exception as e:
            logger.error(f"Error saving IDF_APPROX: {str(e)}")
            st.error(f"Error saving IDF_APPROX: {str(e)}")

        # Save to cache file for previous results (temporary)
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

        # Visualize IDF Scores
        st.subheader("Visualize IDF Scores")
        chart_type = st.selectbox("Select Chart Type", ["Bar", "Scatter", "Line"])
        data_type_filter = st.multiselect(
            "Filter by Data Type",
            options=list(set(df["Data Type"])),
            default=list(set(df["Data Type"]))
        )

        # Filter data
        filtered_df = df[df["Data Type"].isin(data_type_filter)]
        if filtered_df.empty:
            st.warning("No data available for selected filters.")
        else:
            # Sort by IDF for consistent ranking
            filtered_df = filtered_df.sort_values("IDF", ascending=False)

            if chart_type == "Bar":
                fig = px.bar(
                    filtered_df,
                    x="Term",
                    y="IDF",
                    color="Data Type",
                    hover_data=["Document Count", "Data Type"],
                    title="IDF Scores by Term (Ranked)"
                )
                fig.update_layout(xaxis_tickangle=45, xaxis_title="Term", yaxis_title="IDF Score")
            elif chart_type == "Scatter":
                fig = px.scatter(
                    filtered_df,
                    x="Document Count",
                    y="IDF",
                    color="Data Type",
                    hover_data=["Term", "Document Count", "Data Type"],
                    title="IDF vs Document Count",
                    size=[10] * len(filtered_df)
                )
                fig.update_layout(xaxis_title="Document Count", yaxis_title="IDF Score")
            else:  # Line
                fig = px.line(
                    filtered_df,
                    x="Term",
                    y="IDF",
                    color="Data Type",
                    hover_data=["Document Count", "Data Type"],
                    title="IDF Scores by Term (Ranked)"
                )
                fig.update_layout(xaxis_tickangle=45, xaxis_title="Term", yaxis_title="IDF Score")

            st.plotly_chart(fig, use_container_width=True)

        # Download button
        st.subheader("Download Updated IDF Results")
        if os.path.exists(output_path):
            try:
                with open(output_path, "r") as f:
                    st.download_button(
                        label="Download idf_approx.json",
                        data=f,
                        file_name="idf_approx.json",
                        mime="application/json"
                    )
                st.markdown(
                    "**Important**: After downloading, manually update the `idf_approx.json` in your GitHub repository "
                    "(e.g., via GitHub web interface or git push) to make the new IDFs available in the app."
                )
            except Exception as e:
                st.error(f"Error accessing idf_approx.json for download: {str(e)}")
                logger.error(f"Error accessing idf_approx.json for download: {str(e)}")
        else:
            st.warning("No idf_approx.json file available to download. Please compute IDFs first.")

# Display previous JSON (or initial JSON with warnings) after the button
st.subheader("Previous or Initial Computation Results")
st.json(display_json)
st.markdown("*Note: New or uncomputed keywords are marked with a warning and use a default IDF. These will be updated during computation.*")

# Instructions for integration
st.subheader("Integration Instructions")
st.markdown("""
1. **Compute and Download `idf_approx.json`**:
   - Run the computation in this app to generate updated IDF values.
   - Download the `idf_approx.json` file using the download button.
2. **Update GitHub Repository**:
   - Replace the existing `idf_approx.json` in your GitHub repository with the downloaded file.
   - Use the GitHub web interface: Navigate to your repository, upload the new `idf_approx.json`, and commit.
   - Alternatively, use git commands:
     ```bash
     git pull origin main
     cp path/to/downloaded/idf_approx.json idf_approx.json
     git add idf_approx.json
     git commit -m "Update idf_approx.json with new IDF values"
     git push origin main
     ```
3. **Restart Streamlit App**:
   - After updating the repository, redeploy or restart the Streamlit app to load the new `idf_approx.json`.
   - In Streamlit Cloud, this happens automatically on new commits, or trigger a redeploy manually.
4. **Update Main App**:
   - Ensure your main app loads `idf_approx.json` from the repository:
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
5. **Run Main App**:
   - Run your main app (`streamlit run your_app.py`) and verify that TF-IDF scores reflect the updated IDFs in visualizations.
""")

# Footer
st.markdown("---")
st.markdown("Understanding how common or rare a word or phrase is in lithium-ion battery and phase field modeling research.")
