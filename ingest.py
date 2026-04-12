import os
import json
import hashlib
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import glob

from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.chunking.basic import chunk_elements

load_dotenv()

# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("ingestion.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ==========================================
# GLOBAL CLIENTS
# ==========================================
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
local_embedding_model = SentenceTransformer(model_name)

# ==========================================
# PROGRESS TRACKER
# ==========================================
PROGRESS_LOG = "processed_files.log"

def load_processed_files() -> set:
    """Load the set of already-processed file paths from disk."""
    if not Path(PROGRESS_LOG).exists():
        return set()
    with open(PROGRESS_LOG, "r") as f:
        return set(line.strip() for line in f if line.strip())

def mark_file_processed(file_path: str):
    """Append a file path to the progress log so it's skipped on re-runs."""
    with open(PROGRESS_LOG, "a") as f:
        f.write(file_path + "\n")

# ==========================================
# HELPERS
# ==========================================
def compute_hash(text: str) -> str:
    """SHA-256 hash of a chunk's text — used for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_embedding(text: str) -> List[float]:
    """
    Generate a 768-dim embedding.
    Raises on failure — never returns a zero vector silently.
    """
    embedding = local_embedding_model.encode(text).tolist()
    if all(v == 0.0 for v in embedding):
        raise ValueError("Embedding model returned an all-zero vector.")
    return embedding

def supabase_upsert_with_retry(
    table: str,
    data: Dict[str, Any],
    conflict_column: str,
    max_retries: int = 3,
    backoff: float = 2.0
):
    """
    Upsert a row into Supabase with exponential backoff retry.
    Uses conflict_column for deduplication (requires a UNIQUE constraint on that column in Supabase).
    """
    for attempt in range(1, max_retries + 1):
        try:
            supabase.table(table).upsert(data, on_conflict=conflict_column).execute()
            return
        except Exception as e:
            log.warning(f"Supabase upsert attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(backoff ** attempt)
            else:
                log.error(f"All {max_retries} upsert attempts failed for table '{table}'. Data: {data.get('content_hash', data.get('drug_name', '?'))}")
                raise

def is_likely_scanned(file_path: str) -> bool:
    """
    Heuristic: attempt fast parse and check if text yield is very low.
    If fewer than 100 characters extracted, assume it's a scanned PDF.
    """
    try:
        elements = partition_pdf(filename=file_path, strategy="fast")
        total_text = " ".join(e.text for e in elements if hasattr(e, "text"))
        return len(total_text.strip()) < 100
    except Exception:
        return True  # Assume scanned if fast parse errors out

def parse_pdf_metadata_from_manifest(
    filename: str,
    manifest: Optional[Dict[str, Any]]
) -> tuple[str, str, str]:
    """
    Look up metadata from a manifest dict first.
    Falls back to filename parsing as a last resort.
    
    Manifest format example (manifest.json):
    {
        "WHO_Hypertension_2023.pdf": {
            "source_org": "WHO",
            "disease_topic": "Hypertension",
            "url_reference": "https://who.int/..."
        }
    }
    """
    if manifest and filename in manifest:
        entry = manifest[filename]
        return (
            entry.get("source_org", "Unknown_Org"),
            entry.get("disease_topic", "General_Topic"),
            entry.get("url_reference", f"local://{filename}")
        )

    # Fallback: parse from filename
    name_parts = Path(filename).stem.split("_")
    source_org = name_parts[0] if len(name_parts) > 0 else "Unknown_Org"
    disease_topic = name_parts[1] if len(name_parts) > 1 else "General_Topic"
    log.warning(
        f"No manifest entry for '{filename}'. "
        f"Falling back to filename parsing → source_org='{source_org}', disease_topic='{disease_topic}'. "
        f"Consider adding this file to manifest.json for accurate metadata."
    )
    return source_org, disease_topic, f"local://{filename}"


# ==========================================
# PIPELINE 1: Clinical Guidelines (PDFs)
# ==========================================
def process_guideline_pdf(
    file_path: str,
    source_org: str,
    disease_topic: str,
    url_reference: str
):
    log.info(f"Processing Guideline PDF: {file_path}")

    # 1. Choose strategy based on whether the PDF is scanned
    scanned = is_likely_scanned(file_path)
    strategy = "hi_res" if scanned else "fast"
    log.info(f"Using PDF strategy: '{strategy}' ({'scanned' if scanned else 'digital'} PDF detected)")

    elements = partition_pdf(
        filename=file_path,
        strategy=strategy,
        infer_table_structure=True
    )

    # 2. Chunk by title with fallback to basic chunking
    chunks = chunk_by_title(
        elements,
        max_characters=2000,
        combine_text_under_n_chars=500
    )

    if len(chunks) == 1 and len(chunks[0].text) > 2000:
        log.warning(
            f"chunk_by_title produced one oversized chunk ({len(chunks[0].text)} chars). "
            f"Falling back to basic fixed-size chunking."
        )
        chunks = chunk_elements(elements, max_characters=2000, overlap=200)

    log.info(f"Extracted {len(chunks)} chunks. Uploading to Supabase...")

    inserted, skipped, failed = 0, 0, 0

    for i, chunk in enumerate(chunks):
        chunk_text = chunk.text.strip()
        if len(chunk_text) < 10:
            skipped += 1
            continue

        content_hash = compute_hash(chunk_text)

        try:
            embedding = get_embedding(chunk_text)
        except Exception as e:
            log.error(f"Embedding failed for chunk {i+1} in '{file_path}': {e}. Skipping chunk.")
            failed += 1
            continue

        data = {
            "content_hash": content_hash,
            "source_org": source_org,
            "disease_topic": disease_topic,
            "url_reference": url_reference,
            "chunk_content": chunk_text,
            "embedding": embedding
        }

        try:
            supabase_upsert_with_retry("clinical_guidelines", data, conflict_column="content_hash")
            inserted += 1
        except Exception:
            failed += 1

    log.info(
        f"Done: {file_path} → inserted/updated={inserted}, skipped_short={skipped}, failed={failed}"
    )


# ==========================================
# PIPELINE 2: DailyMed FDA Labels (JSON)
# ==========================================
def process_drug_label(drug_data: Dict[str, str]):
    drug_name = drug_data.get("drug_name", "Unknown Drug")
    log.info(f"Processing Drug Label: {drug_name}")

    chunk_content = (
        f"DRUG NAME: {drug_name}\n"
        f"INDICATION: {drug_data.get('indication', 'Not specified.')}\n"
        f"DOSAGE AND ADMINISTRATION: {drug_data.get('dosage_and_administration', 'Not specified.')}\n"
        f"WARNINGS AND PRECAUTIONS: {drug_data.get('warnings_and_precautions', 'Not specified.')}\n"
        f"RENAL ADJUSTMENT: {drug_data.get('renal_adjustment', 'No specific renal adjustment data provided.')}"
    )

    content_hash = compute_hash(chunk_content)

    try:
        embedding = get_embedding(chunk_content)
    except Exception as e:
        log.error(f"Embedding failed for drug '{drug_name}': {e}. Skipping.")
        return

    data = {
        "content_hash": content_hash,
        "drug_name": drug_name,
        "indication": drug_data.get("indication", ""),
        "dosage_and_administration": drug_data.get("dosage_and_administration", ""),
        "warnings_and_precautions": drug_data.get("warnings_and_precautions", ""),
        "renal_adjustment": drug_data.get("renal_adjustment", ""),
        "chunk_content": chunk_content,
        "embedding": embedding
    }

    try:
        supabase_upsert_with_retry("drug_labels", data, conflict_column="content_hash")
        log.info(f"Upserted: {drug_name}")
    except Exception as e:
        log.error(f"Failed to insert '{drug_name}' after all retries: {e}")


# ==========================================
# BATCH PROCESSING
# ==========================================
def load_manifest(folder_path: str) -> Optional[Dict[str, Any]]:
    """Load manifest.json from the guidelines folder if it exists."""
    manifest_path = Path(folder_path) / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            log.info(f"Loaded manifest from {manifest_path}")
            return json.load(f)
    log.warning(f"No manifest.json found in {folder_path}. Metadata will be parsed from filenames.")
    return None

def batch_process_guidelines(folder_path: str = "data/guidelines"):
    pdf_files = glob.glob(f"{folder_path}/*.pdf")
    if not pdf_files:
        log.warning(f"No PDFs found in {folder_path}")
        return

    manifest = load_manifest(folder_path)
    processed = load_processed_files()

    log.info(f"Found {len(pdf_files)} guideline PDFs. {len(processed)} already processed.")

    for pdf_path in pdf_files:
        abs_path = str(Path(pdf_path).resolve())

        if abs_path in processed:
            log.info(f"Skipping already-processed file: {pdf_path}")
            continue

        filename = os.path.basename(pdf_path)
        source_org, disease_topic, url_reference = parse_pdf_metadata_from_manifest(filename, manifest)

        try:
            process_guideline_pdf(pdf_path, source_org, disease_topic, url_reference)
            mark_file_processed(abs_path)
        except Exception as e:
            log.error(f"Pipeline failed for {pdf_path}: {e}. File will be retried on next run.")

def batch_process_drugs(folder_path: str = "data/dailymed"):
    json_files = glob.glob(f"{folder_path}/*.json")
    if not json_files:
        log.warning(f"No JSON files found in {folder_path}")
        return

    processed = load_processed_files()
    log.info(f"Found {len(json_files)} DailyMed JSON files. {len(processed)} already processed.")

    for json_path in json_files:
        abs_path = str(Path(json_path).resolve())

        if abs_path in processed:
            log.info(f"Skipping already-processed file: {json_path}")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                drug_data = json.load(f)

            if isinstance(drug_data, list):
                for drug in drug_data:
                    process_drug_label(drug)
            elif isinstance(drug_data, dict):
                process_drug_label(drug_data)
            else:
                log.error(f"Unexpected JSON structure in {json_path}: {type(drug_data)}")
                continue

            mark_file_processed(abs_path)
        except json.JSONDecodeError as e:
            log.error(f"Invalid JSON in {json_path}: {e}")
        except Exception as e:
            log.error(f"Pipeline failed for {json_path}: {e}. File will be retried on next run.")


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    log.info("Starting Medical Data Bulk Ingestion Pipeline...")

    log.info("--- Phase 1: Guidelines ---")
    batch_process_guidelines()

    log.info("--- Phase 2: Pharmacopoeia ---")
    batch_process_drugs()

    log.info("Bulk Ingestion Complete!")
