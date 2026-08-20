import sys
import csv
import subprocess
import os
import json
import uuid
import threading
import time
from azure.kusto.data.exceptions import KustoNetworkError
from datetime import datetime
import traceback
import argparse
import logging
from typing import List, Dict, Tuple
from timing_utils import start_timing, end_timing, time_operation, time_context, print_timing_summary, save_timing_report, reset_timing_data

# Configure centralized logging
def setup_logging():
    """Setup centralized logging for the entire application"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create logger for this module
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler - detailed logging
    file_handler = logging.FileHandler('logs/summarizer.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler - only warnings and errors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

# Setup logging at module level
logger = setup_logging()

def _detect_redacted_in_csv(csv_path: str) -> bool:
    """Quick scan for exact redaction token in the 'Authored summary' section of the CSV."""
    try:
        if not os.path.exists(csv_path):
            return False
        in_authored = False
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                first = row[0].strip() if row[0] else ''
                # Section headers are like: --- Authored summary ---
                if first.startswith('---') and first.endswith('---'):
                    section_name = first.strip('- ').strip()
                    in_authored = (section_name.lower() == 'authored summary')
                    continue
                if in_authored:
                    # Look for an exact token: ** REDACTED ** (case-insensitive, flexible inner spacing)
                    import re
                    for c in row:
                        if not isinstance(c, str):
                            continue
                        if re.match(r"^\s*\*\*\s*REDACTED\s*\*\*\s*$", c.strip(), flags=re.IGNORECASE):
                            return True
        return False
    except Exception as e:
        logger.warning(f"Failed CSV redaction scan: {e}")
        return False

def show_prompt_menu():
    """Display available prompts and get user selection"""
    try:
        with open("prompts.json", "r", encoding="utf-8") as f:
            prompts = json.load(f)
        
        # Show all available prompt types
        prompt_types = list(prompts.keys())

        if not prompt_types:
            print("No prompt types found in prompts.json")
            sys.exit(1)

        # Load curated list of interactive menu prompts with emojis
        try:
            with open("interactive_menu_prompts.json", "r", encoding="utf-8") as f:
                prompt_emojis = json.load(f)
        except FileNotFoundError:
            prompt_emojis = {}
        except json.JSONDecodeError:
            prompt_emojis = {}

        # Filter to only show prompts that have emojis and exist in prompts.json
        prompt_types_with_emojis = [pt for pt in prompt_types if pt in prompt_emojis]
        
        if not prompt_types_with_emojis:
            print("No prompt types with emojis found in prompts.json")
            sys.exit(1)
        
        # Append free-text as last option (not in prompts.json; prompts generated from user input)
        menu_options = prompt_types_with_emojis + ["free_text"]
        free_text_label = "free text (describe what you need)"

        print("\nAvailable prompt types:")
        print("=" * 40)
        for i, prompt_type in enumerate(menu_options, 1):
            emoji = prompt_emojis.get(prompt_type, "")
            label = free_text_label if prompt_type == "free_text" else prompt_type
            print(f"{i:2d}. {emoji} {label}")
        print("=" * 40)
        
        while True:
            try:
                choice = input("Select a prompt type (enter number): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(menu_options):
                    selected_prompt = menu_options[choice_num - 1]
                    print(f"Selected: {selected_prompt}")
                    
                    # Automatically set vector database path for article search mode
                    if selected_prompt == 'article_search':
                        from config import config
                        default_vector_db_path = config.default_vector_db_path
                        if default_vector_db_path:
                            print(f"🔍 Article search mode detected - automatically using vector database: {default_vector_db_path}")
                            return selected_prompt, default_vector_db_path
                        else:
                            print("⚠️ Article search mode detected but DEFAULT_VECTOR_DB_PATH not set in .env file")
                            return selected_prompt, None
                    
                    return selected_prompt, None
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(menu_options)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
                
    except Exception as e:
        print(f"Error reading prompts.json: {e}")
        sys.exit(1)


def read_free_text_input():
    """Read multiline prompt from stdin. User submits by pressing Enter on an empty line. Returns non-empty string."""
    print("Enter your prompt (multiple lines OK). Press Enter on an empty line to submit.")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    text = "\n".join(lines).strip()
    if not text:
        text = "Summarize this incident clearly."
        print(f"Using default: {text}")
    return text


def get_free_text_prompts(user_description=None):
    """Generate full system_prompt + user_prompt from user description (or from multiline stdin if user_description is None). Returns dict with system_prompt, user_prompt."""
    from free_text_prompt_generator import generate_prompts_from_free_text
    if user_description is None:
        user_description = read_free_text_input()
    print("Generating custom prompt from your description...")
    prompts = generate_prompts_from_free_text(user_description)
    print("\n" + "=" * 60)
    print("GENERATED PROMPT (will be used for incident analysis)")
    print("=" * 60)
    print("\n--- system_prompt ---")
    print(prompts.get("system_prompt", ""))
    print("\n--- user_prompt ---")
    print(prompts.get("user_prompt", ""))
    print("=" * 60 + "\n")
    return prompts


def _fetch_and_transform_incidents(incident_numbers: List[str]) -> List[str]:
    """Fetch incident data from the configured database and process CSV to JSON. Returns successful incident IDs."""
    successful = []
    for incident_number in incident_numbers:
        if fetch_incident_data(incident_number):
            successful.append(incident_number)
        else:
            logger.warning(f"Skipping incident {incident_number} due to fetch failure")
            print(f"Skipping incident {incident_number} due to fetch failure")
    for incident_number in list(successful):
        try:
            process_incident_to_json(incident_number)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error processing incident {incident_number} to JSON: {e}")
            print(f"Error processing incident {incident_number} to JSON: {e}")
            successful.remove(incident_number)
    return successful


@time_operation("fetch_incident_data", "fetch")
def fetch_incident_data(incident_number):
    """Fetch data for a single incident from database"""
    logger.info(f"Starting database data fetch for incident {incident_number}")
    print(f"Fetching data for incident {incident_number} from database...")
    
    with time_context("kusto_subprocess", "fetch", {"incident_number": incident_number}):
        fetch_proc = subprocess.run([
            sys.executable, "kusto_fetcher.py", str(incident_number), "--output-dir", "incidents"
        ], capture_output=True, text=True)
    
    if fetch_proc.returncode != 0:
        # Log all output to a debug file
        debug_log_path = f"logs/fetcher_debug_{incident_number}.log"
        with open(debug_log_path, "w") as log_file:
            log_file.write("STDOUT:\n" + fetch_proc.stdout + "\n\n")
            log_file.write("STDERR:\n" + fetch_proc.stderr + "\n")
        
        logger.error(f"Database fetch failed for incident {incident_number}. Return code: {fetch_proc.returncode}")
        logger.error(f"STDOUT: {fetch_proc.stdout}")
        logger.error(f"STDERR: {fetch_proc.stderr}")
        
        # Check for VPN/database network error in stderr or stdout
        network_error_indicators = [
            "Could not connect to database",
            "KustoNetworkError",
            "Failed to process network request",
            "Network error"
        ]
        
        is_network_error = any(indicator in fetch_proc.stdout or indicator in fetch_proc.stderr 
                              for indicator in network_error_indicators)
        
        if is_network_error:
            logger.error(f"Network error detected for incident {incident_number}. VPN connection may be required.")
            print(f"❌ Network Error: Could not connect to database for incident {incident_number}")
            print(f"🔧 Solution: Please ensure your VPN connection is active and try again.")
            print(f"📋 Full error details are available in {debug_log_path}")
            return False
        
        print(f"Database fetch step failed for incident {incident_number}. See {debug_log_path} for details.")
        return False

    # Check if the CSV file is empty or only contains the placeholder line
    # Try the new incident-specific folder structure first, then fall back to flat structure
    csv_path = os.path.join("incidents", str(incident_number), f"{incident_number}.csv")
    if not os.path.exists(csv_path):
        # Fall back to flat structure for backward compatibility
        csv_path = os.path.join("incidents", f"{incident_number}.csv")
    logger.info(f"Checking CSV file: {csv_path}")
    
    is_empty = False
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            if len(lines) == 0 or (len(lines) == 1 and lines[0] == "--- Discussions ---"):
                is_empty = True
                logger.warning(f"CSV file is empty or contains only placeholder for incident {incident_number}")
    except Exception as e:
        is_empty = True
        logger.error(f"Error reading CSV file {csv_path}: {e}")
        print(f"[main] Error reading CSV file for incident {incident_number}: {e}")
        return False

    if is_empty:
        logger.warning(f"No data fetched for incident {incident_number}. CSV file is empty.")
        
        # Check if this might be due to a network error by looking at recent logs
        try:
            with open("logs/fetcher.log", "r", encoding="utf-8") as log_file:
                recent_logs = log_file.read()
                if any(indicator in recent_logs for indicator in ["KustoNetworkError", "Failed to process network request", "Network error"]):
                    print(f"❌ Network Error: No data fetched for incident {incident_number} due to network connectivity issues.")
                    print(f"🔧 Solution: Please ensure your VPN connection is active and try again.")
                    print(f"📋 Check logs/fetcher.log for detailed error information.")
                    return False
        except FileNotFoundError:
            pass  # Log file doesn't exist, continue with normal empty file handling
        
        print(f"[main] No data was fetched for incident {incident_number}. The CSV file is empty.")
        
        # No manual fallback available - return failure
        print(f"[main] No data available for incident {incident_number}. Please check the incident ID or try again later.")
        return False
    
    logger.info(f"Successfully fetched data for incident {incident_number}. CSV file created at {csv_path}")
    print(f"✅ Created: {csv_path}")
    # Immediately detect redacted authored summary and stop processing if manual.docx is not available
    try:
        if _detect_redacted_in_csv(csv_path):
            print(f"⚠️  Authored summary appears REDACTED for incident {incident_number}.")
            # Check if manual.docx exists to handle redacted content
            try:
                from config import config
                manual_docx_path = os.path.join(str(config.root_dir), "manual.docx")
                if os.path.exists(manual_docx_path):
                    print(f"✅ Found manual.docx at {manual_docx_path}. Continuing with processing.")
                    print(f"   Transformer will use manual.docx to replace redacted summary.")
                else:
                    print(f"❌ REDACTED summary detected but manual.docx not found at {manual_docx_path}.")
                    print(f"   Stopping processing. Please provide manual.docx and retry.")
                    logger.warning(f"Stopping processing for incident {incident_number} due to redacted summary without manual.docx")
                    return False
            except Exception as config_error:
                logger.error(f"Failed to check for manual.docx: {config_error}")
                print(f"❌ Failed to check for manual.docx. Stopping processing to avoid using redacted content.")
                return False
    except Exception as e:
        logger.warning(f"Failed CSV redaction scan: {e}")
        # If detection fails, continue but log the issue
        print(f"⚠️  Warning: Could not scan for redacted content. Continuing with caution.")
    return True

@time_operation("process_incident_to_json", "process")
def process_incident_to_json(incident_number):
    """Process CSV to JSON for a single incident"""
    logger.info(f"Starting CSV to JSON conversion for incident {incident_number}")
    print(f"Processing CSV to JSON for incident {incident_number}...")
    
    # Try the new incident-specific folder structure first, then fall back to flat structure
    csv_path = os.path.join("incidents", str(incident_number), f"{incident_number}.csv")
    if not os.path.exists(csv_path):
        # Fall back to flat structure for backward compatibility
        csv_path = os.path.join("incidents", f"{incident_number}.csv")
    logger.info(f"Processing CSV file: {csv_path}")
    
    try:
        with time_context("transformer_subprocess", "process", {"incident_number": incident_number, "csv_path": csv_path}):
            result = subprocess.run([
                sys.executable, "transformer.py", csv_path
            ], capture_output=True, text=True, check=True)
        
        logger.info(f"Successfully processed CSV to JSON for incident {incident_number}")
        logger.info(f"STDOUT: {result.stdout}")
        # Surface transformer logs (including LLM request/response previews) to console before AI step
        if result.stdout:
            print(result.stdout)
        print(f"✅ Created: processed_incidents/{incident_number}.json")
        
        # Gate: if the produced JSON has redacted summary, do not proceed to AI
        json_path = os.path.join("processed_incidents", f"{incident_number}.json")
        try:
            with open(json_path, 'r', encoding='utf-8') as jf:
                produced = json.load(jf)
            produced_summary = (produced.get("summary", "") or "").strip()
            import re
            if isinstance(produced_summary, str) and re.match(r"^\s*\*\*\s*REDACTED\s*\*\*\s*$", produced_summary, flags=re.IGNORECASE):
                print(f"❌ Summary is '** REDACTED **' in {json_path}. Aborting further AI processing.")
                # Raise to prevent downstream processing
                raise RuntimeError("Redacted summary detected - stopping pipeline before AI step")
        except FileNotFoundError:
            # If file is missing, let the caller handle
            pass
        
    except subprocess.CalledProcessError as e:
        logger.error(f"CSV to JSON conversion failed for incident {incident_number}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise

def _parse_manual_txt(txt_path: str) -> Tuple[str, List[Dict]]:
    """Parse a manual text file (format from write_manual_txt.py) into summary and discussion_items.
    Returns (summary_text, discussion_items) where discussion_items have 'Text', 'author', 'Date' keys."""
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    if not content.strip():
        return "", []

    summary_parts = []
    discussion_section_lines = []  # raw lines of Discussion section
    state = None
    lines = content.splitlines()
    sep_equals = "=" * 60
    sep_dashes = "-" * 40

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped == sep_equals or (stripped.startswith("=") and len(stripped) >= 40 and set(stripped.replace("=", "")) == {""}):
            i += 1
            section_name = ""
            while i < len(lines):
                section_name = lines[i].strip()
                i += 1
                if section_name:
                    break
            if i < len(lines) and lines[i].strip().startswith("="):
                i += 1
            if "authored summary" in section_name.lower():
                state = "summary"
            elif "discussion" in section_name.lower():
                state = "discussion"
            elif "incident images" in section_name.lower():
                state = "images"
            else:
                state = None
            continue

        if state == "summary":
            if stripped == sep_equals or (stripped.startswith("=") and len(stripped) >= 40):
                state = None
                i += 1
                continue
            if stripped == "(No summary content)":
                i += 1
                continue
            summary_parts.append(line)
        elif state == "discussion":
            if stripped == sep_equals or (stripped.startswith("=") and len(stripped) >= 40):
                state = None
                i += 1
                continue
            if stripped == "(No discussion entries after filter)":
                i += 1
                continue
            if stripped == sep_dashes:
                discussion_section_lines.append(stripped)
                i += 1
                continue
            discussion_section_lines.append(line)
        elif state == "images":
            if stripped.startswith("=") and len(stripped) >= 40:
                state = None
            i += 1
            continue
        i += 1

    # Parse discussion section: blocks split by line of 40 dashes; each block first line = "Author - Date", rest = body
    discussion_items = []
    discussion_text = "\n".join(discussion_section_lines)
    for block in discussion_text.split("\n" + sep_dashes + "\n"):
        block = block.strip()
        if not block or block == "(No discussion entries after filter)":
            continue
        first_newline = block.find("\n")
        if first_newline >= 0:
            author_date = block[:first_newline].strip()
            body = block[first_newline + 1 :].strip()
        else:
            author_date = block
            body = ""
        author, date = author_date, ""
        if " - " in author_date:
            parts = author_date.split(" - ", 1)
            author = parts[0].strip()
            date = parts[1].strip() if len(parts) > 1 else ""
        discussion_items.append({"author": author, "Date": date, "Text": body})

    summary_text = "\n".join(summary_parts).strip()
    return summary_text, discussion_items


def process_manual_txt_only(incident_number, manual_txt_path=None):
    """Create a processed incident JSON using a manual text file instead of fetching database data."""
    logger.info(f"Starting manual text processing for incident {incident_number}")
    start_time = time.monotonic()
    print(f"Processing manual text for incident {incident_number}...")

    try:
        from config import config
        from transformer import dump_discussion_items_to_json
    except Exception as import_error:
        logger.error(f"Failed to import helpers: {import_error}")
        raise

    txt_path = manual_txt_path or os.path.join(str(config.root_dir), "manual.txt")
    manual_name = os.path.basename(txt_path)
    if not os.path.exists(txt_path):
        error_msg = f"{manual_name} not found at {txt_path}. Provide the file and retry."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    with open(txt_path, "r", encoding="utf-8") as _f:
        raw_manual = _f.read()
    if not raw_manual.strip():
        error_msg = f"{manual_name} is empty. Add incident content and retry."
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        summary_text, discussion_items = _parse_manual_txt(txt_path)
    except Exception as parse_error:
        logger.error(f"Failed to parse {manual_name}: {parse_error}")
        raise

    if not summary_text and not discussion_items:
        # File has content but not the expected section layout, so process the full body as summary.
        warn_msg = (
            f"{manual_name} has no Authored summary / Discussion sections; using entire file as summary. "
            "For structured discussion entries, use the manual exporter section layout."
        )
        logger.warning(warn_msg)
        print(f"⚠️  WARNING: {warn_msg}")
        summary_text = raw_manual.strip()
        discussion_items = []

    logger.info(f"Parsed {manual_name}: summary {len(summary_text)} chars, {len(discussion_items)} discussion entries")
    print(f"📄 Read summary ({len(summary_text)} chars) and {len(discussion_items)} discussion entries from {manual_name}")

    try:
        output_file, _ = dump_discussion_items_to_json(
            discussion_items,
            str(incident_number),
            summary_content=summary_text or None,
            summary_images=None,
        )
    except Exception as dump_error:
        logger.error(f"Failed to write processed incident JSON: {dump_error}")
        raise

    elapsed = time.monotonic() - start_time
    logger.info(f"Manual text processing completed in {elapsed:.2f}s. Output: {output_file}")
    print(f"✅ Created: {output_file} ({elapsed:.2f}s)")
    return output_file


def process_manual_docx_only(incident_number):
    """Create a processed incident JSON using manual.docx content only."""
    logger.info(f"Starting manual.docx processing for incident {incident_number}")
    print(f"Processing manual.docx for incident {incident_number}...")

    try:
        from transformer import (
            prompt_user_for_docx,
            extract_images_from_docx,
            extract_summary_from_docx_text,
            dump_discussion_items_to_json,
        )
    except Exception as import_error:
        logger.error(f"Failed to import manual.docx helpers: {import_error}")
        raise

    docx_path = prompt_user_for_docx(incident_number)
    if not docx_path:
        error_msg = "manual.docx not found at project root. Provide the document and retry."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        # Use new extraction function that returns both text and images
        docx_text, docx_images = extract_images_from_docx(docx_path)
        print(f"📸 Extracted {len(docx_images)} screenshot(s) from manual.docx")
    except Exception as extract_error:
        logger.error(f"Failed to read manual.docx: {extract_error}")
        raise

    if not docx_text or not docx_text.strip():
        error_msg = "manual.docx appears empty. Populate it with incident details and retry."
        logger.error(error_msg)
        raise ValueError(error_msg)

    extracted_summary = None
    try:
        extracted_summary = extract_summary_from_docx_text(docx_text)
    except Exception as summary_error:
        logger.warning(f"LLM extraction failed; using raw manual.docx content. Error: {summary_error}")
        extracted_summary = None

    final_summary = extracted_summary.strip() if extracted_summary and extracted_summary.strip() else docx_text.strip()

    try:
        output_file, _ = dump_discussion_items_to_json(
            [],
            str(incident_number),
            summary_content=final_summary,
            summary_images=docx_images if docx_images else None,
        )
    except Exception as dump_error:
        logger.error(f"Failed to write processed incident JSON: {dump_error}")
        raise

    logger.info(f"Manual processing completed. Output: {output_file}")
    print(f"✅ Created: {output_file}")
    return output_file

def process_markdown_only(incident_number: str, markdown_path: str):
    """Create a processed incident JSON using a markdown file content only."""
    logger.info(f"Starting markdown file processing for incident {incident_number}")
    print(f"Processing markdown file for incident {incident_number}...")

    # Validate markdown file exists
    if not os.path.exists(markdown_path):
        error_msg = f"Markdown file not found at {markdown_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        from transformer import dump_discussion_items_to_json
    except Exception as import_error:
        logger.error(f"Failed to import markdown helpers: {import_error}")
        raise

    try:
        # Read the markdown file content
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()

        if not markdown_text or not markdown_text.strip():
            error_msg = f"Markdown file appears empty: {markdown_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Read {len(markdown_text)} characters from {markdown_path}")
        print(f"📄 Read {len(markdown_text)} characters from {markdown_path}")

        # Use the markdown content as the summary
        final_summary = markdown_text.strip()

        # Create the processed incident JSON
        output_file, _ = dump_discussion_items_to_json(
            [],
            str(incident_number),
            summary_content=final_summary,
            summary_images=None,  # Markdown files don't contain embedded images
        )
    except Exception as dump_error:
        logger.error(f"Failed to write processed incident JSON: {dump_error}")
        raise

    logger.info(f"Markdown processing completed. Output: {output_file}")
    print(f"✅ Created: {output_file}")
    return output_file

@time_operation("combine_incident_data", "process")
def combine_incident_data(incident_numbers):
    """Combine data from multiple incidents into a single JSON file"""
    logger.info(f"Starting to combine data from {len(incident_numbers)} incidents")
    print("Combining data from multiple incidents...")
    
    combined_data = {
        "incidents": [],
        "total_incidents": len(incident_numbers),
        "combined_timestamp": datetime.now().isoformat()
    }
    
    for incident_number in incident_numbers:
        json_path = os.path.join("processed_incidents", f"{incident_number}.json")
        logger.info(f"Processing incident {incident_number} from {json_path}")
        
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    incident_data = json.load(f)
                    incident_data["incident_number"] = incident_number
                    combined_data["incidents"].append(incident_data)
                logger.info(f"Successfully loaded incident {incident_number}")
            except Exception as e:
                logger.error(f"Error reading JSON for incident {incident_number}: {e}")
                print(f"Error reading JSON for incident {incident_number}: {e}")
                continue
        else:
            logger.warning(f"JSON file not found for incident {incident_number}: {json_path}")
            print(f"Warning: JSON file not found for incident {incident_number}")
    
    # Save combined data
    combined_path = os.path.join("processed_incidents", "combined_incidents.json")
    logger.info(f"Saving combined data to {combined_path}")
    
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Successfully combined {len(combined_data['incidents'])} incidents into {combined_path}")
    print(f"✅ Created: {combined_path}")
    return combined_path

@time_operation("create_troubleshooting_plan_data", "process")
def create_troubleshooting_plan_data(incident_numbers):
    """Create a special combined data structure for troubleshooting plan mode.
    The first incident is treated as the primary incident, others as historical references."""
    print("Creating troubleshooting plan data structure...")
    
    if len(incident_numbers) < 2:
        raise ValueError("Troubleshooting plan mode requires at least 2 incidents")
    
    primary_incident = incident_numbers[0]
    historical_incidents = incident_numbers[1:]
    
    combined_data = {
        "mode": "troubleshooting_plan",
        "primary_incident": primary_incident,
        "historical_incidents": historical_incidents,
        "incidents": [],
        "total_incidents": len(incident_numbers),
        "combined_timestamp": datetime.now().isoformat()
    }
    
    # Load primary incident first
    primary_json_path = os.path.join("processed_incidents", f"{primary_incident}.json")
    if os.path.exists(primary_json_path):
        try:
            with open(primary_json_path, "r", encoding="utf-8") as f:
                primary_data = json.load(f)
                primary_data["incident_number"] = primary_incident
                primary_data["role"] = "primary"
                combined_data["incidents"].append(primary_data)
        except Exception as e:
            print(f"Error reading JSON for primary incident {primary_incident}: {e}")
            raise
    else:
        print(f"Error: Primary incident JSON file not found for {primary_incident}")
        raise FileNotFoundError(f"Primary incident file not found: {primary_json_path}")
    
    # Load historical incidents
    for incident_number in historical_incidents:
        json_path = os.path.join("processed_incidents", f"{incident_number}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    incident_data = json.load(f)
                    incident_data["incident_number"] = incident_number
                    incident_data["role"] = "historical"
                    combined_data["incidents"].append(incident_data)
            except Exception as e:
                print(f"Error reading JSON for historical incident {incident_number}: {e}")
                continue
        else:
            print(f"Warning: JSON file not found for historical incident {incident_number}")
    
    # Save combined data
    combined_path = os.path.join("processed_incidents", f"troubleshooting_plan_{primary_incident}_with_{'_'.join(historical_incidents)}.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created: {combined_path}")
    return combined_path

def _process_combined_incidents(processor, combined_data, prompts, prompt_type, debug_api, incident_numbers):
    """Helper function to process combined incidents for troubleshooting plan."""
    # Extract incidents from combined data
    incidents = combined_data.get('incidents', [])
    
    if len(incidents) < 2:
        logger.error("Combined incidents data must contain at least 2 incidents")
        return
    
    # First incident is the reference (with successful troubleshooting)
    reference_incident = incidents[0]
    # Second incident is the primary (needs troubleshooting plan)
    primary_incident = incidents[1]
    
    # Extract conversations from both incidents
    reference_conversation = reference_incident.get('conversation', [])
    primary_conversation = primary_incident.get('conversation', [])
    
    # Create a structured data format for the LLM
    structured_data = {
        "reference_incident": {
            "incident_id": reference_incident.get('incident_id', incident_numbers[0]),
            "conversation": reference_conversation,
            "total_entries": reference_incident.get('total_entries', 0)
        },
        "primary_incident": {
            "incident_id": primary_incident.get('incident_id', incident_numbers[1]),
            "conversation": primary_conversation,
            "total_entries": primary_incident.get('total_entries', 0)
        }
    }
    
    # Process using the processor with the structured data
    try:
        summary_result = processor.process_incident(
            incident_data=structured_data,
            prompts=prompts,
            prompt_type=prompt_type,
            debug_api=debug_api
        )
        
        # Display the summary
        if summary_result and 'summary' in summary_result:
            print("\nAI Generated Summary:")
            print("="*80)
            print(summary_result['summary'])
            print("="*80)
            
        
        # Save the results
        output_file = f"combined_{'_'.join(incident_numbers)}"
        processor.save_results(summary_result, output_file)
        
    except Exception as e:
        logger.error(f"Error processing combined incidents: {e}")
        print(f"Error processing combined incidents: {e}")
        raise

def _process_single_incident(processor, incident_data, prompts, prompt_type, debug_api, incident_id):
    """Helper function to process a single incident."""
    # Extract conversation data (needed for all modes)
    conversation = incident_data.get('conversation', [])

    # Defensive check to ensure conversation is always a list
    if not isinstance(conversation, list):
        logger.warning(f"Conversation data is not a list, got {type(conversation)}. Converting to empty list.")
        conversation = []

    # Extract summary images if available (multimodal data)
    summary_images = incident_data.get('summary_images', None)
    if summary_images:
        logger.info(f"Incident has {len(summary_images)} screenshot(s) from manual.docx")

    # Check if this is article search mode
    if prompt_type == 'article_search':
        # Use article search processing
        summary_result = processor.process_article_search(
            incident_data=incident_data,
            prompts=prompts,
            prompt_type=prompt_type,
            debug_api=debug_api
        )
        
        # Display incident summary first
        print("\n" + "="*80)
        print("INCIDENT SUMMARY")
        print("="*80)
        summary = incident_data.get('summary', 'No summary available')
        if summary and len(summary) > 0:
            # Truncate summary if it's too long for display
            display_summary = summary[:800] + "..." if len(summary) > 800 else summary
            print(display_summary)
        else:
            print("No incident summary available")
        print("="*80)
        
        # Display article search results
        if summary_result and 'analysis' in summary_result:
            print("\n" + "="*80)
            print("ARTICLE SEARCH RESULTS")
            print("="*80)
            print(summary_result['analysis'])
            print("="*80)
    elif prompt_type == 'prev_act':
        # Use preventative action processing
        summary = incident_data.get('summary', None)
        formatted_content = processor.format_conversation_with_ai_summary(conversation, summary=summary, summary_images=summary_images)

        # Check if formatted_content is multimodal (list) or text-only (string)
        if isinstance(formatted_content, list):
            # Multimodal content - pass directly
            content_for_llm = formatted_content
        else:
            # Text-only content - wrap in list
            content_for_llm = [{'type': 'text', 'content': formatted_content}]

        summary_result = processor.generate_summary(
            content_for_llm,
            prompts['system_prompt'],
            prompts['user_prompt'],
            prompt_type=prompt_type,
            debug_api=debug_api,
            incident_data=incident_data
        )
        
        # Display LLM analysis
        if summary_result and 'summary' in summary_result:
            print("\n" + "="*80)
            print("PREVENTATIVE ACTION ANALYSIS")
            print("="*80)
            print(summary_result['summary'])
            print("="*80)
        
        # Preventative action database management
        print("\n" + "="*80)
        print("PREVENTATIVE ACTION MANAGEMENT")
        print("="*80)
        
        try:
            analysis_text = summary_result.get('summary', '') if summary_result else ''
            
            # Interactive dialog to manage preventative actions database
            _interactive_preventative_action_dialog(incident_id, analysis_text, processor)
        
        except Exception as e:
            logger.error(f"Error in preventative action management: {e}")
            print(f"⚠️  Error in preventative action management: {e}")
            import traceback
            traceback.print_exc()
    elif prompt_type == 'logs_analyzer':
        # Use logs analyzer processing
        summary_result = processor.process_logs_analyzer(
            incident_data=incident_data,
            prompts=prompts,
            prompt_type=prompt_type,
            debug_api=debug_api,
            incident_id=incident_id
        )
        
        # Display incident summary first
        print("\n" + "="*80)
        print("INCIDENT SUMMARY")
        print("="*80)
        summary = incident_data.get('summary', 'No summary available')
        if summary and len(summary) > 0:
            # Truncate summary if it's too long for display
            display_summary = summary[:800] + "..." if len(summary) > 800 else summary
            print(display_summary)
        else:
            print("No incident summary available")
        print("="*80)
        
        # Display logs analysis results
        if summary_result and 'analysis' in summary_result:
            print("\n" + "="*80)
            print("LOGS ANALYSIS RESULTS")
            print("="*80)
            # Extract the actual analysis content from the result
            analysis_content = summary_result['analysis']
            if isinstance(analysis_content, dict) and 'summary' in analysis_content:
                print(analysis_content['summary'])
            else:
                print(analysis_content)
            print("="*80)
        
        # Display log analysis details
        if summary_result and 'log_analysis' in summary_result:
            log_analysis = summary_result['log_analysis']
            if 'analyzer_path' in log_analysis:
                print(f"\n📁 Analyzed logs from: {log_analysis['analyzer_path']}")
            if 'log_files' in log_analysis:
                print(f"📄 Analyzed {len(log_analysis['log_files'])} log files")
            if 'mde_logs' in log_analysis:
                print(f"🔍 Analyzed {len(log_analysis['mde_logs'])} security-specific log files")
    else:
        # Generate summary for other modes
        summary = incident_data.get('summary', None)

        # Check if this is team engagement mode - need to load Teams discussion CSV
        formatted_content = processor.format_conversation_with_ai_summary(conversation, summary=summary, summary_images=summary_images)

        # Check if formatted_content is multimodal (list) or text-only (string)
        if isinstance(formatted_content, list):
            # Multimodal content - pass directly
            content_for_llm = formatted_content
        else:
            # Text-only content - wrap in list
            content_for_llm = [{'type': 'text', 'content': formatted_content}]

        summary_result = processor.generate_summary(
            content_for_llm,
            prompts['system_prompt'],
            prompts['user_prompt'],
            prompt_type=prompt_type,
            debug_api=debug_api,
            incident_data=incident_data
        )
    
    # Save results (always save, but for prev_act_molecular we also launch PA manager)
    operation_time = datetime.now().isoformat()
    model_name = processor.deployment_name if hasattr(processor, 'deployment_name') else "unknown"
    
    try:
        processor.save_to_json(
            conversation,
            incident_id,
            ai_summary=summary_result,
            prompt_type=prompt_type,
            operation_time=operation_time,
            model_name=model_name
        )
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        # Continue processing even if save fails
    
    # Store memory about this incident
    if processor.memory_manager:
        try:
            processor.store_incident_memory(incident_id, incident_data, summary_result)
        except Exception as e:
            logger.error(f"Failed to store memory for incident {incident_id}: {e}")
    
    # Display team recommendations and pending updates (skip for prev_act)
    # Also skip for workflows that auto-save (1-4 and 12)
    auto_save_workflows = [
        'customer_pending_facilitation',
        'dev_pending_facilitation',
        'escalation',
        'mitigation',
        'create_prompt_for_logs_analyze',
        'create_prompt_for_logs_analyze_linux'
    ]

    if processor.team_knowledge_manager:
        try:
            # Display team recommendations if available in summary result
            if summary_result and 'team_recommendations' in summary_result:
                recommendations = summary_result['team_recommendations']
                if recommendations:
                    print("\n" + "=" * 80)
                    print("👥 TEAM RECOMMENDATIONS")
                    print("=" * 80)
                    for i, rec in enumerate(recommendations, 1):
                        print(f"{i}. {rec.get('team_name', 'Unknown')} (confidence: {rec.get('confidence', 0):.2f})")
                        evidence = rec.get('evidence', [])
                        if evidence:
                            print(f"   Evidence: {'; '.join(evidence[:2])}")
                    print("=" * 80)
            
            # Display transfer reasons if available
            if summary_result and 'transfer_reasons' in summary_result:
                transfer_reasons = summary_result['transfer_reasons']
                if transfer_reasons:
                    print("\n" + "=" * 80)
                    print("🔄 TRANSFER REASONS EXTRACTED")
                    print("=" * 80)
                    for reason in transfer_reasons[:3]:  # Show top 3
                        team_name = reason.get('team_name', 'Unknown')
                        reason_text = reason.get('transfer_reason', '')
                        evidence = reason.get('evidence', [])
                        print(f"Team: {team_name}")
                        print(f"Reason: {reason_text}")
                        if evidence:
                            print(f"Evidence: {evidence[0][:150]}...")
                        print("-" * 80)
                    print("=" * 80)
        except Exception as e:
            logger.warning(f"Failed to display team recommendations: {e}")

def _interactive_preventative_action_dialog(incident_id: str, analysis_text: str, processor):
    """Interactive dialog to manage preventative actions in Azure DevOps."""
    # Query Azure DevOps for active preventative actions
    print(f"\n🔍 Querying Azure DevOps for active preventative actions...")
    active_work_items = []
    ado_client = None
    try:
        from azure_devops_client import AzureDevOpsClient
        from config import config
        
        if config.azure_devops_pat:
            ado_client = AzureDevOpsClient(
                org=config.azure_devops_org,
                project=config.azure_devops_project,
                pat=config.azure_devops_pat
            )
            # Query for work items assigned to the configured default assignee, filtered by the configured custom field value
            active_work_items = ado_client.get_active_preventative_actions(
                assigned_to=config.azure_devops_default_assignee,
                custom_field_value=config.azure_devops_custom_field1_value,
                max_results=50
            )
        else:
            print("⚠️  Azure DevOps PAT not configured, skipping Azure DevOps query")
            return
    except Exception as e:
        logger.warning(f"Error querying Azure DevOps: {e}")
        print(f"⚠️  Could not query Azure DevOps: {e}")
        return
    
    if not ado_client:
        print("❌ Azure DevOps client not available")
        return
    
    # Display Azure DevOps work items
    if active_work_items:
        print(f"\n📋 Found {len(active_work_items)} active preventative action(s) in Azure DevOps:")
        for i, work_item in enumerate(active_work_items, 1):
            print(f"\n{i}. Work Item {work_item.get('id', 'unknown')}")
            print(f"   Title: {work_item.get('title', 'No title')}")
            print(f"   State: {work_item.get('state', 'Unknown')}")
            # Display custom fields
            related_count = work_item.get('related_incident_count')
            if related_count is not None:
                print(f"   Related Incident Count: {related_count}")
            related_ids = work_item.get('related_incident_ids')
            if related_ids:
                print(f"   Related Incident IDs: {related_ids}")
            repair_type = work_item.get('repair_item_type')
            if repair_type:
                print(f"   Repair Item Type: {repair_type}")
    
    if active_work_items:
        total_count = len(active_work_items)
        response = input(f"\nDoes this incident match any of the above? Enter number (1-{total_count}) or 'n' for new: ").strip().lower()
        
        if response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(active_work_items):
                selected_work_item = active_work_items[idx]
                work_item_id = selected_work_item.get('id')
                print(f"\n✅ Selected Azure DevOps Work Item: #{work_item_id} - {selected_work_item.get('title')}")
                
                # Update the work item with the new incident ID
                if ado_client.update_work_item_with_incident(work_item_id, incident_id):
                    print(f"✅ Updated work item #{work_item_id} with incident {incident_id}")
                else:
                    print(f"❌ Failed to update work item #{work_item_id}")
                return
    
    # Create new work item
    print(f"\n📝 Creating new preventative action work item...")
    
    # Ask for title
    title_input = input("Enter title: ").strip()
    if not title_input:
        print("❌ Title is required")
        return
    
    # Ask for Repair Item Type
    print("\nCommon Repair Item Types: Product Improvement, Process Enablement, Documentation, Technical Enablement, Diagnostic Tools")
    repair_type_input = input("Enter Repair Item Type: ").strip()
    if not repair_type_input:
        print("❌ Repair Item Type is required")
        return

    # Create the work item
    work_item_id = ado_client.create_preventative_action_work_item(
        title=title_input,
        repair_item_type=repair_type_input,
        incident_id=incident_id,
        description=analysis_text
    )
    
    if work_item_id:
        print(f"✅ Created new preventative action work item: #{work_item_id}")
        from config import config
        work_item_url = f"https://dev.azure.com/{config.azure_devops_org}/{config.azure_devops_project.replace(' ', '%20')}/_workitems/edit/{work_item_id}"
        print(f"   URL: {work_item_url}")
    else:
        print(f"❌ Failed to create work item")

def main():
    # Parse arguments first to check if timing is enabled
    processed_args = []
    for arg in sys.argv[1:]:
        if arg == "-5":
            processed_args.append("--azure-5")
        elif arg == "-4":
            processed_args.append("--azure")
        else:
            processed_args.append(arg)
    
    parser = argparse.ArgumentParser(description="Process multiple support incidents and provide unified summarization")
    parser.add_argument("incident_numbers", nargs="*", help="One or more incident numbers to process")
    parser.add_argument("--prompt", "-p", help="Type of prompt to use for summarization")
    # Always use AI Service (GPT-5) - no model selection needed
    parser.add_argument("--debug", "-d", action="store_true", help="Enable API debugging")
    parser.add_argument("--troubleshooting-plan", action="store_true", help="Generate troubleshooting plan mode - first incident is primary, others are historical references")
    parser.add_argument("--articles-embeddings", help="Path to article embeddings file (for article search mode)")
    parser.add_argument("--vector-db-path", help="Path to vector database file (for memory management)")
    parser.add_argument("--manual-docx", "--doc", "-doc", action="store_true", help="Use manual.docx content from fixed path instead of fetching incident data")
    parser.add_argument("--manual", action="store_true", help="Use manual text content instead of fetching incident data")
    parser.add_argument("--manual-file", help="Path to the manual text file to use with --manual (defaults to <project root>/manual.txt)")
    parser.add_argument("--markdown-file", "--md", "-md", help="Path to markdown file with incident summary and discussion")
    parser.add_argument("--timing", action="store_true", help="Enable detailed timing analysis and reporting")
    parser.add_argument("--teams", "-t", action="store_true", help="Enable team knowledge and team matching (disabled by default)")
    parser.add_argument("--multi-incident", action="store_true", help="Process multiple incidents directly (for debugging or specific use cases)")
    parser.add_argument("--input-file", help="Path to a JSON file containing incident data (for single or multi-incident mode)")
    parser.add_argument("--use-azure-ad", action="store_true", help="Use Azure AD / managed identity for AI service (overrides .env USE_AZURE_AD for this run)")

    args = parser.parse_args(processed_args)
    enable_timing = args.timing
    manual_docx_mode = args.manual_docx
    manual_txt_mode = args.manual
    manual_txt_path = args.manual_file
    markdown_file_mode = args.markdown_file
    markdown_file_path = args.markdown_file

    # Check for conflicts between input file modes
    if manual_docx_mode and manual_txt_mode:
        logger.error("--manual-docx and --manual cannot be combined")
        print("Error: --manual-docx and --manual cannot be combined")
        sys.exit(1)
    if manual_docx_mode and markdown_file_mode:
        logger.error("--manual-docx/-doc and --markdown-file/-md cannot be combined")
        print("Error: --manual-docx/-doc and --markdown-file/-md cannot be combined")
        sys.exit(1)

    if manual_docx_mode and args.input_file:
        logger.error("--manual-docx cannot be combined with --input-file")
        print("Error: --manual-docx cannot be combined with --input-file")
        sys.exit(1)
    if manual_txt_mode and args.input_file:
        logger.error("--manual cannot be combined with --input-file")
        print("Error: --manual cannot be combined with --input-file")
        sys.exit(1)
    if manual_txt_mode and markdown_file_mode:
        logger.error("--manual and --markdown-file/-md cannot be combined")
        print("Error: --manual and --markdown-file/-md cannot be combined")
        sys.exit(1)
    if manual_txt_path and not manual_txt_mode:
        logger.error("--manual-file requires --manual")
        print("Error: --manual-file requires --manual")
        sys.exit(1)

    if markdown_file_mode and args.input_file:
        logger.error("--markdown-file/-md cannot be combined with --input-file")
        print("Error: --markdown-file/-md cannot be combined with --input-file")
        sys.exit(1)

    # Apply Azure AD override from CLI (before any code uses config for AI)
    from config import config
    if getattr(args, 'use_azure_ad', False):
        config.use_azure_ad = True

    # Handle --input-file mode early (works for both single and multi-incident)
    if args.input_file:
        # Handle --multi-incident mode with input file
        if args.multi_incident:
            if not args.prompt:
                logger.error("--prompt is required when using --multi-incident mode")
                print("Error: --prompt is required when using --multi-incident mode")
                sys.exit(1)

            # Process directly using processor.py
            try:
                from processor import IncidentProcessor, load_prompts

                # Load prompts (free_text: generate from user input; else load from prompts.json)
                if args.prompt == 'free_text':
                    prompts = get_free_text_prompts()
                else:
                    prompts = load_prompts(args.prompt)
                
                # Initialize processor
                processor = IncidentProcessor(
                    enable_memory=True,
                    enable_team_analysis=args.teams,
                    articles_path=args.articles_embeddings,
                    vector_db_path=args.vector_db_path,
                    enable_timing=enable_timing
                )
                
                # Process multiple incidents
                processor.process_multiple_incidents(args.input_file, prompts, args.prompt, args.debug)
                return
            except Exception as e:
                logger.error(f"Error processing multiple incidents: {e}")
                print(f"Error: {e}")
                sys.exit(1)
        
        # Handle single incident with input file
        else:
            # Reset and start timing for the entire workflow only if enabled
            if enable_timing:
                reset_timing_data()
                start_timing()
            
            logger.info("=" * 80)
            logger.info("Starting Summarizer application with input file")
            logger.info("=" * 80)
            
            # Validate input file exists
            if not os.path.exists(args.input_file):
                logger.error(f"Input file not found: {args.input_file}")
                print(f"Error: Input file not found: {args.input_file}")
                sys.exit(1)
            
            # Load and adapt the file format if needed
            try:
                with open(args.input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Adapt file format if needed (convert content/ai_summary to conversation/summary)
                if 'content' in data and 'conversation' not in data:
                    data['conversation'] = data.pop('content')
                    logger.info("Adapted 'content' field to 'conversation'")
                
                if 'ai_summary' in data and 'summary' not in data:
                    if isinstance(data['ai_summary'], dict) and 'summary' in data['ai_summary']:
                        data['summary'] = data['ai_summary']['summary']
                    logger.info("Adapted 'ai_summary.summary' to 'summary'")
                
                # Extract incident number
                incident_number = data.get('incident_number') or data.get('incident_id') or os.path.basename(args.input_file).replace('.json', '').replace('incident_', '')

                # Select prompt type if not provided
                if not args.prompt:
                    logger.info("No prompt type specified, showing interactive menu")
                    prompt_type, auto_vector_db_path = show_prompt_menu()
                    if auto_vector_db_path and not args.vector_db_path:
                        args.vector_db_path = auto_vector_db_path
                        logger.info(f"Auto-detected vector database path: {auto_vector_db_path}")
                else:
                    prompt_type = args.prompt
                
                # Validate prompt type (skip for free_text - prompts are generated from user input)
                if prompt_type != 'free_text':
                    try:
                        with open("prompts.json", "r", encoding="utf-8") as f:
                            prompts_dict = json.load(f)
                        if prompt_type not in prompts_dict:
                            available = list(prompts_dict.keys())
                            error_msg = f"Prompt type '{prompt_type}' not found. Available types: {available}"
                            logger.error(error_msg)
                            print(error_msg)
                            sys.exit(1)
                    except Exception as e:
                        logger.error(f"Error reading prompts.json: {e}")
                        print(f"Error reading prompts.json: {e}")
                        sys.exit(1)
                
                # Set default vector database path if needed
                if prompt_type == 'article_search' and not args.vector_db_path:
                    from config import config
                    default_vector_db_path = config.default_vector_db_path
                    if default_vector_db_path:
                        args.vector_db_path = default_vector_db_path
                        logger.info(f"🔍 Article search mode detected - automatically using vector database: {default_vector_db_path}")
                        print(f"🔍 Article search mode detected - automatically using vector database: {default_vector_db_path}")
                
                # Load prompts (free_text: generate from user input; else load from prompts.json)
                from processor import IncidentProcessor, load_prompts
                if prompt_type == 'free_text':
                    prompts = get_free_text_prompts()
                else:
                    prompts = load_prompts(prompt_type)
                
                # Initialize processor
                processor = IncidentProcessor(
                    enable_memory=True,
                    enable_team_analysis=args.teams,
                    articles_path=args.articles_embeddings,
                    vector_db_path=args.vector_db_path,
                    enable_timing=enable_timing
                )
                
                # Process the incident
                print(f"Processing incident from file: {args.input_file}")
                print(f"Incident number: {incident_number}")
                print(f"Prompt type: {prompt_type}")
                
                if enable_timing:
                    from timing_utils import time_context
                    with time_context("ai_processing_detailed", "ai", {
                        "incident_count": 1,
                        "prompt_type": prompt_type,
                        "from_input_file": True
                    }):
                        _process_single_incident(processor, data, prompts, prompt_type, args.debug, incident_number)
                else:
                    _process_single_incident(processor, data, prompts, prompt_type, args.debug, incident_number)
                
                logger.info("AI processing completed successfully")
                
                # End timing and print summary only if timing is enabled
                if enable_timing:
                    end_timing()
                    print_timing_summary()
                    save_timing_report()
                
                logger.info("=" * 80)
                logger.info("Summarizer application completed successfully")
                logger.info("=" * 80)
                
                return
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON file: {e}")
                print(f"Error: Invalid JSON file: {e}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Error processing input file: {e}")
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
    
    # Validate that incident_numbers are provided for non-multi-incident mode
    if not args.multi_incident and not args.incident_numbers and not args.input_file:
        parser.error("incident_numbers are required when not using --multi-incident or --input-file mode")
    
    # Reset and start timing for the entire workflow only if enabled
    if enable_timing:
        reset_timing_data()
        start_timing()
    
    logger.info("=" * 80)
    logger.info("Starting Summarizer application")
    logger.info("=" * 80)
    
    # Extract arguments
    incident_numbers = args.incident_numbers
    prompt_type = args.prompt
    # Always use AI Service (GPT-5)
    use_ai_service_default = True
    debug_api = args.debug
    troubleshooting_plan_mode = args.troubleshooting_plan
    articles_embeddings = args.articles_embeddings
    vector_db_path = args.vector_db_path
    # Team analysis is disabled by default, can be enabled with --teams/-t
    enable_team_analysis = args.teams
    multi_incident_mode = args.multi_incident

    if manual_docx_mode and troubleshooting_plan_mode:
        logger.error("--manual-docx cannot be combined with --troubleshooting-plan")
        print("Error: --manual-docx cannot be combined with --troubleshooting-plan")
        sys.exit(1)
    if manual_txt_mode and troubleshooting_plan_mode:
        logger.error("--manual cannot be combined with --troubleshooting-plan")
        print("Error: --manual cannot be combined with --troubleshooting-plan")
        sys.exit(1)

    if markdown_file_mode and troubleshooting_plan_mode:
        logger.error("--markdown-file/-md cannot be combined with --troubleshooting-plan")
        print("Error: --markdown-file/-md cannot be combined with --troubleshooting-plan")
        sys.exit(1)

    if manual_docx_mode and multi_incident_mode:
        logger.error("--manual-docx cannot be combined with --multi-incident")
        print("Error: --manual-docx cannot be combined with --multi-incident")
        sys.exit(1)
    if manual_txt_mode and multi_incident_mode:
        logger.error("--manual cannot be combined with --multi-incident")
        print("Error: --manual cannot be combined with --multi-incident")
        sys.exit(1)

    if markdown_file_mode and multi_incident_mode:
        logger.error("--markdown-file/-md cannot be combined with --multi-incident")
        print("Error: --markdown-file/-md cannot be combined with --multi-incident")
        sys.exit(1)
    
    # Set default values from config only when needed (article_search); avoids loading HuggingFace model for other prompts
    if not articles_embeddings and prompt_type == 'article_search':
        from config import config
        articles_embeddings = config.default_vector_db_path

    logger.info(f"Command line arguments:")
    logger.info(f"  Incident numbers: {incident_numbers}")
    logger.info(f"  Prompt type: {prompt_type}")
    logger.info(f"  Use AI Service (GPT-5): {use_ai_service_default}")
    logger.info(f"  Debug API: {debug_api}")
    logger.info(f"  Troubleshooting plan mode: {troubleshooting_plan_mode}")
    logger.info(f"  Articles embeddings: {articles_embeddings}")
    logger.info(f"  Vector DB path: {vector_db_path}")
    logger.info(f"  Enable team analysis: {enable_team_analysis}")

    print(f"Processing {len(incident_numbers)} incident(s): {', '.join(incident_numbers)}")

    # Handle troubleshooting plan mode (disabled - prompt not in kept list)
    if troubleshooting_plan_mode:
        error_msg = "Troubleshooting plan mode is no longer supported (troubleshooting_plan_molecular prompt was removed)"
        logger.error(error_msg)
        print(f"Error: {error_msg}")
        sys.exit(1)

    if prompt_type is None:
        # If no prompt type specified, show interactive menu
        if prompt_type is None:
            logger.info("No prompt type specified, showing interactive menu")
            prompt_type, auto_vector_db_path = show_prompt_menu()
            # Use the automatically detected vector database path if available
            if auto_vector_db_path and not vector_db_path:
                vector_db_path = auto_vector_db_path
                logger.info(f"Auto-detected vector database path: {vector_db_path}")
        else:
            logger.info(f"Using specified prompt type: {prompt_type}")

    # Validate prompt_type before proceeding (skip for free_text - prompts are generated from user input)
    if prompt_type != 'free_text':
        logger.info("Validating prompt type...")
        try:
            with open("prompts.json", "r", encoding="utf-8") as f:
                prompts = json.load(f)
            if prompt_type not in prompts:
                available = list(prompts.keys())
                error_msg = f"Prompt type '{prompt_type}' not found. Available types: {available}"
                logger.error(error_msg)
                print(error_msg)
                sys.exit(1)
            logger.info(f"Prompt type '{prompt_type}' validated successfully")
        except Exception as e:
            logger.error(f"Error reading prompts.json: {e}")
            print(f"Error reading prompts.json: {e}")
            sys.exit(1)

    # Auto-detect vector database path for article search mode
    if prompt_type == 'article_search' and not vector_db_path:
        from config import config
        default_vector_db_path = config.default_vector_db_path
        if default_vector_db_path:
            vector_db_path = default_vector_db_path
            logger.info(f"🔍 Article search mode detected - automatically using vector database: {vector_db_path}")
            print(f"🔍 Article search mode detected - automatically using vector database: {vector_db_path}")
        else:
            logger.warning("⚠️ Article search mode detected but DEFAULT_VECTOR_DB_PATH not set in .env file")
            print("⚠️ Article search mode detected but DEFAULT_VECTOR_DB_PATH not set in .env file")

    successful_incidents = []
    preloaded_prompts = None  # set when prompt_type == 'free_text' and we run prompt generation in parallel

    if manual_docx_mode:
        if len(incident_numbers) != 1:
            error_msg = "--manual-docx requires exactly one incident number"
            logger.error(error_msg)
            print(f"Error: {error_msg}")
            sys.exit(1)
        incident_number = incident_numbers[0]

        if prompt_type == 'free_text':
            # Free text + manual docx: collect prompt first, then run generator LLM and manual.docx in parallel
            logger.info("Manual docx mode + free text: prompt first, then manual.docx and prompt generation in parallel")
            print("Free text + manual.docx: enter your prompt below. Press Enter on an empty line to submit.")
            user_description = read_free_text_input()
            docx_result = {"success": False, "error": None}

            def run_manual_docx():
                try:
                    process_manual_docx_only(incident_number)
                    docx_result["success"] = True
                except Exception as e:
                    docx_result["error"] = e

            docx_thread = threading.Thread(target=run_manual_docx)
            docx_thread.start()
            preloaded_prompts = get_free_text_prompts(user_description)
            docx_thread.join()

            if not docx_result["success"]:
                err = docx_result["error"]
                logger.error(f"Manual docx processing failed: {err}")
                print(f"Error: {err}")
                sys.exit(1)
            successful_incidents.append(incident_number)
        else:
            logger.info("Manual docx mode enabled; skipping database fetch and CSV processing")
            try:
                process_manual_docx_only(incident_number)
                successful_incidents.append(incident_number)
            except Exception as manual_error:
                logger.error(f"Manual docx processing failed: {manual_error}")
                print(f"Error: {manual_error}")
                sys.exit(1)
    elif manual_txt_mode:
        if len(incident_numbers) != 1:
            error_msg = "--manual requires exactly one incident number"
            logger.error(error_msg)
            print(f"Error: {error_msg}")
            sys.exit(1)
        incident_number = incident_numbers[0]

        if prompt_type == 'free_text':
            logger.info("Manual txt mode + free text: prompt first, then manual text processing and prompt generation in parallel")
            print("Free text + manual text: enter your prompt below. Press Enter on an empty line to submit.")
            user_description = read_free_text_input()
            txt_result = {"success": False, "error": None}

            def run_manual_txt():
                try:
                    process_manual_txt_only(incident_number, manual_txt_path)
                    txt_result["success"] = True
                except Exception as e:
                    txt_result["error"] = e

            txt_thread = threading.Thread(target=run_manual_txt)
            txt_thread.start()
            preloaded_prompts = get_free_text_prompts(user_description)
            txt_thread.join()

            if not txt_result["success"]:
                err = txt_result["error"]
                logger.error(f"Manual txt processing failed: {err}")
                print(f"Error: {err}")
                sys.exit(1)
            successful_incidents.append(incident_number)
        else:
            logger.info("Manual txt mode enabled; skipping database fetch and CSV processing")
            try:
                process_manual_txt_only(incident_number, manual_txt_path)
                successful_incidents.append(incident_number)
            except Exception as manual_error:
                logger.error(f"Manual txt processing failed: {manual_error}")
                print(f"Error: {manual_error}")
                sys.exit(1)
    elif markdown_file_mode:
        if len(incident_numbers) != 1:
            error_msg = "--markdown-file/-md requires exactly one incident number"
            logger.error(error_msg)
            print(f"Error: {error_msg}")
            sys.exit(1)
        incident_number = incident_numbers[0]

        if prompt_type == 'free_text':
            # Free text + markdown file: collect prompt first, then run generator LLM and markdown processing in parallel
            logger.info("Markdown file mode + free text: prompt first, then markdown and prompt generation in parallel")
            print(f"Free text + markdown: enter your prompt below. Press Enter on an empty line to submit.")
            user_description = read_free_text_input()
            md_result = {"success": False, "error": None}

            def run_markdown():
                try:
                    process_markdown_only(incident_number, markdown_file_path)
                    md_result["success"] = True
                except Exception as e:
                    md_result["error"] = e

            md_thread = threading.Thread(target=run_markdown)
            md_thread.start()
            preloaded_prompts = get_free_text_prompts(user_description)
            md_thread.join()

            if not md_result["success"]:
                err = md_result["error"]
                logger.error(f"Markdown file processing failed: {err}")
                print(f"Error: {err}")
                sys.exit(1)
            successful_incidents.append(incident_number)
        else:
            logger.info(f"Markdown file mode enabled; using {markdown_file_path}")
            try:
                process_markdown_only(incident_number, markdown_file_path)
                successful_incidents.append(incident_number)
            except Exception as md_error:
                logger.error(f"Markdown file processing failed: {md_error}")
                print(f"Error: {md_error}")
                sys.exit(1)
    elif prompt_type == 'free_text':
        # Free text: collect prompt first (no background output while typing), then fetch+transform and prompt generation in parallel
        print("Free text mode: enter your prompt below. Press Enter on an empty line to submit.")
        user_description = read_free_text_input()
        print("Fetching incident data and generating prompt...")
        fetch_result = {"successful_incidents": []}

        def run_fetch_and_transform():
            logger.info("=" * 50)
            logger.info("STEP 1: Fetching data from database (parallel)")
            logger.info("=" * 50)
            fetch_result["successful_incidents"] = _fetch_and_transform_incidents(incident_numbers)

        fetch_thread = threading.Thread(target=run_fetch_and_transform)
        fetch_thread.start()
        preloaded_prompts = get_free_text_prompts(user_description)
        fetch_thread.join()
        successful_incidents = fetch_result["successful_incidents"]

        if not successful_incidents:
            logger.error("No incidents were successfully fetched or processed. Exiting.")
            print("No incidents were successfully fetched or processed. Exiting.")
            sys.exit(1)
        logger.info(f"Successfully fetched data for {len(successful_incidents)} incident(s): {successful_incidents}")
        print(f"Successfully fetched data for {len(successful_incidents)} incident(s): {', '.join(successful_incidents)}")
    else:
        # Step 1: Fetch data for all incidents from database
        logger.info("=" * 50)
        logger.info("STEP 1: Fetching data from database")
        logger.info("=" * 50)
        
        for incident_number in incident_numbers:
            if fetch_incident_data(incident_number):
                successful_incidents.append(incident_number)
            else:
                logger.warning(f"Skipping incident {incident_number} due to fetch failure")
                print(f"Skipping incident {incident_number} due to fetch failure")
        
        if not successful_incidents:
            logger.error("No incidents were successfully fetched. Exiting.")
            print("No incidents were successfully fetched. Exiting.")
            sys.exit(1)
        
        logger.info(f"Successfully fetched data for {len(successful_incidents)} incident(s): {successful_incidents}")
        print(f"Successfully fetched data for {len(successful_incidents)} incident(s): {', '.join(successful_incidents)}")

        # Step 2: Process CSV to JSON for all successful incidents
        logger.info("=" * 50)
        logger.info("STEP 2: Converting CSV to JSON")
        logger.info("=" * 50)
        
        for incident_number in list(successful_incidents):
            try:
                process_incident_to_json(incident_number)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error processing incident {incident_number} to JSON: {e}")
                print(f"Error processing incident {incident_number} to JSON: {e}")
                successful_incidents.remove(incident_number)
        
        if not successful_incidents:
            logger.error("No incidents were successfully processed. Exiting.")
            print("No incidents were successfully processed. Exiting.")
            sys.exit(1)

    # Step 3: Combine data from all incidents
    logger.info("=" * 50)
    logger.info("STEP 3: Processing with AI")
    logger.info("=" * 50)
    
    if troubleshooting_plan_mode:
        # For troubleshooting plan mode, we need to create a special combined file
        # that clearly identifies the primary incident vs historical incidents
        logger.info("Creating troubleshooting plan data structure...")
        combined_json_path = create_troubleshooting_plan_data(successful_incidents)
        print("Processing incidents for troubleshooting plan generation...")
        ai_cmd = [
            sys.executable, "processor.py", combined_json_path, "--prompt-type", prompt_type, "--multi-incident"
        ]
        if enable_team_analysis:
            ai_cmd.append("--teams")
        if args.use_azure_ad:
            ai_cmd.append("--use-azure-ad")
    elif len(successful_incidents) > 1:
        logger.info("Combining multiple incidents for unified processing...")
        combined_json_path = combine_incident_data(successful_incidents)
        # Step 4: Process combined JSON with AI directly
        print("Processing combined incidents with AI...")
        
        # Import and call processor directly for multiple incidents to show output
        try:
            from processor import IncidentProcessor, load_prompts
            
            # Load the combined incident data
            with open(combined_json_path, 'r', encoding='utf-8') as f:
                incident_data = json.load(f)
            
            # Load prompts (free_text: use preloaded from parallel step or generate; else load from prompts.json)
            if prompt_type == 'free_text' and preloaded_prompts is not None:
                prompts = preloaded_prompts
            elif prompt_type == 'free_text':
                prompts = get_free_text_prompts()
            else:
                prompts = load_prompts(prompt_type)
            
            # Initialize processor (always uses AI Service GPT-5)
            processor = IncidentProcessor(
                enable_memory=True,
                enable_team_analysis=enable_team_analysis,
                articles_path=articles_embeddings,
                vector_db_path=vector_db_path,
                enable_timing=enable_timing
            )
            
            # Process the combined incidents
            if enable_timing:
                from timing_utils import time_context
                with time_context("ai_processing_detailed", "ai", {
                    "incident_count": len(successful_incidents),
                    "prompt_type": prompt_type,
                    "troubleshooting_plan_mode": False
                }):
                    _process_combined_incidents(processor, incident_data, prompts, prompt_type, debug_api, successful_incidents)
            else:
                _process_combined_incidents(processor, incident_data, prompts, prompt_type, debug_api, successful_incidents)
            
            logger.info("AI processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error in direct AI processing: {e}")
            # Fallback to subprocess approach
            ai_cmd = [
                sys.executable, "processor.py", combined_json_path, "--prompt-type", prompt_type, "--multi-incident"
            ]
            if enable_team_analysis:
                ai_cmd.append("--teams")
            if args.use_azure_ad:
                ai_cmd.append("--use-azure-ad")
            
            logger.info(f"Falling back to subprocess: {' '.join(ai_cmd)}")
            
            try:
                result = subprocess.run(ai_cmd, check=True)
                logger.info("AI processing completed successfully")
            except subprocess.CalledProcessError as e:
                logger.error("AI processing failed")
                logger.error(f"Return code: {e.returncode}")
                raise
    else:
        # Single incident - process directly for better timing granularity
        json_path = os.path.join("processed_incidents", f"{successful_incidents[0]}.json")
        logger.info(f"Processing single incident: {json_path}")
        print("Processing single incident with AI...")
        
        # Import and call processor directly for single incidents to get granular timing
        try:
            from processor import IncidentProcessor, load_prompts
            
            # Load the incident data
            with open(json_path, 'r', encoding='utf-8') as f:
                incident_data = json.load(f)
            
            # Load prompts (free_text: use preloaded from parallel step or generate; else load from prompts.json)
            if prompt_type == 'free_text' and preloaded_prompts is not None:
                prompts = preloaded_prompts
            elif prompt_type == 'free_text':
                prompts = get_free_text_prompts()
            else:
                prompts = load_prompts(prompt_type)
            
            # Initialize processor (always uses AI Service GPT-5)
            processor = IncidentProcessor(
                enable_memory=True,
                enable_team_analysis=enable_team_analysis,
                articles_path=articles_embeddings,
                vector_db_path=vector_db_path,
                enable_timing=enable_timing
            )
            
            # Process the incident
            if enable_timing:
                from timing_utils import time_context
                with time_context("ai_processing_detailed", "ai", {
                    "incident_count": 1,
                    "prompt_type": prompt_type,
                    "troubleshooting_plan_mode": False
                }):
                    _process_single_incident(processor, incident_data, prompts, prompt_type, debug_api, successful_incidents[0])
            else:
                _process_single_incident(processor, incident_data, prompts, prompt_type, debug_api, successful_incidents[0])
            
            logger.info("AI processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error in direct AI processing: {e}")
            # Fallback to subprocess approach
            ai_cmd = [
                sys.executable, "processor.py", json_path, "--prompt-type", prompt_type
            ]
            
            # Add debug argument if needed (always uses AI Service GPT-5)
            if debug_api:
                ai_cmd.append("--debug")
            
            # Add article search parameters if provided
            if articles_embeddings:
                ai_cmd.extend(["--articles-embeddings", articles_embeddings])
            if vector_db_path:
                ai_cmd.extend(["--vector-db-path", vector_db_path])
            if enable_team_analysis:
                ai_cmd.append("--teams")
            if args.use_azure_ad:
                ai_cmd.append("--use-azure-ad")
            
            logger.info(f"Falling back to subprocess: {' '.join(ai_cmd)}")
            
            try:
                result = subprocess.run(ai_cmd, check=True)
                logger.info("AI processing completed successfully")
            except subprocess.CalledProcessError as e:
                logger.error("AI processing failed")
                logger.error(f"Return code: {e.returncode}")
                raise
    
    # End timing and print summary only if timing is enabled
    if enable_timing:
        end_timing()
        
        logger.info("=" * 80)
        logger.info("Summarizer application completed successfully")
        logger.info("=" * 80)
        
        # Print timing summary
        print_timing_summary()
        
        # Save timing report
        save_timing_report()
    else:
        logger.info("=" * 80)
        logger.info("Summarizer application completed successfully")
        logger.info("=" * 80)

if __name__ == "__main__":
    main()
