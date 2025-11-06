import os
import sys
import csv
import time
import json
import base64
import requests
# Token-based authentication only - no Azure identity imports needed
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
import re
from azure.kusto.data.exceptions import KustoNetworkError
from datetime import datetime
import traceback
from config import config

# Token cache file
TOKEN_CACHE_FILE = ".kusto_token_cache.json"

def _load_cached_token():
    """Load cached token from file if it exists and is still valid."""
    try:
        if os.path.exists(TOKEN_CACHE_FILE):
            with open(TOKEN_CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            
            # Check if token is still valid (with 5 minute buffer)
            if time.time() < (cache_data['expires_at'] - 300):
                return cache_data['token']
    except Exception as e:
        print(f"Warning: Could not load cached token: {e}")
    return None

def _save_token_to_cache(token, expires_at):
    """Save token to cache file."""
    try:
        cache_data = {
            'token': token,
            'expires_at': expires_at,
            'cached_at': time.time()
        }
        with open(TOKEN_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        print(f"Warning: Could not save token to cache: {e}")

def _get_valid_token():
    """Get a valid token - only supports direct token input.
    
    Token must be provided via AZURE_ACCESS_TOKEN or DATABASE_ACCESS_TOKEN environment variable.
    Get token by running: python3 get_azure_token.py
    """
    # Direct token input (ONLY method - no enterprise app needed!)
    access_token = os.environ.get('AZURE_ACCESS_TOKEN') or os.environ.get('DATABASE_ACCESS_TOKEN')
    if access_token:
        print("Using provided Azure access token")
        return access_token.strip()
    
    # Try to load from cache as fallback
    cached_token = _load_cached_token()
    if cached_token:
        print("Using cached authentication token")
        return cached_token
    
    # No token available
    raise ValueError(
        "Azure access token is required. Please:\n"
        "1. Run: python3 get_azure_token.py\n"
        "   Or: az account get-access-token --resource https://icmcluster.kusto.windows.net\n"
        "2. Copy the token to Settings page → 'Azure Access Token' field\n"
        "3. Click 'Save Configuration'\n\n"
        "Token expires after ~1 hour. Get a new token when it expires."
    )

def download_screenshot_from_data_url(data_url, output_path):
    """
    Download a screenshot from a data URL and save it to the specified path.
    
    Args:
        data_url (str): The complete data URL (e.g., "data:image/png;base64,iVBORw0...")
        output_path (str): Path where to save the screenshot
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract the base64 data from the URL
        if ',' not in data_url:
            print(f"[kusto_fetcher] Invalid data URL format: {data_url[:100]}...")
            return False
        
        # Split by comma to get the base64 data
        header, base64_data = data_url.split(',', 1)
        
        # Validate it's an image
        if not header.startswith('data:image/'):
            print(f"[kusto_fetcher] Not an image data URL: {header}")
            return False
        
        # Decode base64 data
        try:
            image_data = base64.b64decode(base64_data)
        except Exception as e:
            print(f"[kusto_fetcher] Failed to decode base64 data: {e}")
            return False
        
        # Determine file extension from MIME type
        mime_type = header.split(';')[0].split(':')[1]
        extension_map = {
            'image/png': '.png',
            'image/jpeg': '.jpg',
            'image/jpg': '.jpg',
            'image/gif': '.gif',
            'image/svg+xml': '.svg',
            'image/webp': '.webp'
        }
        extension = extension_map.get(mime_type, '.bin')
        
        # Ensure output path has correct extension
        if not output_path.endswith(extension):
            output_path = output_path + extension
        
        # Save the image
        with open(output_path, 'wb') as f:
            f.write(image_data)
        
        print(f"[kusto_fetcher] Screenshot saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"[kusto_fetcher] Error downloading screenshot: {e}")
        return False

def extract_and_download_screenshots(text, incident_number, output_dir):
    """
    Extract data URLs from text and download screenshots.
    
    Args:
        text (str): Text containing data URLs
        incident_number (str): Incident number for naming files
        output_dir (str): Directory to save screenshots
    
    Returns:
        tuple: (processed_text, screenshot_count)
    """
    if not isinstance(text, str):
        return text, 0
    
    # Pattern to match data URLs
    data_url_pattern = re.compile(
        r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+',
        re.IGNORECASE | re.DOTALL
    )
    
    screenshot_count = 0
    processed_text = text
    
    # Find all data URLs
    matches = data_url_pattern.findall(text)
    
    for i, data_url in enumerate(matches):
        # Create unique filename for each screenshot
        screenshot_filename = f"screenshot_{incident_number}_{i+1:03d}"
        screenshot_path = os.path.join(output_dir, screenshot_filename)
        
        # Download the screenshot
        if download_screenshot_from_data_url(data_url, screenshot_path):
            screenshot_count += 1
            # Replace the data URL in text with a reference to the saved file
            filename = os.path.basename(screenshot_path)
            if filename.endswith('.bin'):
                # Try to determine extension from the data URL
                if 'image/png' in data_url:
                    filename = filename.replace('.bin', '.png')
                elif 'image/jpeg' in data_url or 'image/jpg' in data_url:
                    filename = filename.replace('.bin', '.jpg')
                elif 'image/svg+xml' in data_url:
                    filename = filename.replace('.bin', '.svg')
            
            # Replace the data URL with a reference
            processed_text = processed_text.replace(data_url, f"[SCREENSHOT: {filename}]")
    
    return processed_text, screenshot_count

def remove_img_data_tags(text):
    """
    Replace encrypted image data in various formats with REDACTED.
    Handles:
    - <img src="data:image/png;base64,ENCRYPTED_DATA">
    - <img src="data:image/svg+xml;base64,ENCRYPTED_DATA">
    - background-image: url("data:image/svg+xml;base64,ENCRYPTED_DATA")
    - Any other data:image/*;base64,ENCRYPTED_DATA patterns
    """
    if not isinstance(text, str):
        return text
    
    # If text doesn't contain any image data URLs, return it unchanged
    # This ensures we don't accidentally process or modify text that doesn't need processing
    # This preserves any REDACTED markers or other content that comes from Kusto
    if 'data:image' not in text.lower():
        return text
    
    # Pattern to match any data:image/*;base64, followed by the encrypted data
    # This handles img tags, background-image CSS, and other data URL patterns
    # The pattern now properly handles quoted attributes and various base64 character sets
    data_url_pattern = re.compile(
        r'(data:image/[^;]+;base64,)[A-Za-z0-9+/=_-]+',
        re.IGNORECASE | re.DOTALL
    )
    
    # Replace the encrypted data part with REDACTED
    processed_text = data_url_pattern.sub(r'\1REDACTED', text)
    
    # Also handle any remaining base64 data that might have been missed
    # Look for any img tags with data URLs that still contain base64 data
    img_data_pattern = re.compile(
        r'(<img[^>]*src=["\'])(data:image/[^;]+;base64,)[A-Za-z0-9+/=_-]+([^"\']*["\'][^>]*>)',
        re.IGNORECASE | re.DOTALL
    )
    
    processed_text = img_data_pattern.sub(r'\1\2REDACTED\3', processed_text)
    
    return processed_text

def fetch_incident_to_csv(incident_number, kql_template_path, output_dir="icms"):
    """
    Fetch incident details from Azure Data Explorer and save as CSV in incident-specific folder.
    Args:
        incident_number (str or int): The incident number to fetch.
        kql_template_path (str): Path to the KQL query template file (should contain {incident_number}).
        output_dir (str): Base directory to save the incident folder.
    Returns:
        str: Path to the saved CSV file.
    """
    # Get cluster and database from config
    cluster = config.database_cluster
    database = config.database_name
    
    # Get valid token (cached if not expired)
    token = _get_valid_token()
    
    kcsb = KustoConnectionStringBuilder.with_aad_user_token_authentication(cluster, token)
    client = KustoClient(kcsb)

    # Read and split KQL queries by blank line
    with open(kql_template_path, "r") as file:
        kql_content = file.read()
    queries = [q.strip() for q in kql_content.split('\n\n') if q.strip()]

    if not queries:
        raise ValueError(f"No queries found in {kql_template_path}")

    # Filter out commented queries (lines starting with //)
    active_queries = []
    for query in queries:
        lines = query.split('\n')
        active_lines = [line for line in lines if not line.strip().startswith('//')]
        if active_lines:
            active_query = '\n'.join(active_lines).strip()
            if active_query:
                active_queries.append(active_query)

    if not active_queries:
        raise ValueError(f"No active queries found in {kql_template_path}")

    # Create incident-specific folder
    incident_folder = os.path.join(output_dir, str(incident_number))
    os.makedirs(incident_folder, exist_ok=True)
    
    output_path = os.path.join(incident_folder, f"{incident_number}.csv")
    section_headers = ["Discussions", "Authored summary"]
    
    total_screenshots = 0
    authored_summary_text = ""
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for idx, query in enumerate(active_queries):
            if idx > 0:
                writer.writerow([])  # Blank line between sections
            # Write section header
            writer.writerow([f"--- {section_headers[idx] if idx < len(section_headers) else 'Additional Result Set'} ---"])
            # Replace incident_number placeholder
            query_filled = query.replace("{incident_number}", str(incident_number))
            # Execute query
            response = client.execute(database, query_filled)
            table = response.primary_results[0]
            column_names = [col.column_name for col in table.columns]
            writer.writerow(column_names)
            for row in table:
                processed_row = []
                for cell_idx, cell in enumerate(row):
                    # Debug logging for Authored summary section to see raw Kusto data
                    if idx == 1 and cell_idx == 0:  # Authored summary section, first column (Summary)
                        cell_preview = str(cell)[:200] if cell else "None"
                        print(f"[kusto_fetcher] DEBUG: Raw cell from Kusto (Preview): {cell_preview}...")
                        if cell and isinstance(cell, str) and 'REDACTED' in cell:
                            print(f"[kusto_fetcher] WARNING: Cell contains REDACTED marker from Kusto (length: {len(cell)})")
                    cell_original = cell
                    # For authored summary section, collect text for screenshot processing
                    if idx == 1 and isinstance(cell, str):  # Authored summary section
                        authored_summary_text += cell + "\n"
                    
                    # Apply redaction only to image data URLs, preserving all other text content
                    if isinstance(cell, str):
                        cleaned_cell = remove_img_data_tags(cell)
                        # Additional debug logging if content was changed
                        if idx == 1 and cell_idx == 0 and cell_original != cleaned_cell:
                            print(f"[kusto_fetcher] DEBUG: Cell content was modified by remove_img_data_tags")
                            print(f"[kusto_fetcher] DEBUG: Original (first 200 chars): {str(cell_original)[:200]}")
                            print(f"[kusto_fetcher] DEBUG: Cleaned (first 200 chars): {str(cleaned_cell)[:200]}")
                    else:
                        cleaned_cell = cell
                    processed_row.append(cleaned_cell)
                writer.writerow(processed_row)
    
    # Process authored summary for screenshots
    if authored_summary_text:
        print(f"[kusto_fetcher] Processing authored summary for screenshots...")
        processed_summary, screenshot_count = extract_and_download_screenshots(
            authored_summary_text, 
            str(incident_number), 
            incident_folder
        )
        total_screenshots += screenshot_count
        
        # Save processed summary to a separate file
        summary_path = os.path.join(incident_folder, f"{incident_number}_summary_processed.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(processed_summary)
        
        if screenshot_count > 0:
            print(f"[kusto_fetcher] Downloaded {screenshot_count} screenshots to {incident_folder}")
    
    print(f"[kusto_fetcher] Incident data saved to: {incident_folder}")
    print(f"[kusto_fetcher] Total screenshots downloaded: {total_screenshots}")
    
    return output_path 

def main():
    import argparse
    print("[kusto_fetcher] main() started")
    parser = argparse.ArgumentParser(description='Fetch incident data from Azure Kusto')
    parser.add_argument('incident_number', help='The incident number to fetch')
    parser.add_argument('--kql-template', default='query.kql', 
                       help='Path to KQL query template')
    parser.add_argument('--output-dir', default='icms',
                       help='Directory to save CSV files')
    args = parser.parse_args()
    print(f"[kusto_fetcher] Args: {args}")
    try:
        print("[kusto_fetcher] Fetching incident to CSV...")
        csv_path = fetch_incident_to_csv(
            args.incident_number, 
            args.kql_template, 
            args.output_dir
        )
        print(f"[kusto_fetcher] CSV created at: {csv_path}")
    except KustoNetworkError as e:
        concise_msg = f"[kusto_fetcher] Network error: {e}\n[WARNING] Could not connect to Azure Kusto. Please ensure your VPN connection (e.g., 'MSFT-AzVPN-Manual') is active and try again."
        print(concise_msg)
        # Log full stack trace to logs/fetcher.log
        os.makedirs("logs", exist_ok=True)
        with open("logs/fetcher.log", "a", encoding="utf-8") as logf:
            logf.write(f"{datetime.now().isoformat()} - {concise_msg}\n")
            logf.write(traceback.format_exc())
            logf.write("\n")
        sys.exit(1)  # Exit with error code to indicate failure
    except Exception as e:
        concise_msg = f"[kusto_fetcher] Error: {e}"
        print(concise_msg)
        # Log full stack trace to logs/fetcher.log
        os.makedirs("logs", exist_ok=True)
        with open("logs/fetcher.log", "a", encoding="utf-8") as logf:
            logf.write(f"{datetime.now().isoformat()} - {concise_msg}\n")
            logf.write(traceback.format_exc())
            logf.write("\n")
        sys.exit(1)  # Exit with error code to indicate failure

if __name__ == "__main__":
    print("[kusto_fetcher] __main__ entrypoint")
    main() 