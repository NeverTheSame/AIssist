import os
import csv
import time
import json
from azure.identity import InteractiveBrowserCredential
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
import re
from azure.kusto.data.exceptions import KustoNetworkError
from datetime import datetime
import traceback

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
    """Get a valid token, using cached token if not expired."""
    # Try to load from cache first
    cached_token = _load_cached_token()
    if cached_token:
        print("Using cached authentication token")
        return cached_token
    
    # Get new token
    print("Authenticating with Azure...")
    credential = InteractiveBrowserCredential()
    
    # Get token scope from environment variable
    token_scope = os.environ.get('AZURE_KUSTO_TOKEN_SCOPE', 'https://your-cluster.kusto.windows.net/.default')
    token_response = credential.get_token(token_scope)
    token = token_response.token
    expires_at = token_response.expires_on
    
    # Save to cache
    _save_token_to_cache(token, expires_at)
    print("Authentication successful - token cached for future use")
    
    return token

def remove_img_data_tags(text):
    """
    Replace all <img ...src="data:image..."> tags (even multiline, any attributes) with a placeholder.
    """
    if not isinstance(text, str):
        return text
    img_tag_pattern = re.compile(
        r'<img\b[^>]*src\s*=\s*([\'\"])data:image.*?\1[^>]*?>',
        re.IGNORECASE | re.DOTALL
    )
    return img_tag_pattern.sub('[IMAGE DATA SHRUNK - TODO: handle image data]', text)

def fetch_incident_to_csv(incident_number, kql_template_path, output_dir="icms"):
    """
    Fetch incident details from Azure Data Explorer and save as CSV in the icms directory.
    Args:
        incident_number (str or int): The incident number to fetch.
        kql_template_path (str): Path to the KQL query template file (should contain {incident_number}).
        output_dir (str): Directory to save the CSV file.
    Returns:
        str: Path to the saved CSV file.
    """
    # Get cluster and database from environment variables
    cluster = os.environ.get('AZURE_KUSTO_CLUSTER', 'https://your-cluster.kusto.windows.net')
    database = os.environ.get('AZURE_KUSTO_DATABASE', 'YourDatabase')
    
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

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{incident_number}.csv")
    section_headers = ["Discussions", "Internal AI Summary"]
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        image_data_shrunk = False  # Track if we've shrunk any image data
        for idx, query in enumerate(queries):
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
                for cell in row:
                    cleaned_cell = remove_img_data_tags(cell)
                    if isinstance(cell, str) and cleaned_cell != cell:
                        if not image_data_shrunk:
                            print("[kusto_fetcher] NOTE: Image data detected and shrunk in CSV output. TODO: Find a way to process this data.")
                            image_data_shrunk = True
                        processed_row.append('[IMAGE DATA SHRUNK - TODO: handle image data]')
                    else:
                        processed_row.append(cell)
                writer.writerow(processed_row)
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
        return
    except Exception as e:
        concise_msg = f"[kusto_fetcher] Error: {e}"
        print(concise_msg)
        # Log full stack trace to logs/fetcher.log
        os.makedirs("logs", exist_ok=True)
        with open("logs/fetcher.log", "a", encoding="utf-8") as logf:
            logf.write(f"{datetime.now().isoformat()} - {concise_msg}\n")
            logf.write(traceback.format_exc())
            logf.write("\n")
        return

if __name__ == "__main__":
    print("[kusto_fetcher] __main__ entrypoint")
    main() 