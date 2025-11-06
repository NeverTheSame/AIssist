#!/usr/bin/env python3
"""
Azure Blob Storage Bridge for Kusto Data

This script fetches incident data from Kusto (with VPN) and uploads it to
Azure Blob Storage, where Streamlit Cloud can access it.

Usage:
    python3 azure_blob_bridge.py <incident_number> [--container container-name]
    
Prerequisites:
    - Azure Storage Account and container
    - Set AZURE_STORAGE_CONNECTION_STRING environment variable
    - Or set AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY
"""

import sys
import os
import argparse
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError
import subprocess
import json
from datetime import datetime

def get_blob_client(container_name="incident-data"):
    """Initialize Azure Blob Storage client"""
    connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    account_name = os.environ.get('AZURE_STORAGE_ACCOUNT_NAME')
    account_key = os.environ.get('AZURE_STORAGE_ACCOUNT_KEY')
    
    if connection_string:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    elif account_name and account_key:
        from azure.storage.blob import BlobServiceClient
        account_url = f"https://{account_name}.blob.core.windows.net"
        blob_service_client = BlobServiceClient(account_url=account_url, credential=account_key)
    else:
        raise ValueError(
            "Azure Storage credentials not found. Set one of:\n"
            "- AZURE_STORAGE_CONNECTION_STRING\n"
            "- AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY"
        )
    
    # Create container if it doesn't exist
    try:
        container_client = blob_service_client.get_container_client(container_name)
        container_client.create_container()  # Safe - won't error if exists
    except AzureError:
        pass  # Container already exists
    
    return blob_service_client, container_name

def fetch_and_upload_incident(incident_number, container_name="incident-data"):
    """Fetch incident from Kusto and upload to Azure Blob Storage"""
    print(f"Fetching incident {incident_number} from Kusto...")
    
    # Step 1: Fetch from Kusto (using local kusto_fetcher.py with VPN)
    try:
        result = subprocess.run(
            [sys.executable, "kusto_fetcher.py", str(incident_number)],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Successfully fetched from Kusto")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to fetch from Kusto: {e.stderr}")
        return False
    
    # Step 2: Find the CSV file
    csv_path = Path(f"icms/{incident_number}/{incident_number}.csv")
    if not csv_path.exists():
        csv_path = Path(f"icms/{incident_number}.csv")
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return False
    
    # Step 3: Upload to Azure Blob Storage
    print(f"Uploading to Azure Blob Storage (container: {container_name})...")
    
    try:
        blob_service_client, container = get_blob_client(container_name)
        blob_client = blob_service_client.get_blob_client(
            container=container,
            blob=f"{incident_number}/{incident_number}.csv"
        )
        
        # Upload the file
        with open(csv_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        print(f"✅ Successfully uploaded to Azure Blob Storage")
        print(f"   Blob path: {container}/{incident_number}/{incident_number}.csv")
        
        # Also upload metadata
        metadata = {
            "incident_number": incident_number,
            "uploaded_at": datetime.now().isoformat(),
            "source": "local_fetch_with_vpn"
        }
        
        metadata_blob = blob_service_client.get_blob_client(
            container=container,
            blob=f"{incident_number}/metadata.json"
        )
        metadata_blob.upload_blob(
            json.dumps(metadata, indent=2),
            overwrite=True
        )
        
        return True
        
    except AzureError as e:
        print(f"❌ Failed to upload to Azure Blob Storage: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Fetch incident from Kusto and upload to Azure Blob Storage'
    )
    parser.add_argument('incident_number', help='Incident number to fetch')
    parser.add_argument(
        '--container',
        default='incident-data',
        help='Azure Blob Storage container name (default: incident-data)'
    )
    args = parser.parse_args()
    
    success = fetch_and_upload_incident(args.incident_number, args.container)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

