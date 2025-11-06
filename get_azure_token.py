#!/usr/bin/env python3
"""
Helper script to get Azure access token for Kusto database access.

This script uses Azure CLI to get an access token that can be pasted
into the Streamlit app's Settings page.

Usage:
    python3 get_azure_token.py [kusto-cluster-url]
    
Example:
    python3 get_azure_token.py https://icmcluster.kusto.windows.net
"""

import sys
import subprocess
import json
from datetime import datetime, timedelta

def get_token_via_cli(resource_url):
    """Get Azure access token using Azure CLI."""
    try:
        # Get token using Azure CLI
        cmd = [
            'az', 'account', 'get-access-token',
            '--resource', resource_url,
            '--output', 'json'
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        token_data = json.loads(result.stdout)
        return token_data['accessToken'], token_data['expiresOn']
    
    except subprocess.CalledProcessError as e:
        print(f"Error: Azure CLI command failed: {e.stderr}", file=sys.stderr)
        print("\nMake sure you are logged in to Azure CLI:", file=sys.stderr)
        print("  az login", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Azure CLI not found. Please install it first:", file=sys.stderr)
        print("  https://docs.microsoft.com/en-us/cli/azure/install-azure-cli", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Failed to parse Azure CLI output", file=sys.stderr)
        sys.exit(1)

def main():
    # Default Kusto cluster URL (can be overridden)
    default_resource = "https://icmcluster.kusto.windows.net/.default"
    
    if len(sys.argv) > 1:
        resource_url = sys.argv[1]
        if not resource_url.endswith('.default'):
            resource_url = f"{resource_url}/.default"
    else:
        resource_url = default_resource
    
    print("Getting Azure access token...")
    print(f"Resource: {resource_url}\n")
    
    try:
        token, expires_on = get_token_via_cli(resource_url)
        
        # Parse expiry time
        expires_datetime = datetime.fromisoformat(expires_on.replace('Z', '+00:00'))
        expires_timestamp = expires_datetime.timestamp()
        now = datetime.now().timestamp()
        expires_in_hours = (expires_timestamp - now) / 3600
        
        print("=" * 80)
        print("ACCESS TOKEN (copy this to Streamlit Settings):")
        print("=" * 80)
        print(token)
        print("=" * 80)
        print(f"\nToken expires: {expires_on}")
        print(f"Expires in: {expires_in_hours:.1f} hours")
        print("\nTo use this token:")
        print("1. Copy the token above")
        print("2. Go to Streamlit app Settings page")
        print("3. Paste into 'Azure Access Token' field")
        print("4. Optionally set 'Token Expiry' to:", expires_on)
        print("\nNote: Tokens typically expire after 1 hour. Run this script again to get a new token.")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

