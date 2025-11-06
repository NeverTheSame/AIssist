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
import os
from datetime import datetime, timedelta

def get_token_via_cli(resource_url, extra_args=None):
    """Get Azure access token using Azure CLI."""
    try:
        # Get token using Azure CLI
        cmd = [
            'az', 'account', 'get-access-token',
            '--resource', resource_url,
            '--output', 'json'
        ]
        
        if extra_args:
            cmd.extend(extra_args)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        token_data = json.loads(result.stdout)
        return token_data['accessToken'], token_data['expiresOn']
    
    except subprocess.CalledProcessError as e:
        error_output = e.stderr or e.stdout or ""
        print(f"Error: Azure CLI command failed", file=sys.stderr)
        if error_output:
            print(f"Details: {error_output}", file=sys.stderr)
        
        # Provide helpful error messages
        if "AADSTS500011" in error_output or "resource principal" in error_output.lower():
            print("\n⚠️  Resource not found. This might mean:", file=sys.stderr)
            print("   - The cluster URL format is incorrect", file=sys.stderr)
            print("   - You need to use a different resource identifier", file=sys.stderr)
            print("\n💡 Try one of these commands manually:", file=sys.stderr)
            print(f"   az account get-access-token --resource {resource_url}", file=sys.stderr)
            print(f"   az account get-access-token --resource https://kusto.kusto.windows.net", file=sys.stderr)
            if extra_args:
                tenant_part = f" {' '.join(extra_args)}" if extra_args else ""
                print(f"   az account get-access-token --resource {resource_url}{tenant_part}", file=sys.stderr)
        
        print("\nTroubleshooting:", file=sys.stderr)
        print("1. Make sure you are logged in: az login", file=sys.stderr)
        if extra_args and '--tenant' in extra_args:
            tenant_idx = extra_args.index('--tenant')
            if tenant_idx + 1 < len(extra_args):
                tenant_val = extra_args[tenant_idx + 1]
                print(f"2. Make sure you're logged into the correct tenant: {tenant_val}", file=sys.stderr)
                print(f"   Try: az login --tenant {tenant_val}", file=sys.stderr)
        print("3. Verify the Kusto cluster URL is correct", file=sys.stderr)
        print("4. Check that your account has access to the cluster", file=sys.stderr)
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
    # For Kusto, use the cluster URL WITHOUT .default suffix
    default_resource = "https://icmcluster.kusto.windows.net"
    
    if len(sys.argv) > 1:
        resource_url = sys.argv[1]
        # Remove .default if present (wrong format for Kusto)
        resource_url = resource_url.rstrip('/.default')
    else:
        resource_url = default_resource
    
    print("Getting Azure access token...")
    print(f"Resource: {resource_url}\n")
    
    # Check if tenant ID provided as argument
    tenant_id = None
    if len(sys.argv) > 2 and sys.argv[2].startswith('--tenant'):
        tenant_id = sys.argv[2].split('=')[-1] if '=' in sys.argv[2] else sys.argv[3] if len(sys.argv) > 3 else None
    elif not tenant_id:
        tenant_id = os.environ.get('AZURE_TENANT_ID')
    
    cmd_extra = []
    if tenant_id:
        cmd_extra = ['--tenant', tenant_id]
        print(f"Using tenant: {tenant_id}\n")
    
    try:
        # Try the cluster URL directly (correct for Kusto)
        # For Azure Data Explorer, resource should be just the cluster URL
        token, expires_on = get_token_via_cli(resource_url, cmd_extra)
        
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

