#!/bin/bash
# Helper script to get Azure access token for Kusto database access
# Uses Azure CLI to get a token that can be pasted into Streamlit Settings

# Default Kusto cluster URL (modify if needed)
KUSTO_CLUSTER="${1:-https://icmcluster.kusto.windows.net}"

# Ensure it ends with .default
if [[ ! "$KUSTO_CLUSTER" == *".default" ]]; then
    KUSTO_CLUSTER="${KUSTO_CLUSTER}/.default"
fi

echo "Getting Azure access token..."
echo "Resource: $KUSTO_CLUSTER"
echo ""

# Get token using Azure CLI
TOKEN_DATA=$(az account get-access-token --resource "$KUSTO_CLUSTER" --output json 2>&1)

if [ $? -ne 0 ]; then
    echo "Error: Failed to get token. Make sure you are logged in:" >&2
    echo "  az login" >&2
    exit 1
fi

# Extract token and expiry
TOKEN=$(echo "$TOKEN_DATA" | grep -o '"accessToken": "[^"]*' | cut -d'"' -f4)
EXPIRES_ON=$(echo "$TOKEN_DATA" | grep -o '"expiresOn": "[^"]*' | cut -d'"' -f4)

if [ -z "$TOKEN" ]; then
    echo "Error: Failed to extract token from Azure CLI output" >&2
    exit 1
fi

echo "================================================================================"
echo "ACCESS TOKEN (copy this to Streamlit Settings):"
echo "================================================================================"
echo "$TOKEN"
echo "================================================================================"
echo ""
echo "Token expires: $EXPIRES_ON"
echo ""
echo "To use this token:"
echo "1. Copy the token above"
echo "2. Go to Streamlit app Settings page"
echo "3. Paste into 'Azure Access Token' field"
echo "4. Optionally set 'Token Expiry' to: $EXPIRES_ON"
echo ""
echo "Note: Tokens typically expire after 1 hour. Run this script again to get a new token."

