"""
Azure Authentication utilities for Cognitive Services.
Supports both API key and Azure AD/Managed Identity authentication.
"""

import os
import time
import json
import logging
from typing import Optional, Tuple
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ClientAuthenticationError

import guard

logger = logging.getLogger(__name__)
TOKEN_CACHE_FILE = ".azure_openai_token_cache.json"


def load_cached_token() -> Optional[str]:
    """Load cached token if valid."""
    if os.path.exists(TOKEN_CACHE_FILE):
        try:
            with open(TOKEN_CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            # Check if token is still valid (with 5 minute buffer)
            if time.time() < (cache_data['expires_at'] - 300):
                logger.info("Using cached Azure AD token")
                return cache_data['token']
        except Exception as e:
            logger.warning(f"Could not load cached token: {e}")
    return None


def save_token_to_cache(token: str, expires_at: float):
    """Save token to cache."""
    try:
        with open(TOKEN_CACHE_FILE, 'w') as f:
            json.dump({'token': token, 'expires_at': expires_at}, f)
        logger.info("Azure AD token cached for future use")
    except Exception as e:
        logger.warning(f"Could not cache token: {e}")


def get_azure_ad_token(force_refresh: bool = False) -> Optional[str]:
    """
    Get Azure AD token using DefaultAzureCredential.

    This function:
    1. Tries to use cached token if valid
    2. Falls back to DefaultAzureCredential which tries multiple sources:
       - Environment variables (for Managed Identity in Azure)
       - Managed Identity (when deployed to Azure)
       - Azure CLI (az login)
       - Interactive browser (as last resort)

    Args:
        force_refresh: If True, bypass cache and get new token

    Returns:
        Token string or None if authentication fails
    """
    # Try cache first (unless force refresh)
    if not force_refresh:
        cached = load_cached_token()
        if cached:
            return cached

    try:
        logger.info("Authenticating with Azure AD...")
        credential = DefaultAzureCredential()
        token_response = credential.get_token("https://cognitiveservices.azure.com/.default")
        # expires_on can be int (Unix timestamp) or datetime-like; normalize to float for cache
        expires_on = token_response.expires_on
        expires_at = float(expires_on) if isinstance(expires_on, (int, float)) else expires_on.timestamp()
        save_token_to_cache(token_response.token, expires_at)
        return token_response.token
    except ClientAuthenticationError as e:
        logger.error(f"Azure AD auth failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during Azure AD authentication: {e}")
        return None


def get_openai_client_with_auth(config) -> Tuple['AzureOpenAI', str]:
    """
    Create AzureOpenAI client with appropriate authentication.

    Args:
        config: Configuration object with authentication settings

    Returns:
        Tuple of (AzureOpenAI client, auth_method_used)
    """
    from openai import AzureOpenAI

    if config.use_azure_ad:
        token = get_azure_ad_token()
        if not token:
            raise ValueError(
                "Azure AD authentication failed. Please run 'az login' or configure Managed Identity. "
                "Alternatively, set USE_AZURE_AD=false in .env to use API key authentication."
            )
        client = AzureOpenAI(
            api_version=config.ai_service_api_version,
            azure_endpoint=config.ai_service_endpoint,
            azure_ad_token=token,
            timeout=300
        )
        logger.info(f"Using Azure AD authentication for endpoint: {config.ai_service_endpoint}")
        client = guard.wrap_client(client)
        return client, "azure_ad"
    else:
        client = AzureOpenAI(
            api_key=config.ai_service_api_key,
            api_version=config.ai_service_api_version,
            azure_endpoint=config.ai_service_endpoint,
            timeout=300
        )
        logger.info(f"Using API key authentication for endpoint: {config.ai_service_endpoint}")
        client = guard.wrap_client(client)
        return client, "api_key"
