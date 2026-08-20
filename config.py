import os
from pathlib import Path

class Config:
    def __init__(self):
        # Get the project root directory
        self.root_dir = Path(__file__).parent.absolute()
        
        # Load environment variables from .env file
        self._load_env()
        
        # Initialize configuration
        self._init_config()
    
    def _load_env(self):
        """Load environment variables from .env file."""
        env_path = self.root_dir / '.env'
        if not env_path.exists():
            raise FileNotFoundError(
                f".env file not found at {env_path}. "
                "Please create a .env file with your configuration."
            )
        
        # Read and parse .env file
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Split on first '=' only
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith(('"', "'")) and value.endswith(('"', "'")):
                        value = value[1:-1]
                    
                    # Set environment variable
                    os.environ[key] = value
    
    def _init_config(self):
        """Initialize configuration from environment variables."""
        # AI Service Configuration (Primary)
        self.ai_service_api_key = os.environ.get('AI_SERVICE_API_KEY')
        self.ai_service_endpoint = os.environ.get('AI_SERVICE_ENDPOINT')
        self.ai_service_model_name = os.environ.get('AI_SERVICE_MODEL_NAME')
        self.ai_service_deployment_name = os.environ.get('AI_SERVICE_DEPLOYMENT_NAME')
        self.ai_service_api_version = os.environ.get('AI_SERVICE_API_VERSION')
        # Optional: faster/smaller model for free-text prompt generation (e.g. gpt-5-nano). If unset, uses AI_SERVICE_DEPLOYMENT_NAME.
        self.free_text_prompt_deployment_name = os.environ.get('FREE_TEXT_PROMPT_DEPLOYMENT_NAME') or None

        # Azure AD Authentication Configuration
        self.use_azure_ad = os.environ.get('USE_AZURE_AD', 'false').lower() == 'true'
        self.ai_resource_name = os.environ.get('AI_RESOURCE_NAME')
        
        # Database Configuration (Kusto)
        self.database_cluster = os.environ.get('DATABASE_CLUSTER', 'https://your-cluster.kusto.windows.net')
        self.database_name = os.environ.get('DATABASE_NAME', 'YourDatabase')
        self.database_token_scope = os.environ.get('DATABASE_TOKEN_SCOPE', 'https://your-cluster.kusto.windows.net/.default')

        # Secondary Telemetry Backend Configuration (optional, e.g. for device/endpoint telemetry)
        self.secondary_cluster = os.environ.get('SECONDARY_CLUSTER', '')
        self.secondary_database = os.environ.get('SECONDARY_DATABASE', '')
        self.secondary_token_scope = os.environ.get('SECONDARY_TOKEN_SCOPE', '')
        
        # Cost Configuration (for AI service)
        self.input_cost = float(os.environ.get('AI_SERVICE_INPUT_COST', '0.01'))  # Cost per 1K input tokens
        self.output_cost = float(os.environ.get('AI_SERVICE_OUTPUT_COST', '0.03'))  # Cost per 1K output tokens
        
        # Vision Service Configuration
        self.vision_api_key = os.environ.get('VISION_API_KEY')
        self.vision_endpoint = os.environ.get('VISION_ENDPOINT')
        
        # Article Search Configuration
        self.default_vector_db_path = os.environ.get('DEFAULT_ARTICLES_EMBEDDINGS_PATH')
        self.vector_db_path = os.environ.get('VECTOR_DB_PATH')
        self.articles_base_path = os.environ.get('ARTICLES_BASE_PATH')

        # Azure DevOps Configuration
        self.azure_devops_org = os.environ.get('AZURE_DEVOPS_ORG')
        self.azure_devops_project = os.environ.get('AZURE_DEVOPS_PROJECT')
        self.azure_devops_pat = os.environ.get('AZURE_DEVOPS_PAT')
        self.azure_devops_default_assignee = os.environ.get('AZURE_DEVOPS_DEFAULT_ASSIGNEE', '')
        self.azure_devops_custom_field1_value = os.environ.get('AZURE_DEVOPS_CUSTOM_FIELD1_VALUE', '')
        # Optional: a known work item ID used to sniff the real custom field reference name.
        # If unset, field reference detection falls back to trying common naming patterns.
        self.azure_devops_reference_work_item_id = os.environ.get('AZURE_DEVOPS_REFERENCE_WORK_ITEM_ID', '')
        # Custom field reference names for preventative action work items (process-template specific)
        self.azure_devops_repair_type_field = os.environ.get('AZURE_DEVOPS_REPAIR_TYPE_FIELD', 'Custom.RepairItemType')
        self.azure_devops_incident_ids_field = os.environ.get('AZURE_DEVOPS_INCIDENT_IDS_FIELD', 'Custom.IncidentIDs')
        self.azure_devops_incident_count_field = os.environ.get('AZURE_DEVOPS_INCIDENT_COUNT_FIELD', 'Custom.IncidentCount')

        # Optional noise filter for discussion entries from an automated/service account
        # (e.g. a bot that posts boilerplate enrichment text you want excluded from analysis)
        self.noise_filter_author = os.environ.get('NOISE_FILTER_AUTHOR', '')
        self.noise_filter_content_prefix = os.environ.get('NOISE_FILTER_CONTENT_PREFIX', '')

        # Optional: keywords used to detect security-agent-related content in incident text
        # (comma-separated, lowercase, e.g. names/acronyms of the EDR/antivirus product you support).
        # Leave unset to disable this tagging.
        self.security_agent_keywords = [k.strip().lower() for k in os.environ.get('SECURITY_AGENT_KEYWORDS', '').split(',') if k.strip()]
        self.security_agent_display_name = os.environ.get('SECURITY_AGENT_DISPLAY_NAME', 'Security Agent')

        # Optional: keywords used to detect auto-update-related content in incident text
        # (comma-separated, lowercase). Leave unset to disable this tagging.
        self.autoupdate_keywords = [k.strip().lower() for k in os.environ.get('AUTOUPDATE_KEYWORDS', '').split(',') if k.strip()]
        self.autoupdate_display_name = os.environ.get('AUTOUPDATE_DISPLAY_NAME', 'Auto-Update Service')

        # Diagnostic tool output file names to analyze from incident log bundles (comma-separated)
        self.diagnostic_log_files = [f.strip() for f in os.environ.get('DIAGNOSTIC_LOG_FILES', 'log.txt,console.txt,syslog.txt').split(',') if f.strip()]

        # Validate required configurations
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required configuration is present."""
        if self.use_azure_ad:
            # For Azure AD, API key is optional
            required = {
                'AI_SERVICE_ENDPOINT': self.ai_service_endpoint,
                'AI_SERVICE_API_VERSION': self.ai_service_api_version,
                'AI_SERVICE_DEPLOYMENT_NAME': self.ai_service_deployment_name
            }
            missing = [k for k, v in required.items() if not v]
            if missing:
                raise ValueError(f"Missing required config for Azure AD: {', '.join(missing)}")
            if not self.ai_service_api_key:
                print("Note: Using Azure AD authentication (API key not required)")
        else:
            # Original validation for API key auth
            ai_service_config = {
                'AI_SERVICE_API_KEY': self.ai_service_api_key,
                'AI_SERVICE_ENDPOINT': self.ai_service_endpoint,
                'AI_SERVICE_API_VERSION': self.ai_service_api_version,
                'AI_SERVICE_DEPLOYMENT_NAME': self.ai_service_deployment_name,
                'AI_SERVICE_MODEL_NAME': self.ai_service_model_name
            }
            if not all(ai_service_config.values()):
                missing = [k for k, v in ai_service_config.items() if not v]
                raise ValueError(f"Missing config: {', '.join(missing)}")
        
        # Validate Azure DevOps PAT if needed (optional, will be checked when Azure DevOps client is used)
        if not self.azure_devops_pat:
            # Don't raise error here, as Azure DevOps is only used for prev_act prompt type
            # Will be checked when AzureDevOpsClient is initialized
            pass

# Create a global config instance
config = Config()
