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
        
        # Database Configuration (ICM)
        self.database_cluster = os.environ.get('DATABASE_CLUSTER', 'https://your-cluster.kusto.windows.net')
        self.database_name = os.environ.get('DATABASE_NAME', 'YourDatabase')
        self.database_token_scope = os.environ.get('DATABASE_TOKEN_SCOPE', 'https://your-cluster.kusto.windows.net/.default')

        # Microsoft Defender Backend Configuration (for device telemetry like HeartbeatLog)
        self.defender_cluster = os.environ.get('DEFENDER_CLUSTER', '')
        self.defender_database = os.environ.get('DEFENDER_DATABASE', '')
        self.defender_token_scope = os.environ.get('DEFENDER_TOKEN_SCOPE', '')
        
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
        
        # Validate required configurations
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required configuration is present."""
        # Check AI Service configuration (required)
        ai_service_config = {
            'AI_SERVICE_API_KEY': self.ai_service_api_key,
            'AI_SERVICE_ENDPOINT': self.ai_service_endpoint,
            'AI_SERVICE_API_VERSION': self.ai_service_api_version,
            'AI_SERVICE_DEPLOYMENT_NAME': self.ai_service_deployment_name,
            'AI_SERVICE_MODEL_NAME': self.ai_service_model_name
        }
        
        # Validate that AI Service configuration is complete
        ai_service_complete = all(ai_service_config.values())
        
        if not ai_service_complete:
            missing_vars = [k for k, v in ai_service_config.items() if not v]
            raise ValueError(
                f"Missing required AI Service environment variables: {', '.join(missing_vars)}. "
                "Please check your .env file configuration."
            )
        
        # Validate Azure DevOps PAT if needed (optional, will be checked when Azure DevOps client is used)
        if not self.azure_devops_pat:
            # Don't raise error here, as Azure DevOps is only used for prev_act prompt type
            # Will be checked when AzureDevOpsClient is initialized
            pass

# Create a global config instance
config = Config() 