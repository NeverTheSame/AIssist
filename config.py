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
        # Azure OpenAI Configuration
        self.azure_openai_api_key = os.environ.get('AZURE_OPENAI_API_KEY')
        self.azure_openai_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
        self.azure_openai_api_version = os.environ.get('AZURE_OPENAI_API_VERSION')
        self.azure_openai_deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME')
        
        # Azure Kusto Configuration
        self.azure_kusto_cluster = os.environ.get('AZURE_KUSTO_CLUSTER', 'https://your-cluster.kusto.windows.net')
        self.azure_kusto_database = os.environ.get('AZURE_KUSTO_DATABASE', 'YourDatabase')
        self.azure_kusto_token_scope = os.environ.get('AZURE_KUSTO_TOKEN_SCOPE', 'https://your-cluster.kusto.windows.net/.default')
        
        # OpenAI Configuration
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        
        # ZAI Configuration
        self.zai_api_key = os.environ.get('ZAI_API_KEY')
        self.zai_base_url = os.environ.get('ZAI_BASE_URL')
        self.zai_model_name = os.environ.get('ZAI_MODEL_NAME', 'glm-4.5-air')
        self.zai_input_cost = float(os.environ.get('ZAI_FLASH_INPUT_COST', '0.2'))  # Cost per 1M input tokens
        self.zai_output_cost = float(os.environ.get('ZAI_FLASH_OUTPUT_COST', '1.1'))  # Cost per 1M output tokens
        
        # OpenAI Configuration
        self.openai_model_name = os.environ.get('OPENAI_MODEL_NAME', 'gpt-4-turbo-preview')
        self.openai_input_cost = float(os.environ.get('OPENAI_INPUT_COST', '0.01'))  # Cost per 1K input tokens
        self.openai_output_cost = float(os.environ.get('OPENAI_OUTPUT_COST', '0.03'))  # Cost per 1K output tokens
        
        # Azure Computer Vision Configuration
        self.azure_vision_key = os.environ.get('AZURE_VISION_KEY')
        self.azure_vision_endpoint = os.environ.get('AZURE_VISION_ENDPOINT')
        
        # Validate required configurations
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required configuration is present."""
        # Check Azure OpenAI configuration
        azure_config = {
            'AZURE_OPENAI_API_KEY': self.azure_openai_api_key,
            'AZURE_OPENAI_ENDPOINT': self.azure_openai_endpoint,
            'AZURE_OPENAI_API_VERSION': self.azure_openai_api_version,
            'AZURE_OPENAI_DEPLOYMENT_NAME': self.azure_openai_deployment_name
        }
        
        # Check OpenAI configuration
        openai_config = {
            'OPENAI_API_KEY': self.openai_api_key
        }
        
        # Check ZAI configuration
        zai_config = {
            'ZAI_API_KEY': self.zai_api_key,
            'ZAI_BASE_URL': self.zai_base_url
        }
        
        # Validate that at least one of the configurations is complete
        azure_complete = all(azure_config.values())
        openai_complete = all(openai_config.values())
        zai_complete = all(zai_config.values())
        
        if not (azure_complete or openai_complete or zai_complete):
            missing_vars = []
            if not azure_complete:
                missing_vars.extend([k for k, v in azure_config.items() if not v])
            if not openai_complete:
                missing_vars.extend([k for k, v in openai_config.items() if not v])
            if not zai_complete:
                missing_vars.extend([k for k, v in zai_config.items() if not v])
            
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                "Please check your .env file configuration."
            )

# Create a global config instance
config = Config() 