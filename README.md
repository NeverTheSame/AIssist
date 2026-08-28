# AI-Powered Incident Summarizer

## Overview

The AI-Powered Incident Summarizer is an advanced tool designed to automate the analysis and summarization of technical support incidents. It streamlines the incident management workflow by fetching data from databases (Azure Data Explorer/Kusto), processing it through intelligent AI models, and generating comprehensive summaries, troubleshooting guides, and actionable insights.

This tool is built for support teams to handle technical incidents more efficiently, providing consistent, high-quality summaries and reducing manual analysis time.

## What This Project Does

### Core Functionality
- **Automated Incident Analysis**: Fetches incident data from databases and processes it through AI models
- **Intelligent Summarization**: Generates various types of summaries including escalation notes, mitigation reports, and troubleshooting guides
- **Context-Aware Processing**: Uses memory systems to provide more relevant and consistent analysis
- **Article Search Integration**: Finds relevant troubleshooting articles and performs gap analysis against incident data
- **Multi-Incident Support**: Processes single incidents or combines multiple related incidents for unified analysis

### Key Features
- **Persistent Authentication**: Caches Azure authentication tokens to avoid repeated logins
- **Screenshot Processing**: Automatically downloads and processes embedded screenshots from incident data
- **Memory Integration**: Uses mem0 for persistent memory across sessions, learning from previous incidents
- **Gap Analysis**: Compares incident troubleshooting against knowledge base articles to identify missing steps
- **AI Service Integration**: Uses AI Service with GPT-5 for all AI operations

## Architecture

The tool follows a sophisticated three-stage pipeline with advanced AI integration:

### Stage 1: Data Fetching (`kusto_fetcher.py`)
- **Purpose**: Retrieves raw incident data from Azure Data Explorer (Kusto)
- **Features**: 
  - Persistent token caching for seamless authentication
  - Screenshot extraction and processing from embedded data URLs
  - Incident-specific folder organization
  - Network error handling with connectivity guidance for network-restricted clusters
- **Output**: CSV files with incident discussions and authored summaries

### Stage 2: Data Processing (`transformer.py`)
- **Purpose**: Transforms raw CSV data into structured JSON format optimized for AI processing
- **Features**:
  - HTML content cleaning and sanitization
  - Screenshot reference replacement
  - Data filtering and noise removal
  - Multi-section CSV parsing
- **Output**: Clean JSON files ready for AI analysis

### Stage 3: AI Analysis (`processor.py`)
- **Purpose**: Generates intelligent summaries and insights using advanced AI models
- **Features**:
  - Context-aware processing for enhanced prompts
  - Memory integration for learning from previous incidents
  - Article search and gap analysis capabilities
  - Multiple prompt types for different use cases
  - Cost tracking and token management
- **Output**: Comprehensive summaries, troubleshooting guides, and actionable insights

## Technologies Used

### Core Technologies
- **Python 3.12**: Primary programming language with virtual environment support
- **Azure Data Explorer (Kusto)**: Data source for incident information
- **AI Service**: Primary AI service for text generation and embeddings (GPT-5)

### AI and Machine Learning
- **mem0**: Universal memory layer for AI agents providing persistent context
- **Local Embeddings**: all-MiniLM-L6-v2 model for consistent semantic search
- **pgvector**: Postgres extension for KB/article retrieval (alternative to Qdrant, see `pgvector_store.py`)
- **TF-IDF Vectorization**: Fallback text similarity matching
- **Cosine Similarity**: Text matching algorithms for article search

### Data Processing
- **Pandas**: Data manipulation and CSV processing
- **BeautifulSoup4**: HTML content cleaning and parsing
- **NumPy**: Numerical operations for embeddings
- **scikit-learn**: Machine learning utilities for text processing

### Azure Services
- **Azure Identity**: Interactive browser authentication
- **Azure Kusto Data**: Database connectivity and query execution
- **Azure Cognitive Services**: Vision API for image processing

### Additional Libraries
- **tiktoken**: Token counting for cost estimation
- **python-docx**: Document processing capabilities
- **Pillow**: Image processing and manipulation
- **tqdm**: Progress bars for long-running operations
- **requests**: HTTP client for API calls

### MCP Integration
- **Model Context Protocol (MCP)**: Enables VS Code integration with GitHub Copilot Chat
- **Azure-Authenticated Kusto Access**: Secure database access through MCP proxy
- **Node.js MCP Proxy**: HTTP proxy server for Azure authentication
- **VS Code Integration**: Seamless integration with GitHub Copilot Chat

## Key Features

### Data Management
- **Automatic CSV Fetching**: Retrieves incident data from databases with persistent authentication
- **Screenshot Processing**: Downloads and processes embedded screenshots from incident data
- **Multi-Incident Support**: Handles single incidents or combines multiple related incidents
- **Data Sanitization**: Cleans HTML content and removes sensitive information

### AI-Powered Analysis
- **Intelligent Summarization**: Generates various types of summaries (escalation, mitigation, troubleshooting)
- **Context Engineering**: Dynamically enhances prompts with relevant examples
- **Memory Integration**: Learns from previous incidents to provide better context
- **Article Search**: Finds relevant troubleshooting articles using semantic search
- **Gap Analysis**: Identifies missing troubleshooting steps by comparing against knowledge base
- **Knowledge Base Generation**: Creates comprehensive runbooks and KB articles from incident data
- **Team Knowledge Management**: Tracks team expertise and interaction patterns
- **Process Improvement Analysis**: Identifies opportunities for process enhancement

### User Experience
- **Interactive Prompt Selection**: Presents menu of available prompt types
- **Flexible Configuration**: Supports multiple AI providers and custom settings
- **Cost Tracking**: Monitors token usage and API costs
- **Comprehensive Logging**: Detailed logs for debugging and monitoring

### Specialized Tools and Features
- **VIP Customer Tracking**: Specialized incident fetcher for high-priority customers
- **MCP Integration**: VS Code integration with GitHub Copilot Chat for Kusto queries
- **Team Knowledge System**: Automated team detection and expertise mapping
- **Documentation Generation**: Creates comprehensive runbooks following industry best practices
- **Process Improvement**: Analyzes incidents for optimization opportunities

## How It Works

### Complete Workflow
1. **Incident Data Retrieval**: Fetches incident data from databases using KQL queries
2. **Data Processing**: Converts raw CSV data to structured JSON format
3. **AI Analysis**: Processes data through AI models with context enhancement
4. **Memory Storage**: Stores results for future context and learning
5. **Output Generation**: Creates summaries, troubleshooting guides, and insights

### Memory System
- **Persistent Learning**: Stores processed incidents for future reference
- **Semantic Search**: Finds relevant previous incidents using vector embeddings
- **Context Enhancement**: Automatically adds relevant historical context to prompts
- **Cross-Session Persistence**: Memory survives across different processing sessions

## Setup

### Prerequisites
- Python 3.12 or higher
- Azure account with access to databases and AI services
- VPN or private network access, if your Kusto cluster is network-restricted

### Installation

1. **Clone the repository and set up your Python environment**

```bash
git clone <repository-url>
cd Summarizer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Configure Azure services**
- Set up Azure credentials for browser-based authentication
- Configure your KQL query template in `query.kql`
- Set up environment variables in `.env` file (see Configuration section)

3. **First-time authentication**
- Run the tool once to authenticate with Azure
- Authentication tokens are cached for future use

## Prompt Types and Use Cases

The system supports various prompt types for different analysis needs:

### Available Prompt Types
- **`customer_pending_facilitation`**: Creates facilitation notes when incident is pending customer action
- **`dev_pending_facilitation`**: Creates facilitation notes when incident is pending developer action
- **`escalation`**: Creates escalation summaries with team recommendations
- **`mitigation`**: Generates mitigation reports with structured troubleshooting sections
- **`prev_act`**: Recommends preventative actions from a predefined taxonomy
- **`article_search`**: Finds relevant troubleshooting articles from your knowledge base
- **`create_prompt_for_logs_analyze`**: Creates tailored investigation prompts for log analysis
- **`simplified_incident_explanation`**: Explains incidents in simple terms for non-experts

### Standard Prompt Types
- **`escalation_plain`**: Basic escalation summaries
- **`technical_article_plain`**: Technical documentation generation
- **`sentiment_analysis_plain`**: Customer sentiment analysis
- **`human_style_rewriting_plain`**: Human-like text rewriting

### Specialized Analysis
- **`troubleshooting_gap_analysis`**: Compares incident steps against knowledge base
- **`customer_care_facilitation`**: Customer care team facilitation summaries
- **`incident_delay_analysis`**: Identifies reasons for incident delays
- **`create_prompt_for_logs_analyze`**: Creates tailored investigation prompts for log analysis

## Configuration

### Required Configuration Files

**Note:** The following files are not included in this public repository due to containing private information. You must create them based on the examples below:

#### 1. KQL Query Template (`query.kql`)

Create a `query.kql` file with your Azure Data Explorer query. Use `{incident_number}` as a placeholder for the incident number:

```kql
cluster('your-cluster.kusto.windows.net').database('YourDatabase').table('Incidents') 
| where IncidentId in ("{incident_number}")
| project Date, ChangedBy, Text

cluster('your-cluster.kusto.windows.net').database('YourDatabase').table('AISummary') 
| where IncidentId in ("{incident_number}")
```

#### 2. Prompt Templates (`prompts.json`)

Create a `prompts.json` file with your prompt templates for different summarization types:

```json
{
  "customer_pending_facilitation": {
    "system_prompt": "You are an expert at preparing technical facilitation notes...",
    "user_prompt": "Create a facilitation note for this incident..."
  },
  "dev_pending_facilitation": {
    "system_prompt": "You are an expert at preparing technical facilitation notes...",
    "user_prompt": "Create a facilitation note for this incident..."
  },
  "escalation": {
    "system_prompt": "You are an expert at preparing technical incident escalations. Your role is to create clear, concise, and technically accurate summaries of unresolved incidents to hand off to engineering or specialized teams.",
    "user_prompt": "Summarize this incident for escalation to another team using exactly four paragraphs: 1) a short issue description, 2) relevant details including environmental context, 3) troubleshooting steps already taken, and 4) the customer's current goal or what support is needed to proceed."
  },
  "mitigation": {
    "system_prompt": "You are an expert at analyzing technical incidents...",
    "user_prompt": "Generate a mitigation report for this incident..."
  },
  "prev_act": {
    "system_prompt": "You are an expert at identifying preventative actions...",
    "user_prompt": "Analyze this incident and recommend preventative actions..."
  },
  "article_search": {
    "system_prompt": "You are an expert at finding relevant technical articles...",
    "user_prompt": "Search for articles relevant to this incident..."
  }
}
```

### Environment Variables (.env)

Create a `.env` file with the following variables:

**AI Service (Required):**
```
AI_SERVICE_API_KEY=your_api_key
AI_SERVICE_ENDPOINT=your_endpoint
AI_SERVICE_API_VERSION=2024-02-15-preview
AI_SERVICE_DEPLOYMENT_NAME=your_deployment_name
AI_SERVICE_MODEL_NAME=gpt-5
```

**Azure Kusto (Required for data fetching):**
```
DATABASE_CLUSTER=https://your-cluster.kusto.windows.net
DATABASE_NAME=YourDatabase
DATABASE_TOKEN_SCOPE=https://your-cluster.kusto.windows.net/.default
```

**Azure DevOps (Optional - for preventative actions workflow):**
```
AZURE_DEVOPS_ORG=your-organization
AZURE_DEVOPS_PROJECT=your-project
AZURE_DEVOPS_PAT=your-personal-access-token
AZURE_DEVOPS_DEFAULT_ASSIGNEE=your-name
AZURE_DEVOPS_CUSTOM_FIELD1_VALUE=your-custom-field-value
AZURE_DEVOPS_REPAIR_TYPE_FIELD=Custom.RepairItemType
AZURE_DEVOPS_INCIDENT_IDS_FIELD=Custom.IncidentIDs
AZURE_DEVOPS_INCIDENT_COUNT_FIELD=Custom.IncidentCount
```

**Noise Filter (Optional - excludes boilerplate entries from a specific automated/service account):**
```
NOISE_FILTER_AUTHOR=service-account-name
NOISE_FILTER_CONTENT_PREFIX=Boilerplate text prefix to match
```

**Article Search (Optional - for article search mode):**
```
DEFAULT_ARTICLES_EMBEDDINGS_PATH=/path/to/article_embeddings.json
VECTOR_DB_PATH=/path/to/qdrant_db
ARTICLES_BASE_PATH=/path/to/articles

# pgvector backend (optional; takes priority over Qdrant/JSON when set)
PGVECTOR_DSN=postgresql://user:password@host:5432/dbname
PGVECTOR_TABLE=articles
```

## How to Run

The main entry point for the application is `main.py`. Here are the primary ways to run the application:

### Basic Command

```bash
python3 main.py <incident_number> [options]
```

### Complete Workflow (Recommended)
The `main.py` script handles the entire pipeline automatically:

Fetch, process, and summarize an incident with interactive prompt selection:
```bash
python main.py 100000001
```

Fetch, process, and summarize an incident with specific escalation prompt:
```bash
python main.py 100000001 --prompt-type escalation
```

All operations use AI Service (GPT-5) by default.

Process multiple incidents with unified summarization:
```bash
python main.py 100000001 654045298 654045299 --prompt-type escalation
```

Generate troubleshooting plan based on historical incidents:
```bash
python main.py 100000001 654045298 654045299 654045300 --troubleshooting-plan
```

Generate weekly insights for multiple incidents (CRI Weekly Insights):
```bash
# Process multiple incidents from scratch
python3 main.py 100000005 100000006 100000007 100000008 100000009 --prompt-type weekly_insights

# Process from existing combined JSON file
python3 main.py --multi-incident --input-file processed_incidents/incident_combined_100000005_100000006_100000007_100000008_100000009.json --prompt-type weekly_insights
```

Generate knowledge base article from incident data:
```bash
python main.py 100000001 --prompt-type kb_article
```

Analyze incident for improvement opportunities:
```bash
python main.py 100000001 --prompt-type improvement_analysis
```

Fetch VIP customer incidents:
```bash
cd vip_incidents
python3 fetch_vip_incidents.py
```

## How to Test

### Testing the Application

**Note:** This project currently doesn't include a `test_main.py` file or automated unit tests. Testing is done through manual execution with sample incident data:

#### 1. Test with Sample Incident
```bash
# Test with a known incident number
python3 main.py 100000004 --prompt-type escalation
```

#### 2. Test Different Prompt Types
```bash
# Test escalation summaries
python3 main.py 100000004 --prompt-type escalation

# Test mitigation reports  
python3 main.py 100000004 --prompt-type mitigation

# Test troubleshooting guides
python3 main.py 100000004 --prompt-type troubleshooting
```

#### 3. Test Article Search Functionality
```bash
# Test article search mode (requires vector database)
python3 main.py 100000004 --prompt-type article_search
```

#### 4. Test Memory Integration
```bash
# Process multiple incidents to test memory learning
python3 main.py 100000004
python3 main.py 100000004  # Second run should use memory context
```

#### 5. Test Different Prompt Types
```bash
# Test different prompt types
python3 main.py 100000004 --prompt-type escalation
python3 main.py 100000004 --prompt-type mitigation
```

### Validation Steps

1. **Check Output Files**: Verify that summaries are generated in the `summaries/` directory
2. **Review Logs**: Check `logs/summarizer.log` for any errors or warnings
3. **Memory Verification**: Confirm memory storage in `memory/` directory
4. **Cost Tracking**: Monitor token usage and costs in the output

### Troubleshooting Tests

If you encounter issues, test these scenarios:

```bash
# Test with debug mode for detailed output
python3 main.py 100000004 --debug

# Test with timing analysis
python3 main.py 100000004 --timing

# All operations use AI Service (GPT-5)
python3 main.py 100000004
```

### Advanced Usage

#### Multi-Incident Command

```bash
python3 main.py <incident_number1> <incident_number2> ... [options]
```

#### Article Search and Gap Analysis

The tool includes advanced article search functionality and gap analysis capabilities:

```bash
# Search for relevant troubleshooting articles
python3 main.py <incident_number> --prompt-type article_search --vector-db-path article_vector_db.json

# Search using text files directly
python3 main.py <incident_number> --prompt-type article_search --articles-path /path/to/articles

# Setup article search from text files
python3 setup_article_search.py --setup /path/to/articles --output article_vector_db.json

# Test article search functionality
python3 setup_article_search.py --test article_vector_db.json --query "agent crashes"

# Run gap analysis after article search
python3 gap_analysis.py <incident_number>
python3 simple_gap_analysis.py <incident_number>
```

#### Gap Analysis Feature

The gap analysis feature compares incident troubleshooting steps against comprehensive knowledge base articles to identify missing steps:

- **Intelligent Comparison**: Analyzes what troubleshooting has been done vs. what should be done
- **Prioritized Action Plan**: Creates high/medium/low priority execution plans
- **Real Content Retrieval**: Accesses actual troubleshooting content from local knowledge base directory
- **Azure OpenAI Integration**: Uses Azure OpenAI for intelligent analysis and gap identification
- **Execution Plans**: Generates specific commands and expected outcomes

#### Available Options
- `--prompt-type TYPE`   Type of prompt (customer_pending_facilitation, dev_pending_facilitation, escalation, mitigation, prev_act, article_search, create_prompt_for_logs_analyze, simplified_incident_explanation)
- All operations use AI Service (GPT-5) by default
- `--debug`              Enable API debugging
- `--articles-path PATH` Path to directory containing troubleshooting articles (for article search mode)
- `--vector-db-path PATH` Path to vector database file (for article search mode)
- `--summ`               Include summary from summary.txt
- `--summ-docx`          Use summary.docx as input
- `--troubleshooting-plan` Generate troubleshooting plan mode (first incident is primary, others are historical references)
- `--multi-incident`     Process multiple incidents directly from existing JSON file (requires --input-file)
- `--input-file PATH`    Path to JSON file containing combined incident data (for use with --multi-incident)
- `--timing`             Enable detailed timing analysis and reporting
- `--teams`, `-t`        Enable team knowledge and team matching (disabled by default)

**Note:** If no `--prompt-type` is specified, the tool will display an interactive menu showing only prompt types for selection.

#### Manual Stage-by-Stage Processing
You can also run each stage manually:

**Stage 1: Fetch data from Kusto**
```bash
python kusto_fetcher.py 100000001
```

**Stage 2: Process CSV to JSON**
```bash
python transformer.py incidents/100000001/100000001.csv
```

**Stage 3: Generate AI insights**
```bash
python processor.py processed_incidents/100000001.json --prompt-type escalation
```

### How It Works

The tool follows a three-stage pipeline:

#### Stage 1: Data Fetching (`kusto_fetcher.py`)
1. **Fetches incident data from Azure Kusto** using the incident number(s) and your `query.kql` template.
2. **Caches the Azure authentication token** in `.kusto_token_cache.json` for reuse until it expires.
3. **Creates incident-specific folders** (e.g., `incidents/100000002/`) for each incident.
4. **Downloads embedded screenshots** from data URLs and saves them as image files.
5. **Saves raw data** as CSV files in the incident folders.

#### Stage 2: Data Processing (`transformer.py`)
1. **Reads the raw CSV data** from incident-specific folders.
2. **Cleans and structures the data** by removing HTML formatting and filtering unwanted entries.
3. **Extracts authored summaries** and processes them for LLM consumption.
4. **Outputs structured JSON** files optimized for LLM processing.

#### Stage 3: AI Analysis (`processor.py`)
1. **For single incidents**: Processes and summarizes the incident using AI Service (GPT-5).
2. **For multiple incidents**: Combines all incident data and generates a unified summary.
3. **Outputs results** to the appropriate directories (`processed_incidents/`, `summaries/`).

### Troubleshooting Plan Mode

**Note:** The troubleshooting plan mode has been removed. The `troubleshooting_plan` prompt type is no longer available.

### Weekly Insights Mode (CRI Weekly Insights)

The `weekly_insights` prompt is designed to create concise, actionable weekly status updates for multiple ongoing technical incidents, focusing on longest-running incidents and their blockers.

**Features:**
- **Active Days Calculation**: Automatically calculates and displays the number of days each incident has been active
- **Blocker Emphasis**: Highlights the primary factors causing extended duration for each incident
- **Overall Summary**: Provides a comprehensive summary paragraph outlining overall delays and common blockers across all incidents
- **Formatted Output**: Each incident appears on separate lines with clear formatting for easy reading

**How it works:**
1. **Process Multiple Incidents**: Provide multiple incident numbers to analyze together
2. **Active Days Calculation**: The system calculates days active from the first timestamp in the incident conversation
3. **Status Analysis**: For each incident, provides current status, days active, primary blocker, and next action
4. **Summary Generation**: Creates an overall summary with average active days and common blocker patterns

**Usage Examples:**

Process multiple incidents from scratch:
```bash
python3 main.py 100000005 100000006 100000007 100000008 100000009 --prompt-type weekly_insights
```

Process from existing combined JSON file (when you already have processed incident data):
```bash
python3 main.py --multi-incident --input-file processed_incidents/incident_combined_100000005_100000006_100000007_100000008_100000009.json --prompt-type weekly_insights
```

**Output Format:**
Each incident includes:
- Incident number and title
- Current status and trend (improving/degrading/stable)
- Number of days active
- Primary blocker or factor causing extended duration
- Immediate next action with timeline

Followed by:
- **Summary**: Overall analysis with average active days and primary reasons these incidents are still open

**Example Output:**
```
Incident #100000005 - Duplicate Device Records in Admin Portal
Status: Stable but unresolved; duplicates persist despite multiple cleanup attempts. Active for 42 days. Primary blocker is a root cause involving backend and device state inconsistencies. Next: Await customer logs and backend team analysis; follow-up within 3 days.

Incident #100000006 - Intermittent Connectivity After Sleep/Wake
Status: Stable; transient connectivity state occurs after device sleep. Active for 18 days. Blocker is a reconnection delay and status update latency. Next: Platform team to confirm expected behavior; update by end of week.

Summary:
The open incidents have been active on average 30 days, with primary delays caused by multi-team coordination challenges, complex backend and client state issues, and configuration management complexities. Engineering teams are prioritizing targeted remediation actions with defined timelines.
```

**Best Practices:**
- Use for weekly status reviews of longest-running incidents
- Focus on incidents that need leadership attention due to extended duration
- Include incidents that are blocked by similar root causes for better pattern identification
- Review the summary section to identify common blockers that may need process improvements

## Memory Integration

The summarizer includes memory integration using [mem0](https://github.com/mem0ai/mem0), a universal memory layer for AI agents. This enables the tool to learn from previous incidents and provide more context-aware analysis.

### How Memory Works

- **Persistent Memory**: Stores information about processed incidents, including summaries, key findings, and technical details
- **Context Enhancement**: Searches for relevant previous incidents and enhances prompts with this context
- **Improved Consistency**: Maintains consistency in analysis and recommendations across similar incidents
- **Learning Over Time**: Becomes more effective at identifying patterns as more incidents are processed
- **Cross-Session Persistence**: Memory persists across different processing sessions

### Vector Database Architecture

Article retrieval (`article_searcher.py`) supports two vector backends behind
the same interface: set `PGVECTOR_DSN` to use Postgres+pgvector, or leave it
unset to use Qdrant (the default). Whichever is active, `_semantic_search()`
returns candidates in the same shape, so nothing downstream (scoring,
re-ranking, gap analysis) needs to know which one answered.

#### pgvector

Set `PGVECTOR_DSN` (a `postgresql://` URL) and, optionally, `PGVECTOR_TABLE`
(default `articles`) in `.env`. See `pgvector_store.py`:

- `ensure_schema()` creates the article table and an HNSW cosine index on
  first use (`CREATE EXTENSION IF NOT EXISTS vector`, so a plain Postgres
  instance with the pgvector extension installed is enough)
- `upsert_articles()` writes `(article_path, title, content_summary,
  embedding)` rows, keyed on `article_path`
- `search()` runs `ORDER BY embedding <=> $1::vector` and returns
  `article_path` / `title` / `content_summary` / `semantic_similarity`
- Embeddings are the same all-MiniLM-L6-v2, 384-dimension vectors used
  everywhere else in this file, so a pgvector table and a Qdrant collection
  are interchangeable for the same article corpus

Tested against a real `pgvector/pgvector:pg16` container (`tests/test_pgvector_store.py`,
skipped unless `PGVECTOR_TEST_DSN` is set — see that file's docstring for the
one-line docker command). This is the honesty pattern used elsewhere in this
codebase for anything that needs live infrastructure: not run means not
claimed, not "should work."

#### Qdrant (default)

The summarizer uses Qdrant as its vector database, which provides several key benefits:

#### Semantic Search Capabilities
- **Vector Embeddings**: Each incident memory is converted to a high-dimensional vector using Azure OpenAI embeddings
- **Similarity Search**: Finds the most semantically similar incidents based on content, not just keywords
- **Fast Retrieval**: Optimized for real-time search across large memory databases

#### Storage Benefits
- **Persistent Storage**: Vector database persists across sessions and project iterations
- **Cross-Project Sharing**: Multiple projects can share the same memory database
- **Scalable**: Handles thousands of incident memories efficiently
- **No Data Loss**: Memory survives project deletion/recreation

#### Technical Details
- **Embedding Model**: BAAI/bge-large-en-v1.5 (1536 dimensions)
- **Distance Metric**: Cosine similarity for semantic matching
- **Storage Type**: File-based persistent storage (not in-memory)
- **Collection**: Single collection named "mem0" for all memories
- **Dimension Consistency**: All embeddings standardized to 1536 dimensions for optimal performance

### Using Memory

Memory is enabled by default. Simply run your processor as usual:

```bash
python processor.py incident.json --prompt-type escalation
```

The processor will automatically:
1. Store the processing result in memory
2. Enhance future prompts with relevant context from previous incidents

#### Terminal Output

When memory is active, you'll see confirmation messages in the terminal:

```
✅ Using mem0 with Azure OpenAI embeddings for memory storage and semantic search
🧠 Enhanced prompt with memory context from previous incidents
💾 Stored memory for incident 123456789
```

These messages confirm that:
- mem0 is successfully initialized with vector database
- Memory context was added to the prompt
- Incident data was stored in the vector database

**Disable Memory**: Use the `--no-memory` flag to disable memory for a specific processing session:
```bash
python processor.py incident.json --no-memory
```

### Memory Storage

The summarizer uses [mem0](https://github.com/mem0ai/mem0) with [Qdrant](https://qdrant.tech/) as the vector database for semantic memory storage. Memories are stored in two locations:

#### Vector Database (Qdrant)
- **Location**: `~/.mem0/migrations_qdrant/`
- **Purpose**: Stores high-dimensional vector embeddings for semantic search
- **Technology**: Qdrant vector database with Azure OpenAI embeddings
- **Benefits**: Enables fast similarity search across incident memories

#### Metadata Database (SQLite)
- **Location**: `~/.mem0/history.db`
- **Purpose**: Stores conversation history, metadata, and full memory content
- **Technology**: SQLite database for structured data storage

#### Configuration
- **Location**: `~/.mem0/config.json`
- **Purpose**: mem0 configuration settings

#### Legacy File-Based Storage (Fallback)
When mem0 is not available, memories fall back to JSON files in the `memory/` directory:
- `memory/memory_summarizer_user.json` - Default user memories
- `memory/memory_[user_id].json` - User-specific memories
- `memory/memory_config.json` - Memory configuration

Each memory entry contains:
- Incident number and timestamp
- Incident type, severity, and description
- Processing summary and key findings
- Technical details and recommendations

### Memory Configuration

Configure memory behavior using `memory_config.json`:

```json
{
  "memory_integration": {
    "enabled": true,
    "priority": "complementary",
    "max_memory_context_length": 1000,
    "memory_search_limit": 3
  }
}
```

**Key Settings**:
- `enabled`: Enable/disable memory integration
- `max_memory_context_length`: Maximum length of memory context
- `memory_search_limit`: Number of relevant memories to include

### Example Output

**Original Prompt**:
```
Please analyze this incident and provide a comprehensive summary.
```

**Enhanced Prompt (with Memory)**:
```
Please analyze this incident and provide a comprehensive summary.

Context from previous similar incidents:
Previous similar incidents:
1. Incident INC-001: Security incident resolved by implementing MFA and blocking suspicious IPs.
2. Incident INC-002: Performance issue caused by inefficient queries, resolved by adding indexes.

Use this context to provide more informed and consistent analysis.
```

### Memory File Management

- **Automatic Creation**: Memory directory is created automatically when needed
- **Git Ignored**: Memory files won't clutter your repository
- **Organized**: All memory-related files in one place
- **Flexible**: Easy to manage and backup memory data
- **Manual Control**: You can manually inspect, edit, or delete memory files if needed

## Customizing Prompts and Context

### Prompt Templates (`prompts.json`)
- The `prompts.json` file contains the base prompt templates (system and user prompts) for each supported prompt type.
- You can add or edit prompt types here to control the instructions and formatting sent to the LLM.

### Interactive Menu Configuration (`interactive_menu_prompts.json`)
- This file defines which prompts appear in the interactive menu with their associated emojis.
- Add or remove entries to control what users see when no prompt type is specified.
- Format: `{"prompt_type": "📱"}`
- Only prompts listed here will appear in the interactive selection menu.

## Folder Structure

After running the tool, you'll have the following structure:

```
incidents/
├── 100000002/                    # Incident-specific folder
│   ├── 100000002.csv            # Raw CSV data from Kusto
│   ├── 100000002_summary_processed.txt  # Processed summary with screenshot references
│   ├── screenshot_100000002_001.png     # Downloaded screenshots
│   ├── screenshot_100000002_002.jpg
│   └── ...
├── 100000003/
│   ├── 100000003.csv
│   └── ...
└── ...

processed_incidents/
├── 100000002.json               # Structured JSON for LLM processing
├── 100000003.json
└── ...

summaries/
├── 100000002.json               # AI-generated summaries
├── 100000003.json
└── ...

memory/                          # Memory storage (git-ignored)
├── memory_config.json           # Memory configuration
├── memory_summarizer_user.json  # User memories
└── ...
```

## Requirements
- Python 3.8+
- Azure Kusto and OpenAI access
- See `requirements.txt` for all dependencies

## Troubleshooting
- If you are prompted to authenticate every time, ensure `.kusto_token_cache.json` is writable and not deleted between runs.
- If you see OpenAI API key errors, ensure your `.env` is set up for Azure OpenAI (not regular OpenAI).
- For Kusto query issues, check your `query.kql` template and Azure permissions.
- If the interactive prompt menu doesn't appear, ensure `interactive_menu_prompts.json` exists and contains valid prompt types.
- If `transformer.py` can't find CSV files, check that they're in the new incident-specific folders (e.g., `incidents/100000002/100000002.csv`).
- For screenshot download issues, ensure the incident data contains valid data URLs and the output directories are writable.
- **Memory Issues**: If memory isn't working, check that `memory_config.json` exists and `enabled` is set to `true`, or verify that the `--no-memory` flag is not being used.
- **Vector Database Issues**: If you see "Using file-based memory system" instead of mem0, ensure `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` is set in your `.env` file. The vector database is stored in `~/.mem0/` and persists across sessions.
- **Embedding Consistency**: The system uses BAAI/bge-large-en-v1.5 consistently for all embeddings, ensuring 1536 dimensions throughout. No dimension mismatches should occur.
- **Article Search Issues**: If article search fails with float() errors, the system properly handles various embedding formats and converts them to the correct data types. Invalid embeddings are replaced with zero vectors.
- **Vector Database Loading**: The system supports multiple vector database formats including nested embeddings structures. All embeddings are automatically standardized to 1536 dimensions.

## Project Structure

```
AIssist/
├── main.py                     # Main entry point and orchestration
├── processor.py                # AI processing and summarization engine
├── transformer.py              # Data transformation and cleaning
├── kusto_fetcher.py            # Azure Kusto data fetching
├── article_searcher.py         # Article search and vector operations
├── pgvector_store.py            # pgvector-backed retrieval (schema, upsert, cosine search)
├── azure_devops_client.py      # Azure DevOps API client for work items
├── azure_auth.py                # Azure authentication helpers
├── config.py                    # Configuration management
├── free_text_prompt_generator.py # Free-text prompt generation helper
├── fetch_new_pa.py              # Preventative action fetch helper
├── pa_triage_runner.py          # Preventative action triage runner
├── timing_utils.py              # Timing/telemetry utilities
├── requirements.txt             # Python dependencies
├── query.kql                    # Kusto query template (create locally, not tracked)
├── prompts.json                 # AI prompt templates (create locally, not tracked)
├── interactive_menu_prompts.json  # Interactive menu config (create locally, not tracked)
├── incidents/                   # Raw incident data (CSV files, git-ignored)
├── processed_incidents/         # Processed JSON data (git-ignored)
├── summaries/                   # Generated summaries (git-ignored)
├── memory/                      # Memory storage (git-ignored)
└── logs/                        # Application logs (git-ignored)
```

Optional modules referenced elsewhere in this README (a team-knowledge system, an MCP proxy for VS Code, VIP-customer tracking, extended documentation) are not included in this public repository — build them yourself following the same patterns if you need them.

## Recent Improvements

### Knowledge Base Article Generation (Latest)
- **KB Article Creation**: New `kb_article` prompt creates comprehensive runbooks from incident data
- **Runbook Best Practices**: Follows industry standards with structured sections including purpose, scope, prerequisites, and validation steps
- **Incident-Based Content**: Uses only actual troubleshooting steps performed in incidents for accuracy
- **Comprehensive Structure**: Includes title, purpose, scope, prerequisites, symptoms, root cause, troubleshooting procedure, resolution, verification, and prevention sections
- **Expected Outcomes**: Each step includes expected outcomes and validation steps for better execution

### Team Knowledge Management System
- **Team Detection Engine**: Automatically identifies teams involved in incidents
- **Team Expertise Mapping**: Builds knowledge about what each team does and their areas of expertise
- **Team Interaction Analysis**: Tracks ownership changes, acknowledgments, and escalation patterns
- **Team Context Enhancement**: Provides team-specific context to AI prompts
- **Continuous Learning**: Updates team knowledge from new incidents over time

### MCP Integration and Kusto Access
- **MCP Proxy**: HTTP proxy for Azure-authenticated Kusto access through VS Code
- **Real Azure Authentication**: Uses actual Azure AD tokens for secure database access
- **VS Code Integration**: Works seamlessly with GitHub Copilot Chat in VS Code
- **Multi-Cluster Support**: Access to multiple Kusto clusters with proper authentication
- **Token Management**: Automatic token caching and renewal

### VIP Customer Incident Tracking
- **VIP Customer Focus**: Specialized incident fetcher for high-priority customers
- **Flexible Search Methods**: Both customer name and tenant ID search options
- **Comprehensive Reporting**: Detailed breakdowns by customer, status, and severity
- **Automated Data Export**: CSV output with timestamped results
- **Customer Variations**: Handles different name variations and spelling differences

### Enhanced Documentation System
- **Kusto MCP Guide**: Comprehensive guide for Kusto database integration
- **Unified MCP Guide**: Complete documentation for MCP server integration
- **Logs Analysis Protocol**: Standardized approach to log analysis and investigation
- **Quick Reference**: Quick reference guides for common operations

### Gap Analysis Feature
- **Intelligent Gap Analysis**: Compares incident data against troubleshooting articles to identify missing steps
- **Real Content Retrieval**: Accesses actual troubleshooting content from local knowledge base directory
- **Azure OpenAI Integration**: Uses Azure OpenAI for intelligent analysis and gap identification
- **Execution Plans**: Generates prioritized action plans with specific commands and expected outcomes
- **Standalone Scripts**: Independent gap analysis tools (`gap_analysis.py`, `simple_gap_analysis.py`)

### Enhanced Article Search
- **Real Content Access**: Retrieves actual troubleshooting guide content
- **Content Summaries**: Stores meaningful content summaries for fast relevance matching
- **Full Content Paths**: Maintains paths to complete articles for detailed analysis
- **Improved Relevance**: Better semantic search using real content

### Azure OpenAI Integration
- **Response Handling**: Fixed compatibility issues with newer OpenAI client versions
- **Error Handling**: Robust error handling for different API response structures
- **Direct Integration**: Both gap analysis and main workflow use Azure OpenAI directly
- **Proper Authentication**: Uses Azure OpenAI credentials from `.env` file

### Code Quality Improvements
- **Comprehensive Logging**: Detailed logging system with file and console output
- **Error Handling**: Improved error handling throughout the system
- **Documentation**: Updated README with comprehensive project overview
- **Code Structure**: Cleaner, more maintainable code structure

## License
MIT

