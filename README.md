# AI-Powered Incident Summarizer

## Overview

The AI-Powered Incident Summarizer is an advanced tool designed to automate the analysis and summarization of technical support incidents. It streamlines the incident management workflow by fetching data from databases (Azure Data Explorer/Kusto), processing it through intelligent AI models, and generating comprehensive summaries, troubleshooting guides, and actionable insights.

This tool is built for support teams to handle technical incidents more efficiently, providing consistent, high-quality summaries and reducing manual analysis time.

## What This Project Does

### Core Functionality
- **Automated Incident Analysis**: Fetches incident data from databases and processes it through AI models
- **Intelligent Summarization**: Generates various types of summaries including escalation notes, mitigation reports, and troubleshooting guides
- **Context-Aware Processing**: Uses molecular context engineering and memory systems to provide more relevant and consistent analysis
- **Article Search Integration**: Finds relevant troubleshooting articles and performs gap analysis against incident data
- **Multi-Incident Support**: Processes single incidents or combines multiple related incidents for unified analysis

### Key Features
- **Persistent Authentication**: Caches Azure authentication tokens to avoid repeated logins
- **Screenshot Processing**: Automatically downloads and processes embedded screenshots from incident data
- **Memory Integration**: Uses mem0 for persistent memory across sessions, learning from previous incidents
- **Molecular Context**: Dynamically selects relevant examples to enhance prompt quality
- **Gap Analysis**: Compares incident troubleshooting against knowledge base articles to identify missing steps
- **Azure Router Integration**: Uses Azure Router with GPT-5 for all AI operations

## Architecture

The tool follows a sophisticated three-stage pipeline with advanced AI integration:

### Stage 1: Data Fetching (`kusto_fetcher.py`)
- **Purpose**: Retrieves raw incident data from Azure Data Explorer (Kusto)
- **Features**: 
  - Persistent token caching for seamless authentication
  - Screenshot extraction and processing from embedded data URLs
  - Incident-specific folder organization
  - Network error handling with VPN connection guidance
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
  - Molecular context engineering for enhanced prompts
  - Memory integration for learning from previous incidents
  - Article search and gap analysis capabilities
  - Multiple prompt types for different use cases
  - Cost tracking and token management
- **Output**: Comprehensive summaries, troubleshooting guides, and actionable insights

## Technologies Used

### Core Technologies
- **Python 3.12**: Primary programming language with virtual environment support
- **Azure Data Explorer (Kusto)**: Data source for incident information
- **Azure Router**: Primary AI service for text generation and embeddings (GPT-5)

### AI and Machine Learning
- **mem0**: Universal memory layer for AI agents providing persistent context
- **Local Embeddings**: all-MiniLM-L6-v2 model for consistent semantic search
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

## Key Features

### Data Management
- **Automatic CSV Fetching**: Retrieves incident data from databases with persistent authentication
- **Screenshot Processing**: Downloads and processes embedded screenshots from incident data
- **Multi-Incident Support**: Handles single incidents or combines multiple related incidents
- **Data Sanitization**: Cleans HTML content and removes sensitive information

### AI-Powered Analysis
- **Intelligent Summarization**: Generates various types of summaries (escalation, mitigation, troubleshooting)
- **Molecular Context Engineering**: Dynamically enhances prompts with relevant examples
- **Memory Integration**: Learns from previous incidents to provide better context
- **Article Search**: Finds relevant troubleshooting articles using semantic search
- **Gap Analysis**: Identifies missing troubleshooting steps by comparing against knowledge base

### User Experience
- **Interactive Prompt Selection**: Presents menu of available prompt types
- **Flexible Configuration**: Supports multiple AI providers and custom settings
- **Cost Tracking**: Monitors token usage and API costs
- **Comprehensive Logging**: Detailed logs for debugging and monitoring

## How It Works

### Complete Workflow
1. **Incident Data Retrieval**: Fetches incident data from databases using KQL queries
2. **Data Processing**: Converts raw CSV data to structured JSON format
3. **AI Analysis**: Processes data through AI models with context enhancement
4. **Memory Storage**: Stores results for future context and learning
5. **Output Generation**: Creates summaries, troubleshooting guides, and insights

### Molecular Context Engineering
The system uses a sophisticated molecular context engine that:
- Extracts technical keywords from incident data
- Matches against a database of curated examples
- Dynamically enhances prompts with relevant context
- Improves consistency and quality of AI responses

### Memory System
- **Persistent Learning**: Stores processed incidents for future reference
- **Semantic Search**: Finds relevant previous incidents using vector embeddings
- **Context Enhancement**: Automatically adds relevant historical context to prompts
- **Cross-Session Persistence**: Memory survives across different processing sessions

## Setup

### Prerequisites
- Python 3.12 or higher
- Azure account with access to databases and AI services
- VPN connection for accessing internal resources (if required)

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

### Molecular Prompt Types (Recommended)
- **`escalation_molecular`**: Creates escalation summaries with dynamic examples
- **`mitigation_molecular`**: Generates mitigation reports with contextual guidance
- **`troubleshooting_molecular`**: Produces detailed troubleshooting guides
- **`article_search_molecular`**: Finds relevant troubleshooting articles
- **`troubleshooting_plan_molecular`**: Creates comprehensive troubleshooting plans
- **`wait_time_molecular`**: Analyzes incident wait times by team
- **`prev_act_molecular`**: Recommends preventative actions
- **`weekly_insights_molecular`**: Generates weekly status updates

### Standard Prompt Types
- **`escalation_plain`**: Basic escalation summaries
- **`technical_article_plain`**: Technical documentation generation
- **`sentiment_analysis_plain`**: Customer sentiment analysis
- **`human_style_rewriting_plain`**: Human-like text rewriting

### Specialized Analysis
- **`troubleshooting_gap_analysis`**: Compares incident steps against knowledge base
- **`care_incident_facilitation`**: CARE team facilitation summaries
- **`icm_delay_analysis`**: Identifies reasons for incident delays

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
  "default": {
    "system_prompt": "You are a helpful assistant that summarizes incident discussions. Focus on key points, decisions, and action items.",
    "user_prompt": "Please provide a comprehensive summary of this incident discussion, highlighting the main points, any decisions made, and action items."
  },
  "escalation": {
    "system_prompt": "You are an expert at preparing technical incident escalations. Your role is to create clear, concise, and technically accurate summaries of unresolved incidents to hand off to engineering or specialized teams.",
    "user_prompt": "Summarize this incident for escalation to another team using exactly four paragraphs: 1) a short issue description, 2) relevant details including environmental context, 3) troubleshooting steps already taken, and 4) the customer's current goal or what support is needed to proceed."
  },
  "escalation_molecular": {
    "system_prompt": "You are an expert at preparing technical incident escalations. Your role is to create clear, concise, and technically accurate summaries of unresolved incidents to hand off to engineering or specialized teams.",
    "user_prompt": "Summarize this incident for escalation to another team using exactly four paragraphs: 1) a short issue description, 2) relevant details including environmental context, 3) troubleshooting steps already taken, and 4) the customer's current goal or what support is needed to proceed.\n\nHere are examples of well-structured escalation summaries:\n\n[Examples will be dynamically injected from molecular_examples.json]",
    "additional_guidelines": [
      "Use abbreviated names, e.g. AV, PUA, etc.",
      "Do not start with 'The issue involves' or 'The incident involves'. This is implicit.",
      "Do not say 'The environment involves' or 'The environment consists of'. This is implicit."
    ]
  }
}
```

**Note:** Molecular prompt types (ending with `_molecular`) will have examples dynamically injected from your `molecular_examples.json` file when used.

#### 3. Molecular Examples (`molecular_examples.json`)

Create a `molecular_examples.json` file with example incidents for dynamic context injection:

```json
{
  "escalation_molecular": [
    {
      "keywords": ["agent", "crash", "endpoint", "protection"],
      "severity": "high",
      "category": "agent_issues",
      "example": {
        "input": "Agent crashes on system, telemetry shows errors, customer has multiple endpoints affected",
        "output": "The customer is experiencing agent crashes affecting multiple endpoints in their environment. The crashes occur during protection scans and result in complete agent termination.\n\nThe issue is isolated to specific systems running agent version X.X.X. The customer environment consists of managed devices in a corporate setting with standard security policies. Telemetry data shows consistent error codes preceding each crash event.\n\nOur team performed agent log analysis, reviewed crash dumps, attempted agent reinstallation on affected systems, and verified system compatibility. We also checked for conflicting software and reviewed recent policy changes. The crash pattern persists across multiple reinstallation attempts.\n\nThe customer needs immediate resolution to restore endpoint protection on their fleet. They require either a hotfix, workaround, or rollback guidance to maintain security coverage while a permanent solution is developed."
      }
    }
  ],
  "mitigation_molecular": [
    {
      "keywords": ["performance", "cpu", "memory", "optimization"],
      "severity": "medium",
      "category": "performance_issues",
      "example": {
        "input": "High resource usage by security agent, affecting system performance",
        "output": "Performance optimization was achieved through configuration adjustments and policy tuning. Resource usage returned to normal levels after implementing recommended settings."
      }
    }
  ]
}
```

#### 4. Technical Patterns (`technical_patterns.json`) - Optional

Create a `technical_patterns.json` file to customize keyword extraction for incident classification:

```json
{
  "technical_patterns": [
    "\\b(agent|service|process)\\b",
    "\\b(crash|error|failure|timeout)\\b",
    "\\b(memory|cpu|performance)\\b",
    "\\b(network|connectivity|firewall)\\b",
    "\\b(sync|synchronization)\\b",
    "\\b(authentication|auth)\\b",
    "\\b(macos|windows|linux)\\b",
    "\\b(policy|configuration)\\b",
    "\\b(telemetry|reporting)\\b",
    "\\b(installation|deployment)\\b",
    "\\b(detection|engine)\\b",
    "\\b(security|protection)\\b"
  ],
  "description": "Technical keyword patterns used for incident classification and molecular context selection.",
  "usage": "Add or modify patterns to improve keyword extraction for your specific domain."
}
```

**Note:** This file is optional. If not provided, the tool will use default patterns. Customize these patterns to better match your incident types and technical terminology.

### Environment Variables (.env)

Create a `.env` file with the following variables:

**Azure Router (Required):**
```
AZURE_ROUTER_API_KEY=your_api_key
AZURE_ROUTER_ENDPOINT=your_endpoint
AZURE_ROUTER_API_VERSION=2024-02-15-preview
AZURE_ROUTER_DEPLOYMENT_NAME=your_deployment_name
AZURE_ROUTER_MODEL_NAME=gpt-5
```

**Azure Kusto (Required for data fetching):**
```
AZURE_KUSTO_CLUSTER=https://your-cluster.kusto.windows.net
AZURE_KUSTO_DATABASE=YourDatabase
AZURE_KUSTO_TOKEN_SCOPE=https://your-cluster.kusto.windows.net/.default
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
python main.py 654045297
```

Fetch, process, and summarize an incident with specific escalation prompt:
```bash
python main.py 654045297 --prompt-type escalation_molecular
```

All operations use Azure Router (GPT-5) by default.

Process multiple incidents with unified summarization:
```bash
python main.py 654045297 654045298 654045299 --prompt-type escalation_molecular
```

Generate troubleshooting plan based on historical incidents:
```bash
python main.py 654045297 654045298 654045299 654045300 --troubleshooting-plan
```

## How to Test

### Testing the Application

**Note:** This project currently doesn't include a `test_main.py` file or automated unit tests. Testing is done through manual execution with sample incident data:

#### 1. Test with Sample Incident
```bash
# Test with a known incident number
python3 main.py 692076095 --prompt-type escalation_molecular
```

#### 2. Test Different Prompt Types
```bash
# Test escalation summaries
python3 main.py 692076095 --prompt-type escalation_molecular

# Test mitigation reports  
python3 main.py 692076095 --prompt-type mitigation_molecular

# Test troubleshooting guides
python3 main.py 692076095 --prompt-type troubleshooting_molecular
```

#### 3. Test Article Search Functionality
```bash
# Test article search mode (requires vector database)
python3 main.py 692076095 --prompt-type article_search_molecular
```

#### 4. Test Memory Integration
```bash
# Process multiple incidents to test memory learning
python3 main.py 692076095
python3 main.py 692076095  # Second run should use memory context
```

#### 5. Test Different Prompt Types
```bash
# Test different prompt types
python3 main.py 692076095 --prompt-type escalation_molecular
python3 main.py 692076095 --prompt-type mitigation_molecular
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
python3 main.py 692076095 --debug

# Test with timing analysis
python3 main.py 692076095 --timing

# All operations use Azure Router (GPT-5)
python3 main.py 692076095
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
python3 main.py <incident_number> --prompt-type article_search_molecular --vector-db-path article_vector_db.json

# Search using text files directly
python3 main.py <incident_number> --prompt-type article_search_molecular --articles-path /path/to/articles

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
- `--prompt-type TYPE`   Type of prompt (default, technical, executive, escalation, escalation_molecular, mitigation_molecular, troubleshooting_molecular, article_search_molecular, etc.)
- All operations use Azure Router (GPT-5) by default
- `--debug`              Enable API debugging
- `--articles-path PATH` Path to directory containing troubleshooting articles (for article search mode)
- `--vector-db-path PATH` Path to vector database file (for article search mode)
- `--summ`               Include summary from summary.txt
- `--summ-docx`          Use summary.docx as input
- `--troubleshooting-plan` Generate troubleshooting plan mode (first incident is primary, others are historical references)
- `--timing`             Enable detailed timing analysis and reporting

**Note:** If no `--prompt-type` is specified, the tool will display an interactive menu showing only molecular prompt types for selection.

#### Manual Stage-by-Stage Processing
You can also run each stage manually:

**Stage 1: Fetch data from Kusto**
```bash
python kusto_fetcher.py 654045297
```

**Stage 2: Process CSV to JSON**
```bash
python transformer.py icms/654045297/654045297.csv
```

**Stage 3: Generate AI insights**
```bash
python processor.py processed_incidents/654045297.json --prompt-type escalation_molecular
```

### How It Works

The tool follows a three-stage pipeline:

#### Stage 1: Data Fetching (`kusto_fetcher.py`)
1. **Fetches incident data from Azure Kusto** using the incident number(s) and your `query.kql` template.
2. **Caches the Azure authentication token** in `.kusto_token_cache.json` for reuse until it expires.
3. **Creates incident-specific folders** (e.g., `icms/664099798/`) for each incident.
4. **Downloads embedded screenshots** from data URLs and saves them as image files.
5. **Saves raw data** as CSV files in the incident folders.

#### Stage 2: Data Processing (`transformer.py`)
1. **Reads the raw CSV data** from incident-specific folders.
2. **Cleans and structures the data** by removing HTML formatting and filtering unwanted entries.
3. **Extracts authored summaries** and processes them for LLM consumption.
4. **Outputs structured JSON** files optimized for LLM processing.

#### Stage 3: AI Analysis (`processor.py`)
1. **For single incidents**: Processes and summarizes the incident using Azure OpenAI (or ZAI if specified).
2. **For multiple incidents**: Combines all incident data and generates a unified summary.
3. **For troubleshooting plan mode**: Analyzes the first incident as the primary issue and uses other incidents as historical references to generate a comprehensive troubleshooting plan.
4. **Outputs results** to the appropriate directories (`processed_incidents/`, `summaries/`).
5. **If a `_molecular` prompt type is used**, the tool dynamically selects relevant examples from `molecular_examples.json` and injects them into the prompt for improved context and summary quality.

### Troubleshooting Plan Mode

The `--troubleshooting-plan` mode is designed to help create comprehensive troubleshooting plans for unresolved incidents by analyzing patterns from previously resolved similar incidents.

**How it works:**
1. **Primary Incident**: The first incident number provided is treated as the current unresolved issue that needs a troubleshooting plan.
2. **Historical References**: All subsequent incident numbers are treated as resolved incidents that contain valuable insights and successful resolution strategies.
3. **Pattern Analysis**: The AI analyzes the historical incidents to identify common root causes, successful troubleshooting approaches, and resolution strategies.
4. **Plan Generation**: Based on the patterns found, it creates a structured, step-by-step troubleshooting plan specifically tailored to the primary incident.

**Example workflow:**
```bash
# Generate troubleshooting plan for incident 1111 using insights from resolved incidents 22222, 33333, 4444
python main.py 1111 22222 33333 4444 --troubleshooting-plan
```

**Output includes:**
- Primary incident analysis
- Pattern analysis from historical incidents
- Step-by-step troubleshooting plan
- Risk assessment for each step
- Success criteria and escalation points

## Memory Integration

The summarizer includes memory integration using [mem0](https://github.com/mem0ai/mem0), a universal memory layer for AI agents. This enables the tool to learn from previous incidents and provide more context-aware analysis.

### How Memory Works

- **Persistent Memory**: Stores information about processed incidents, including summaries, key findings, and technical details
- **Context Enhancement**: Searches for relevant previous incidents and enhances prompts with this context
- **Improved Consistency**: Maintains consistency in analysis and recommendations across similar incidents
- **Learning Over Time**: Becomes more effective at identifying patterns as more incidents are processed
- **Cross-Session Persistence**: Memory persists across different processing sessions

### Memory vs Molecular Examples

**Molecular Examples** (`molecular_examples.json`):
- Curated, high-quality examples you manually select
- Immediate availability and consistent quality
- Organized by prompt type with keywords and categories
- Best for: Standard scenarios and quality control

**Mem0 Memory**:
- Automatically learns from every processed incident
- Builds up over time with real processing results
- Provides dynamic, contextual relevance
- Best for: Learning from actual outcomes and patterns

**Integration**: When both are used together, molecular examples provide the foundation, and memory adds real-world context from your actual processing history.

### Vector Database Architecture

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
python processor.py incident.json --prompt-type escalation_molecular
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
    "memory_search_limit": 3,
    "enable_memory_for_molecular": true
  },
  "molecular_integration": {
    "preserve_molecular_examples": true,
    "molecular_examples_priority": "high",
    "allow_memory_enhancement": true
  }
}
```

**Key Settings**:
- `enabled`: Enable/disable memory integration
- `enable_memory_for_molecular`: Whether to use memory with molecular prompts
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

**Enhanced Prompt (with Molecular + Memory)**:
```
[Your molecular examples here]

Additional context from previous similar incidents:
Previous similar incidents:
1. Incident INC-001: Security incident resolved by implementing MFA and blocking suspicious IPs.

Use this additional context alongside the provided examples for more informed analysis.
```

### Memory File Management

- **Automatic Creation**: Memory directory is created automatically when needed
- **Git Ignored**: Memory files won't clutter your repository
- **Organized**: All memory-related files in one place
- **Flexible**: Easy to manage and backup memory data
- **Manual Control**: You can manually inspect, edit, or delete memory files if needed

## Customizing Prompts and Molecular Context

### Prompt Templates (`prompts.json`)
- The `prompts.json` file contains the base prompt templates (system and user prompts) for each supported prompt type.
- You can add or edit prompt types here to control the instructions and formatting sent to the LLM.
- Example prompt types: `default`, `escalation`, `mitigation`, `escalation_molecular`, etc.

### Molecular Examples (`molecular_examples.json`)
- The `molecular_examples.json` file contains example incidents (input/output pairs, keywords, categories) used for dynamic context injection.
- When a prompt type ending with `_molecular` is used, the tool selects the most relevant examples from this file and appends them to the prompt.
- **To add or edit examples:**
  1. Open `molecular_examples.json` in your editor.
  2. Add new examples under the appropriate key (e.g., `escalation_molecular`, `mitigation_molecular`).
  3. Each example should include `keywords`, `severity`, `category`, and an `example` object with `input` and `output` fields.
- This allows the summarizer to adapt its context to the specifics of each incident, improving the quality and relevance of generated summaries.

## Folder Structure

After running the tool, you'll have the following structure:

```
icms/
├── 664099798/                    # Incident-specific folder
│   ├── 664099798.csv            # Raw CSV data from Kusto
│   ├── 664099798_summary_processed.txt  # Processed summary with screenshot references
│   ├── screenshot_664099798_001.png     # Downloaded screenshots
│   ├── screenshot_664099798_002.jpg
│   └── ...
├── 672325151/
│   ├── 672325151.csv
│   └── ...
└── ...

processed_incidents/
├── 664099798.json               # Structured JSON for LLM processing
├── 672325151.json
└── ...

summaries/
├── 664099798.json               # AI-generated summaries
├── 672325151.json
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
- If you encounter errors about missing or malformed `molecular_examples.json`, ensure the file exists and is valid JSON.
- If the interactive prompt menu doesn't appear, ensure `prompts.json` contains molecular prompt types (ending with `_molecular`).
- If `transformer.py` can't find CSV files, check that they're in the new incident-specific folders (e.g., `icms/664099798/664099798.csv`).
- For screenshot download issues, ensure the incident data contains valid data URLs and the output directories are writable.
- **Memory Issues**: If memory isn't working, check that `memory_config.json` exists and `enabled` is set to `true`, or verify that the `--no-memory` flag is not being used.
- **Vector Database Issues**: If you see "Using file-based memory system" instead of mem0, ensure `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` is set in your `.env` file. The vector database is stored in `~/.mem0/` and persists across sessions.
- **Embedding Consistency**: The system uses BAAI/bge-large-en-v1.5 consistently for all embeddings, ensuring 1536 dimensions throughout. No dimension mismatches should occur.
- **Article Search Issues**: If article search fails with float() errors, the system properly handles various embedding formats and converts them to the correct data types. Invalid embeddings are replaced with zero vectors.
- **Vector Database Loading**: The system supports multiple vector database formats including nested embeddings structures. All embeddings are automatically standardized to 1536 dimensions.

## Project Structure

```
Summarizer/
├── main.py                     # Main entry point and orchestration
├── processor.py                # AI processing and summarization engine
├── transformer.py              # Data transformation and cleaning
├── kusto_fetcher.py           # Azure Kusto data fetching
├── memory/                    # Memory management system
│   ├── memory_manager.py      # Memory integration with Qdrant
│   ├── view_memories.py       # Memory viewing utilities
│   └── examples/              # Memory usage examples
├── article_searcher.py        # Article search and vector operations
├── gap_analysis.py            # Advanced gap analysis functionality
├── simple_gap_analysis.py     # Simplified gap analysis tool
├── config.py                  # Configuration management
├── prompts.json               # AI prompt templates
├── molecular_examples.json    # Dynamic context examples
├── technical_patterns.json    # Keyword extraction patterns
├── query.kql                  # Kusto query template
├── requirements.txt           # Python dependencies
├── AGENTS.md                  # Agent guidelines and rules
├── icms/                      # Raw incident data (CSV files)
├── processed_incidents/       # Processed JSON data
├── summaries/                 # Generated summaries
├── memory/                    # Memory storage files
└── logs/                      # Application logs
```

## Recent Improvements

### Gap Analysis Feature (Latest)
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

