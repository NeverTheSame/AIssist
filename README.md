# Summarizer

## Overview

This tool automates the process of fetching incident data from Azure Data Explorer (Kusto), processes it, and generates summaries using Azure OpenAI by default. It supports persistent Azure authentication, so you only need to authenticate once per token lifetime.

## Architecture

The tool follows a three-stage pipeline:

1. **Data Fetching** (`kusto_fetcher.py`): Fetches raw incident data from Azure Kusto and saves it as CSV files in incident-specific folders
2. **Data Processing** (`transformer.py`): Transforms the raw CSV data into structured JSON format optimized for LLM processing
3. **AI Analysis** (`processor.py`): Uses LLMs to generate insights, summaries, and recommendations from the processed data

## Features
- **Automatic CSV Fetching**: Provide incident number(s), and the tool fetches the relevant data from Azure Kusto.
- **Screenshot Extraction**: Automatically downloads and saves embedded screenshots from incident data.
- **Multi-Incident Support**: Process multiple incidents simultaneously and generate unified summaries.
- **Persistent Azure Authentication**: Auth token is cached locally and reused until it expires (no repeated browser logins).
- **Default Azure OpenAI Usage**: Summarization and processing use Azure OpenAI by default.
- **Interactive Prompt Selection**: When no prompt type is specified, the tool presents an interactive menu of available molecular prompt types.
- **Flexible Argument Passing**: Pass summarization and prompt options directly from the command line.
- **Dynamic Molecular Context**: Prompts can be dynamically enhanced with relevant incident examples from a configurable file (`molecular_examples.json`).
- **Memory Integration**: Uses [mem0](https://github.com/mem0ai/mem0) for persistent memory across processing sessions, providing context-aware analysis based on previous incidents.

## Setup

1. **Clone the repository and set up your Python environment**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Configure Azure OpenAI and Kusto access**
- Ensure your Azure credentials are set up for browser-based authentication.
- Create your KQL query template in `query.kql` (see Configuration section below).
- Set up your Azure OpenAI environment variables in `.env` as needed for summarization.

3. **First-time authentication**
- The first time you run the tool, you'll be prompted to authenticate with Azure in your browser. The token will be cached for future runs.

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
| where ICM_IncidentId in ("{incident_number}")
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
      "Use abbreviated names, e.g. MDE, AV, PUA, etc.",
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

**Azure OpenAI (Default):**
```
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
```

**Azure Kusto (Required for data fetching):**
```
AZURE_KUSTO_CLUSTER=https://your-cluster.kusto.windows.net
AZURE_KUSTO_DATABASE=YourDatabase
AZURE_KUSTO_TOKEN_SCOPE=https://your-cluster.kusto.windows.net/.default
```

**OpenAI (Alternative):**
```
OPENAI_API_KEY=your_api_key
OPENAI_MODEL_NAME=gpt-4-turbo-preview
```

**ZAI (Alternative):**
```
ZAI_API_KEY=your_api_key
ZAI_BASE_URL=your_base_url
ZAI_MODEL_NAME=glm-4.5-air
```

## Usage

### Basic Command

```bash
python main.py <incident_number> [options]
```

### Multi-Incident Command

```bash
python main.py <incident_number1> <incident_number2> ... [options]
```

### Options
- `--prompt-type TYPE`   Type of prompt (default, technical, executive, escalation, escalation_molecular, mitigation_molecular, troubleshooting_molecular, etc.)
- `--azure`              Explicitly use Azure OpenAI (default behavior)
- `--zai`                Use ZAI instead of Azure OpenAI
- `--debug_api`          Enable API debugging
- `--summ`               Include summary from summary.txt
- `--summ-docx`          Use summary.docx as input
- `--troubleshooting-plan` Generate troubleshooting plan mode (first incident is primary, others are historical references)

**Note:** If no `--prompt-type` is specified, the tool will display an interactive menu showing only molecular prompt types for selection.

### Example Usage

#### Complete Workflow (Recommended)
The `main.py` script handles the entire pipeline automatically:

Fetch, process, and summarize an incident with interactive prompt selection:
```bash
python main.py 654045297
```

Fetch, process, and summarize an incident with specific escalation prompt:
```bash
python main.py 654045297 --prompt-type escalation
```

Fetch, process, and summarize with molecular context (dynamic examples):
```bash
python main.py 654045297 --prompt-type escalation_molecular
```

Use ZAI instead of Azure OpenAI:
```bash
python main.py 654045297 --zai
```

Process multiple incidents with unified summarization:
```bash
python main.py 654045297 654045298 654045299 --prompt-type escalation_molecular
```

Generate troubleshooting plan based on historical incidents:
```bash
python main.py 654045297 654045298 654045299 654045300 --troubleshooting-plan
```

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
- **Embedding Model**: Azure OpenAI text-embedding-ada-002 (1536 dimensions)
- **Distance Metric**: Cosine similarity for semantic matching
- **Storage Type**: File-based persistent storage (not in-memory)
- **Collection**: Single collection named "mem0" for all memories

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

## License
MIT
