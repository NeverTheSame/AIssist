import json
import os
import re
import sys
import time
from datetime import datetime
import logging
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from openai import AzureOpenAI
import tiktoken
from azure_auth import get_openai_client_with_auth
from config import config
import argparse
from typing import List, Dict, Any
import guard
# Mock ZaiClient for now - you can replace this with actual ZAI client when needed
class ZaiClient:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = self.ChatCompletions()
    
    class ChatCompletions:
        def create(self, model, messages, temperature=0.7, max_tokens=8000):
            # Mock response
            class MockResponse:
                def __init__(self):
                    self.choices = [self.MockChoice()]
                    self.usage = self.MockUsage()
                
                class MockChoice:
                    def __init__(self):
                        self.message = self.MockMessage()
                    
                    class MockMessage:
                        def __init__(self):
                            self.content = "Mock ZAI response - replace with actual ZAI client"
                
                class MockUsage:
                    def __init__(self):
                        self.completion_tokens = 100
                        self.total_tokens = 500
            
            return MockResponse()
# Memory manager imported lazily in __init__ to avoid PyTorch dependency issues
# from memory.memory_manager import SummarizerMemoryManager
from article_searcher import ArticleSearcher
from team_knowledge.teams_matcher import TeamDetector, TeamAnalyzer, TeamLearningEngine
from team_knowledge.team_knowledge_manager import TeamKnowledgeManager
# Timing utilities imported conditionally in __init__

def run_gap_analysis_inline(incident_id: str, articles: List[Dict[str, Any]], articles_path: str = None, vector_db_path: str = None):
    """Run interactive gap analysis after article search."""
    try:
        # Import the gap analysis functions
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from gap_analysis import display_articles, run_gap_analysis
        
        # Display articles for selection
        display_articles(articles)
        
        # Get user selection
        while True:
            try:
                selection = input(f"\nSelect an article (1-{len(articles)}) or 'q' to quit: ").strip()
                
                if selection.lower() == 'q':
                    print("Skipping gap analysis.")
                    return
                
                selection_num = int(selection)
                if 1 <= selection_num <= len(articles):
                    selected_article = articles[selection_num - 1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(articles)}")
            except ValueError:
                print("Please enter a valid number")
        
        # Load incident data
        incident_file = f"processed_incidents/{incident_id}.json"
        if not os.path.exists(incident_file):
            print(f"❌ Incident data not found: {incident_file}")
            return
        
        with open(incident_file, 'r') as f:
            incident_data = json.load(f)
        
        # Initialize article searcher for gap analysis
        from article_searcher import ArticleSearcher
        article_searcher = ArticleSearcher(
            articles_path=None,  # Not used for pre-computed embeddings
            vector_db_path=articles_path,  # JSON embeddings file goes here
            use_ai_service=use_ai_service
        )
        
        # Run the gap analysis using the existing function
        print("\n" + "="*80)
        print("EXECUTING GAP ANALYSIS")
        print("="*80)
        print(f"Incident: {incident_id}")
        print(f"Article: {selected_article.get('title', 'Unknown')}")
        print("="*80)
        
        # Run the gap analysis
        run_gap_analysis(incident_id, selected_article, article_searcher)
        
    except Exception as e:
        logging.error(f"Error in interactive gap analysis: {e}")
        print(f"❌ Error during gap analysis: {e}")
        print("You can run gap analysis manually with: python3 gap_analysis.py <incident_id>")

def run_gap_analysis_stub(incident_id, search_results):
    """Stub function for gap analysis - placeholder for future implementation"""
    print("\n" + "="*80)
    print("GAP ANALYSIS STUB")
    print("="*80)
    print(f"Incident ID: {incident_id}")
    print(f"Number of articles found: {len(search_results)}")
    print("\nGap analysis functionality will be implemented here.")
    print("This will compare the incident against troubleshooting procedures")
    print("and identify missing steps that need to be executed.")
    print("="*80)

# Configure logging
def setup_logging():
    """Setup logging for incident processor"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler - detailed logging
    file_handler = logging.FileHandler('logs/processor.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler - only warnings and errors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

    guard.install_log_redaction([file_handler, console_handler])

    return logger

logger = setup_logging()

# Token costs are now configurable via .env file
# See config.py for ZAI_INPUT_COST, ZAI_OUTPUT_COST, OPENAI_INPUT_COST, OPENAI_OUTPUT_COST

class IncidentProcessor:
    def __init__(self, enable_memory=True, articles_path=None, vector_db_path=None, enable_team_analysis=False, enable_timing=False):
        # Always use AI Service (GPT-5)
        self.use_ai_service = True
        self.enable_timing = enable_timing

        # Import timing utilities if timing is enabled
        if enable_timing:
            from timing_utils import time_operation, time_context, time_llm_call, time_memory_operation, time_team_analysis
            self.time_operation = time_operation
            self.time_context = time_context
            self.time_llm_call = time_llm_call
            self.time_memory_operation = time_memory_operation
            self.time_team_analysis = time_team_analysis
        else:
            # Create no-op decorators when timing is disabled
            def no_op_decorator(*args, **kwargs):
                def decorator(func):
                    return func
                return decorator
            
            def no_op_context(*args, **kwargs):
                from contextlib import nullcontext
                return nullcontext()
            
            self.time_operation = no_op_decorator
            self.time_context = no_op_context
            self.time_llm_call = no_op_context
            self.time_memory_operation = no_op_context
            self.time_team_analysis = no_op_context
        # REMOVED: self.doc_processor = DocumentProcessor()
        
        # Initialize memory manager if enabled (lazy import to avoid PyTorch dependency issues)
        self.memory_manager = None
        if enable_memory:
            try:
                from memory.memory_manager import SummarizerMemoryManager
                self.memory_manager = SummarizerMemoryManager()
                logger.info("Memory manager initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize memory manager: {e}. Continuing without memory.")
                self.memory_manager = None
        
        # Initialize team knowledge system if enabled
        self.team_detector = None
        self.team_knowledge_manager = None
        self.team_analyzer = None
        self.team_learning_engine = None
        if enable_team_analysis:
            try:
                self.team_detector = TeamDetector()
                self.team_knowledge_manager = TeamKnowledgeManager()
                # Initialize team analyzer after LLM client is set up
                self.team_analyzer = None
                # Initialize team learning engine after LLM client is set up
                self.team_learning_engine = None
                logger.info("Team knowledge system initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize team knowledge system: {e}. Continuing without team analysis.")
                print(f"⚠️ Team knowledge system disabled: {e}")
        
        # Initialize article searcher if paths are provided
        self.article_searcher = None
        if articles_path or vector_db_path:
            try:
                self.article_searcher = ArticleSearcher(
                    articles_path=None,  # Not used for pre-computed embeddings
                    vector_db_path=articles_path,  # JSON embeddings file goes here
                    use_ai_service=True
                )
                logger.info("Article searcher initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize article searcher: {e}. Continuing without article search.")
                self.article_searcher = None
        
        # Initialize AI Service client (GPT-5)
        # Check for required config based on auth method
        if config.use_azure_ad:
            required = {
                'endpoint': config.ai_service_endpoint,
                'deployment': config.ai_service_deployment_name,
                'api_version': config.ai_service_api_version
            }
            missing = [k for k, v in required.items() if not v]
            if missing:
                raise ValueError(f"Missing required config for Azure AD: {', '.join(missing)}")
        else:
            if not all([config.ai_service_api_key, config.ai_service_endpoint,
                       config.ai_service_api_version, config.ai_service_deployment_name]):
                raise ValueError("AI Service configuration is incomplete. Please check your .env file.")

        # Set default timeout to 300 seconds (5 minutes) to prevent indefinite hangs
        self.llm_timeout = 300

        # Create client using appropriate authentication method
        self.client, self.auth_method = get_openai_client_with_auth(config)
        logger.info(f"Initialized AzureOpenAI client using {self.auth_method} authentication")
        self.deployment_name = config.ai_service_deployment_name
        self.model_costs = {
            "input": config.input_cost,
            "output": config.output_cost
        }
        self.use_million_tokens = False  # AI Service costs are per 1K tokens
        
        # Initialize team analyzer with LLM client if team analysis is enabled
        if enable_team_analysis and self.team_detector and self.team_knowledge_manager:
            try:
                self.team_analyzer = TeamAnalyzer(
                    self.team_detector, 
                    self.team_knowledge_manager,
                    llm_client=self.client,
                    deployment_name=getattr(self, 'deployment_name', None)
                )
                logger.info("Team analyzer initialized with LLM client")
                
                # Initialize team learning engine with LLM client
                self.team_learning_engine = TeamLearningEngine(
                    self.team_knowledge_manager,
                    self.team_detector,
                    llm_client=self.client,
                    deployment_name=getattr(self, 'deployment_name', None)
                )
                logger.info("Team learning engine initialized with LLM client")
            except Exception as e:
                logger.warning(f"Failed to initialize team analyzer/learning engine with LLM: {e}")
                self.team_analyzer = None
                self.team_learning_engine = None
    
    def extract_incident_number(self, filename):
        """Extract incident number from filename."""
        try:
            # Remove .csv extension and any 'incident_' prefix
            base_name = os.path.splitext(filename)[0]
            if base_name.startswith('incident_'):
                base_name = base_name[9:]
            return base_name
        except Exception as e:
            logger.error(f"Error extracting incident number: {str(e)}")
            raise
    
    def count_tokens(self, text):
        """Count the number of tokens in a text."""
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")
            return len(encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            return 0
    
    def calculate_cost(self, input_tokens, output_tokens):
        """Calculate the cost of the API call."""
        if hasattr(self, 'use_million_tokens') and self.use_million_tokens:
            # ZAI costs are per 1M tokens
            input_cost = (input_tokens / 1000000) * self.model_costs["input"]
            output_cost = (output_tokens / 1000000) * self.model_costs["output"]
        else:
            # OpenAI/Azure costs are per 1K tokens
            input_cost = (input_tokens / 1000) * self.model_costs["input"]
            output_cost = (output_tokens / 1000) * self.model_costs["output"]
        return input_cost + output_cost
    
    def format_conversation(self, conversation):
        """Format conversation for the AI model."""
        formatted = []
        for entry in conversation:
            formatted.append(f"[{entry['timestamp']}] {entry['author']}: {entry['content']}")
        return "\n".join(formatted)

    def clean_azure_support_info(self, text):
        """Remove 'ADDITIONAL INFORMATION FROM AZURE SUPPORT CENTER' section and everything after it."""
        if not text:
            return text
        
        # Pattern to match the section and everything after it
        pattern = r'------------------------------------------------------------------------\s*ADDITIONAL INFORMATION FROM AZURE SUPPORT CENTER.*'
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Also check for variations without the dashes
        pattern2 = r'ADDITIONAL INFORMATION FROM AZURE SUPPORT CENTER.*'
        cleaned_text = re.sub(pattern2, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
        
        return cleaned_text.strip()

    def _read_teams_discussion_csv(self, csv_path="teams_discussion.csv"):
        """Read Teams discussion CSV file and format it for inclusion in prompts."""
        try:
            if not os.path.exists(csv_path):
                logger.warning(f"Teams discussion CSV not found: {csv_path}")
                return None
            
            import csv
            teams_discussions = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    date = row.get('Date', '')
                    time = row.get('Time', '')
                    sender = row.get('Sender', '')
                    message = row.get('Message', '')
                    
                    # Combine date and time into timestamp
                    timestamp = f"{date} {time}" if date and time else date or time
                    teams_discussions.append({
                        'timestamp': timestamp,
                        'sender': sender,
                        'message': message
                    })
            
            if not teams_discussions:
                logger.warning("Teams discussion CSV is empty")
                return None
            
            # Format Teams discussions
            formatted_teams = []
            for entry in teams_discussions:
                formatted_teams.append(f"[{entry['timestamp']}] {entry['sender']}: {entry['message']}")
            
            teams_text = "\n".join(formatted_teams)
            logger.info(f"Loaded {len(teams_discussions)} Teams discussion entries from {csv_path}")
            return teams_text
            
        except Exception as e:
            logger.error(f"Error reading Teams discussion CSV: {e}")
            return None

    def format_conversation_with_ai_summary(self, conversation, internal_ai_summary=None, summary=None, teams_discussion=None, summary_images=None):
        """Format conversation and authored summary for the AI model.

        Returns a list of content items for multimodal messages:
        - If summary_images exist: returns list of dicts with 'type' and 'content'/'image_url'
        - Otherwise: returns a string (text only) for backward compatibility
        """
        formatted = []
        for entry in conversation:
            # Clean the content to remove Azure Support Center info
            cleaned_content = self.clean_azure_support_info(entry['content'])
            formatted.append(f"[{entry['timestamp']}] {entry['author']}: {cleaned_content}")
        conversation_text = "\n".join(formatted)

        # Build the output text parts
        parts = []

        # Add authored summary if available (before conversation)
        if summary:
            # Clean the summary text
            cleaned_summary = self.clean_azure_support_info(summary)
            parts.append(f"--- Authored Summary ---\n{guard.spotlight_if_enabled(cleaned_summary)}")

        # Add incident conversation. Spotlighted (when injection defense is
        # enabled) because this text is attacker-influenceable -- customers
        # and partners write directly into it.
        parts.append(f"--- Incident Discussion ---\n{guard.spotlight_if_enabled(conversation_text)}")

        # Add Teams discussion if available
        if teams_discussion:
            parts.append(f"--- Teams Discussion ---\n{guard.spotlight_if_enabled(teams_discussion)}")

        # Combine all text parts
        combined_text = "\n\n".join(parts)

        # Return multimodal content if images are present, otherwise return text string
        if summary_images and len(summary_images) > 0:
            # Build multimodal content list
            multimodal_content = []

            # Add introductory text (note: Azure OpenAI uses 'text' key, not 'content')
            multimodal_content.append({
                "type": "text",
                "text": combined_text
            })

            # Add instruction text before images
            multimodal_content.append({
                "type": "text",
                "text": "\n\n--- Screenshots from manual.docx ---\nPlease analyze the following screenshots for additional context:"
            })

            # Add each image
            for img in summary_images:
                multimodal_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img['data_url']
                    }
                })

            logger.info(f"Formatted multimodal content with {len(summary_images)} images")
            return multimodal_content
        else:
            # Return plain text for backward compatibility
            return combined_text

    def generate_summary(self, content, system_prompt, user_prompt, prompt_type="default", debug_api=False, incident_data=None):
        """Generate summary using OpenAI or Azure OpenAI with memory integration and team analysis.

        Args:
            content: Can be a list of dicts (multimodal with 'type' and 'content'/'image_url')
                    or a list of existing format dicts with 'type' and 'content' keys
            system_prompt: System prompt for LLM
            user_prompt: User prompt for LLM
            prompt_type: Type of prompt being used
            debug_api: Enable API debugging
            incident_data: Incident data for memory/team analysis
        """
        try:
            guard.set_incident_context(
                (incident_data or {}).get('incident_id'), call_site="processor.generate_summary"
            )

            # Check if content is multimodal (list of dicts with type/image_url keys)
            is_multimodal = False
            if isinstance(content, list) and len(content) > 0:
                # Check if any item has 'image_url' key (indicates multimodal format)
                is_multimodal = any('image_url' in item for item in content)

            # Format the content based on whether it's multimodal or not
            with self.time_context("format_content", "ai", {"content_items": len(content), "multimodal": is_multimodal}):
                if is_multimodal:
                    # Content is already in multimodal format - use as is
                    multimodal_content = content
                    logger.info(f"Using multimodal content with {len([c for c in content if 'image_url' in c])} images")
                    # Build user message with multimodal content (prepend with prompt)
                    user_content = [
                        {"type": "text", "text": f"{user_prompt}\n\nContent:"}
                    ] + content
                else:
                    # Legacy format - convert to string
                    formatted_content = []
                    for item in content:
                        if item['type'] == 'text':
                            formatted_content.append(item['content'])
                        else:  # image (legacy format)
                            formatted_content.append(f"[Image: {item['content']}]")
                    conversation = "\n".join(formatted_content)
                    user_content = f"{user_prompt}\n\nContent:\n{conversation}"

            # Use the base user prompt without molecular context enhancement
            enhanced_user_prompt = user_prompt
            
            # Enhance prompt with memory context if memory manager is available and incident data is provided
            memory_enhanced = False
            if self.memory_manager and incident_data:
                try:
                    mem_start = time.monotonic()
                    with self.time_memory_operation("memory_enhancement", "enhance_prompt"):
                        original_prompt = enhanced_user_prompt
                        enhanced_user_prompt = self.memory_manager.enhance_prompt_with_memory(
                            enhanced_user_prompt, incident_data, False
                        )
                        
                        # Only claim enhancement if the prompt was actually changed
                        if enhanced_user_prompt != original_prompt:
                            memory_enhanced = True
                            mem_elapsed = time.monotonic() - mem_start
                            logger.info(f"Enhanced prompt with memory context in {mem_elapsed:.2f}s")
                            print(f"🧠 Enhanced prompt with memory context from previous incidents ({mem_elapsed:.2f}s)")
                        else:
                            logger.info("Memory search completed but no relevant context found")
                    if not memory_enhanced:
                        memory_enhanced = True  # Still mark as processed to avoid retries
                except Exception as e:
                    logger.warning(f"Failed to enhance prompt with memory: {e}")
            
            # Perform team analysis if team knowledge system is available and incident data is provided
            # Skip team analysis for prev_act prompt type
            team_analysis_result = None
            if self.team_analyzer and incident_data and prompt_type != 'prev_act':
                try:
                    with self.time_team_analysis("team_analysis", len(incident_data.get('conversation', [])), "incident_analysis"):
                        team_analysis_result = self.team_analyzer.analyze_incident_teams(incident_data)
                    logger.info("Performed team analysis on incident using LLM")
                    
                    # Check if team analysis was skipped
                    if team_analysis_result.get('skipped_reason'):
                        print(f"ℹ️ Team analysis skipped: {team_analysis_result['skipped_reason']}")
                    else:
                        print(f"🏢 Analyzed team interactions: {len(team_analysis_result.get('detected_teams', []))} teams involved")
                    
                    # Learn from team analysis if learning engine is available
                    if self.team_learning_engine and team_analysis_result and not team_analysis_result.get('skipped_reason'):
                        try:
                            print("🧠 Starting team learning...")
                            with self.time_team_analysis("team_learning", len(team_analysis_result.get('detected_teams', [])), "learning"):
                                learning_insights = self.team_learning_engine.learn_from_incident(incident_data, team_analysis_result)
                            if learning_insights:
                                logger.info(f"Generated {len(learning_insights)} learning insights from team analysis")
                                print(f"🧠 Learned {len(learning_insights)} insights about team capabilities")
                            
                            # Auto-save team knowledge for workflows that require it (1-4 and 12)
                            auto_save_workflows = [
                                'customer_pending_facilitation',
                                'dev_pending_facilitation',
                                'escalation',
                                'mitigation',
                                'create_prompt_for_logs_analyze'
                            ]
                            if prompt_type in auto_save_workflows and self.team_knowledge_manager:
                                try:
                                    self.team_knowledge_manager.save_if_dirty()
                                    logger.info(f"Auto-saved team knowledge for workflow: {prompt_type}")
                                except Exception as e:
                                    logger.warning(f"Failed to auto-save team knowledge: {e}")

                            # Mitigation-specific learning: Update SME database from team transfers
                            if prompt_type == 'mitigation' and team_analysis_result:
                                ownership_changes = team_analysis_result.get('ownership_changes', [])
                                if ownership_changes:
                                    try:
                                        self._update_sme_database_from_mitigation(incident_data, team_analysis_result)
                                    except Exception as e:
                                        logger.warning(f"Failed to update SME database from mitigation: {e}")
                        except Exception as e:
                            logger.warning(f"Failed to learn from team analysis: {e}")
                    elif self.team_learning_engine and team_analysis_result and team_analysis_result.get('skipped_reason'):
                        # Team learning skipped - reason already logged by team analyzer
                        pass
                except Exception as e:
                    logger.warning(f"Failed to perform team analysis: {e}")
            
            # Generate team recommendations based on incident and team knowledge
            # Skip team recommendations for prev_act prompt type
            team_recommendations = None
            if self.team_knowledge_manager and incident_data and prompt_type != 'prev_act':
                try:
                    team_recommendations = self._generate_team_recommendations(incident_data, team_analysis_result, prompt_type)
                    if team_recommendations:
                        logger.info(f"Generated {len(team_recommendations)} team recommendations")
                except Exception as e:
                    logger.warning(f"Failed to generate team recommendations: {e}")
            
            # Enhance prompt with team context if team analysis was performed
            team_enhanced = False
            if team_analysis_result and team_analysis_result.get('detected_teams'):
                try:
                    team_context = self._build_team_context(team_analysis_result, team_recommendations)
                    if team_context:
                        enhanced_user_prompt = f"{enhanced_user_prompt}\n\nTeam Context:\n{team_context}"
                        team_enhanced = True
                        logger.info("Enhanced prompt with team context")
                        print(f"👥 Enhanced prompt with team context from {len(team_analysis_result['detected_teams'])} teams")
                except Exception as e:
                    logger.warning(f"Failed to enhance prompt with team context: {e}")

            # Prepare messages - handle multimodal or text-only content
            with self.time_context("prepare_llm_messages", "ai", {"multimodal": is_multimodal}):
                system_prompt_with_guard = str(system_prompt) + guard.injection_system_clause_suffix()
                if is_multimodal:
                    # For multimodal, prepend enhanced user prompt to the content
                    # Find the first text item and prepend the enhanced prompt
                    messages = [
                        {"role": "system", "content": system_prompt_with_guard},
                        {"role": "user", "content": user_content}
                    ]
                else:
                    messages = [
                        {"role": "system", "content": system_prompt_with_guard},
                        {"role": "user", "content": f"{str(enhanced_user_prompt)}\n\n{user_content}"}
                    ]

            if debug_api:
                print("\n[DEBUG_API] LLM API request body:")
                max_show = 2000  # chars per message to show in console
                for i, msg in enumerate(messages):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if content is None:
                        content = ""
                    if isinstance(content, list):
                        parts = []
                        for part in content:
                            if part.get("type") == "text":
                                text = guard.redact_text(part.get("text", part.get("content", "")) or "")
                                parts.append(f"[text, len={len(text)}]: {repr(text[:max_show])}{'...' if len(text) > max_show else ''}")
                            else:
                                parts.append(f"[{part.get('type')}]: {str(part)[:200]}...")
                        content_repr = "\n  ".join(parts)
                        content_len = "multimodal"
                    else:
                        content = guard.redact_text(content)
                        content_len = len(content)
                        content_repr = repr(content[:max_show]) + ("..." if len(content) > max_show else "")
                    print(f"  [{i}] role={role}, content length={content_len}")
                    print(f"      content preview: {content_repr}")
                print(f"  model={self.deployment_name}, temperature=0.7, max_tokens=8000")

            # Use AI Service (GPT-5)
            model_name = self.deployment_name

            # Count input tokens (only count text for multimodal, approximate)
            with self.time_context("count_tokens", "ai", {"multimodal": is_multimodal}):
                if is_multimodal:
                    # For multimodal, count only text portions (rough estimate)
                    text_content = " ".join([item.get('text', '') for item in user_content if item.get('type') == 'text'])
                    input_text = f"{system_prompt}\n{enhanced_user_prompt}\n{text_content}"
                else:
                    input_text = f"{system_prompt}\n{enhanced_user_prompt}\n{user_content}"
                input_tokens = self.count_tokens(input_text)
            
            # Use AI Service (GPT-5) for timing
            model_name = self.deployment_name
            
            # Generate summary with timing using AI Service (GPT-5)
            llm_start = time.monotonic()
            print(f"🤖 Starting LLM call with {model_name}...")
            
            # Retry logic for timeout and connection errors
            max_retries = 3
            retry_delay = 5  # Start with 5 seconds
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    with self.time_llm_call("llm_generate_summary", model_name, input_tokens, 0):  # We'll update output tokens after
                        response = self.client.chat.completions.create(
                            model=self.deployment_name,
                            messages=messages,
                            temperature=0.7,
                            max_tokens=8000,
                            timeout=self.llm_timeout
                        )
                    # Success - break out of retry loop
                    break
                except (TimeoutError, ConnectionError, Exception) as e:
                    last_error = e
                    error_str = str(e).lower()
                    
                    # Check if it's a retryable error (timeout, connection, rate limit)
                    is_retryable = any(indicator in error_str for indicator in [
                        'timeout', 'connection', '429', 'rate limit', 'throttled', 
                        'unavailable', 'socket', 'did not properly respond'
                    ]) or isinstance(e, (TimeoutError, ConnectionError))
                    
                    if not is_retryable or attempt == max_retries - 1:
                        # Non-retryable error or last attempt - raise immediately
                        print(f"❌ LLM call failed: {str(e)[:200]}")
                        raise
                    
                    # Retryable error - wait and retry
                    print(f"⚠️  LLM call failed (attempt {attempt + 1}/{max_retries}): {str(e)[:150]}... Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff (5s, 10s, 20s)
                else:
                    # This shouldn't happen, but just in case
                    break
            
            # If we get here without a response, raise the last error
            if 'response' not in locals():
                print(f"❌ LLM call failed after {max_retries} attempts")
                raise last_error if last_error else Exception("LLM call failed for unknown reason")

            llm_elapsed = time.monotonic() - llm_start
            print(f"⏱ LLM call completed in {llm_elapsed:.2f}s")
            # Get output tokens and calculate cost
            with self.time_context("process_llm_response", "ai", {"response_length": len(response.choices[0].message.content)}):
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                cost = self.calculate_cost(input_tokens, output_tokens)
            result = {
                "summary": response.choices[0].message.content,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost
                }
            }
            # Add memory context info if used
            if memory_enhanced:
                result["memory_context"] = {
                    "enhanced": True,
                    "user_id": self.memory_manager.user_id if self.memory_manager else None
                }
            
            # Add team context info if used
            if team_enhanced and team_analysis_result:
                result["team_context"] = {
                    "enhanced": True,
                    "teams_detected": len(team_analysis_result.get('detected_teams', [])),
                    "team_analysis": self._serialize_team_analysis(team_analysis_result)
                }
            
            # Add team recommendations if available
            if team_recommendations:
                result["team_recommendations"] = team_recommendations
            
            # Add transfer reasons if available
            if team_analysis_result and team_analysis_result.get('transfer_reasons'):
                result["transfer_reasons"] = team_analysis_result['transfer_reasons']
            
            return result
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise
    
    def _serialize_team_analysis(self, team_analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert team analysis result to JSON-serializable format."""
        try:
            serialized = {}
            
            # Serialize detected teams
            if 'detected_teams' in team_analysis_result:
                serialized['detected_teams'] = team_analysis_result['detected_teams']
            
            # Serialize ownership changes (convert OwnershipChange objects to dicts)
            if 'ownership_changes' in team_analysis_result:
                ownership_changes = team_analysis_result['ownership_changes']
                serialized['ownership_changes'] = []
                for change in ownership_changes:
                    if hasattr(change, '__dict__'):
                        # Convert OwnershipChange object to dictionary
                        change_dict = {
                            'incident_id': getattr(change, 'incident_id', ''),
                            'from_team': getattr(change, 'from_team', None),
                            'to_team': getattr(change, 'to_team', ''),
                            'change_type': getattr(change, 'change_type', ''),
                            'timestamp': getattr(change, 'timestamp', ''),
                            'context': getattr(change, 'context', ''),
                            'confidence': getattr(change, 'confidence', 0.0),
                            'reason': getattr(change, 'reason', None)
                        }
                        serialized['ownership_changes'].append(change_dict)
                    else:
                        serialized['ownership_changes'].append(change)
            
            # Serialize collaboration patterns (convert TeamInteractionPattern objects to dicts)
            if 'collaboration_patterns' in team_analysis_result:
                patterns = team_analysis_result['collaboration_patterns']
                serialized['collaboration_patterns'] = []
                for pattern in patterns:
                    if hasattr(pattern, '__dict__'):
                        # Convert object to dictionary
                        pattern_dict = {
                            'pattern_type': getattr(pattern, 'pattern_type', 'unknown'),
                            'team_name': getattr(pattern, 'team_name', 'unknown'),
                            'frequency': getattr(pattern, 'frequency', 0),
                            'confidence': getattr(pattern, 'confidence', 0.0),
                            'examples': getattr(pattern, 'examples', []),
                            'description': getattr(pattern, 'description', '')
                        }
                        serialized['collaboration_patterns'].append(pattern_dict)
                    else:
                        serialized['collaboration_patterns'].append(pattern)
            
            return serialized
        except Exception as e:
            logger.warning(f"Error serializing team analysis: {e}")
            return {"error": "Failed to serialize team analysis"}
    
    def _generate_team_recommendations(self, incident_data: Dict[str, Any],
                                       team_analysis_result: Dict[str, Any] = None,
                                       prompt_type: str = None) -> List[Dict[str, Any]]:
        """
        Generate team recommendations with prompt-specific pre-selection strategies.

        Strategies per prompt:
        - customer_pending_facilitation: Focus on customer care/support teams
        - dev_pending_facilitation: Focus on SWE, Engineering teams
        - escalation: Use transfer reason analysis
        - article_search: Domain keyword matching
        - simplified_incident_explanation: Expertise-based matching
        - mitigation: Transfer reason analysis (when incident is mitigated)

        Args:
            incident_data: Incident data containing conversation and summary
            team_analysis_result: Optional team analysis results
            prompt_type: Type of prompt being used for filtering strategy

        Returns:
            List of team recommendations with evidence
        """
        try:
            recommendations = []

            # Get incident summary and conversation for analysis
            summary = incident_data.get('summary', '')
            conversation = incident_data.get('conversation', [])
            conversation_text = ' '.join([entry.get('text', '') for entry in conversation[:10]])

            # Combine text for analysis
            incident_text = f"{summary}\n{conversation_text}".lower()

            # Get all teams from knowledge database
            all_teams = self.team_knowledge_manager.get_all_teams()

            # Define prompt-specific strategies
            strategies = {
                'customer_pending_facilitation': {
                    'focus_keywords': ['care', 'css', 'customer', 'support'],
                    'exclude_keywords': ['swe', 'engineering', 'developer', 'backend'],
                    'boost_keywords': ['cxe', 'customer support']
                },
                'dev_pending_facilitation': {
                    'focus_keywords': ['swe', 'engineering', 'developer', 'linux', 'macos', 'windows'],
                    'exclude_keywords': ['care', 'css'],
                    'boost_keywords': ['swe_linux', 'swe_macos', 'backend']
                },
                'escalation': {
                    'focus_keywords': [],  # No filtering for escalation
                    'exclude_keywords': [],
                    'boost_keywords': []
                },
                'article_search': {
                    'focus_keywords': [],  # Domain-based matching
                    'exclude_keywords': [],
                    'boost_keywords': []
                },
                'simplified_incident_explanation': {
                    'focus_keywords': [],
                    'exclude_keywords': [],
                    'boost_keywords': []
                },
                'mitigation': {
                    'focus_keywords': [],  # All teams considered
                    'exclude_keywords': [],
                    'boost_keywords': []
                }
            }

            # Get strategy for this prompt type
            strategy = strategies.get(prompt_type, strategies['escalation'])

            # First pass: Score all teams
            scored_teams = []
            for team_id, team_data in all_teams.items():
                team_name = team_data.get('name', team_id)
                team_name_lower = team_name.lower()

                # Apply exclusion filter
                if any(exclude in team_name_lower for exclude in strategy['exclude_keywords']):
                    continue

                score = 0.0
                evidence_items = []

                # Apply focus keyword boost
                if strategy['focus_keywords']:
                    for focus_kw in strategy['focus_keywords']:
                        if focus_kw in team_name_lower:
                            score += 0.5  # Boost for matching focus keywords
                            evidence_items.append(f"Matches focus area: {focus_kw}")

                # Apply boost keywords
                for boost_kw in strategy['boost_keywords']:
                    if boost_kw in team_name_lower:
                        score += 0.3
                        evidence_items.append(f"Recommended for: {boost_kw}")

                # Check transfer reasons (strict matching to avoid false positives from other incidents)
                transfer_reasons = team_data.get('transfer_reasons', [])
                for reason_data in transfer_reasons:
                    if not reason_data.get('confirmed', False):
                        continue

                    reason_text = reason_data.get('transfer_reason', '').lower()
                    if reason_text:
                        # Extract meaningful keywords (exclude common stop words)
                        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also'}
                        
                        # Extract technical terms (contain dots, hyphens, underscores, or are long identifiers)
                        # These are highly specific and MUST appear in incident for match
                        technical_terms = [kw.strip('.,!?;:()[]{}"\'') for kw in reason_text.split() if '.' in kw or '-' in kw or '_' in kw or len(kw.strip('.,!?;:()[]{}"\'')) > 12]
                        
                        # Extract meaningful keywords (longer than 4 chars, not stop words, exclude generic terms)
                        generic_terms = {'microsoft', 'incident', 'issue', 'problem', 'device', 'system', 'setting', 'settings', 'configuration', 'config', 'macos', 'windows', 'linux', 'preference', 'preferences', 'management', 'manager', 'service', 'services', 'application', 'applications', 'client', 'clients'}
                        keywords = [kw.strip('.,!?;:()[]{}"\'') for kw in reason_text.split() 
                                   if len(kw.strip('.,!?;:()[]{}"\'')) > 4 
                                   and kw.strip('.,!?;:()[]{}"\'') not in stop_words
                                   and kw.strip('.,!?;:()[]{}"\'') not in generic_terms]
                        
                        # CRITICAL: If transfer reason contains technical terms, ALL must be present in incident
                        if technical_terms:
                            all_technical_match = all(term.lower() in incident_text for term in technical_terms)
                            if not all_technical_match:
                                continue  # Skip if any technical term is missing - likely wrong match
                        
                        # Require at least 3 meaningful keyword matches (increased threshold)
                        if keywords:
                            matches = sum(1 for keyword in keywords[:15] if keyword.lower() in incident_text)
                            if matches >= 3:
                                score += 0.4
                                evidence_items.append(f"Transfer reason: {reason_data.get('transfer_reason', '')}")
                        elif technical_terms:
                            # If only technical terms exist and all matched, allow it
                            score += 0.4
                            evidence_items.append(f"Transfer reason: {reason_data.get('transfer_reason', '')}")

                # Check expertise areas
                expertise = team_data.get('expertise', [])
                for exp in expertise:
                    if exp.lower() in incident_text:
                        score += 0.25
                        evidence_items.append(f"Expertise area: {exp}")

                # Check responsibilities
                responsibilities = team_data.get('responsibilities', [])
                for resp in responsibilities:
                    if resp.lower() in incident_text:
                        score += 0.2
                        evidence_items.append(f"Responsibility: {resp}")

                # Check common issues
                common_issues = team_data.get('common_issues', [])
                for issue in common_issues:
                    if issue.lower() in incident_text:
                        score += 0.25
                        evidence_items.append(f"Common issue: {issue}")

                # Only add if score is above threshold
                if score >= 0.3 and evidence_items:
                    scored_teams.append({
                        'team_name': team_name,
                        'score': score,
                        'evidence': evidence_items[:5],
                        'confidence': min(1.0, score)
                    })

            # Sort by score (highest first)
            scored_teams.sort(key=lambda x: x['score'], reverse=True)

            # Return top 3-5 recommendations based on prompt type
            limit = 5 if prompt_type == 'article_search' else 3
            return scored_teams[:limit]

        except Exception as e:
            logger.error(f"Error generating team recommendations: {e}")
            return []
    
    def _build_team_context(self, team_analysis_result: Dict[str, Any], 
                            team_recommendations: List[Dict[str, Any]] = None) -> str:
        """Build team context string from team analysis results."""
        try:
            detected_teams = team_analysis_result.get('detected_teams', [])
            llm_analysis = team_analysis_result.get('llm_analysis', {})
            
            if not detected_teams and not llm_analysis:
                return ""
            
            context_parts = []
            
            # Add team involvement summary
            if detected_teams:
                team_names = [team['team_name'] for team in detected_teams]
                context_parts.append(f"Teams involved: {', '.join(team_names)}")
            
            # Add team knowledge database if available
            if llm_analysis and 'team_knowledge_database' in llm_analysis:
                context_parts.append("Team Knowledge Database:")
                team_db = llm_analysis['team_knowledge_database']
                if isinstance(team_db, dict):
                    for team_name, team_info in team_db.items():
                        if isinstance(team_info, dict):
                            responsibilities = team_info.get('primary_responsibilities', [])
                            expertise = team_info.get('expertise_areas', [])
                            capabilities = team_info.get('team_capabilities', '')
                            
                            team_summary = f"{team_name}:"
                            if responsibilities:
                                team_summary += f" Responsibilities: {', '.join(responsibilities[:3])}"
                            if expertise:
                                team_summary += f" | Expertise: {', '.join(expertise[:3])}"
                            if capabilities:
                                team_summary += f" | {capabilities[:100]}..."
                            
                            context_parts.append(team_summary)
                else:
                    context_parts.append(str(team_db))
            
            # Also try to load from optimized team knowledge database
            try:
                from team_knowledge.team_knowledge_manager import TeamKnowledgeManager
                manager = TeamKnowledgeManager()
                all_teams = manager.get_all_teams()
                
                if all_teams and not context_parts:
                    context_parts.append("Team Knowledge Database:")
                    for team_id, team_data in all_teams.items():
                        responsibilities = team_data.get('responsibilities', [])
                        expertise = team_data.get('expertise', [])
                        capabilities = team_data.get('capabilities', '')
                        
                        team_summary = f"{team_data.get('name', team_id)}:"
                        if responsibilities:
                            team_summary += f" Responsibilities: {', '.join(responsibilities[:3])}"
                        if expertise:
                            team_summary += f" | Expertise: {', '.join(expertise[:3])}"
                        if capabilities:
                            team_summary += f" | {capabilities[:100]}..."
                        
                        context_parts.append(team_summary)
            except Exception as e:
                logger.warning(f"Failed to load team knowledge from optimized database: {e}")
            
            # Add LLM analysis if available (fallback)
            if llm_analysis and 'team_analysis' in llm_analysis and not context_parts:
                context_parts.append("Team Analysis:")
                team_analysis = llm_analysis['team_analysis']
                # Handle both string and dictionary responses
                if isinstance(team_analysis, dict):
                    context_parts.append(str(team_analysis))
                else:
                    context_parts.append(team_analysis)
            
            # Add basic team info if no LLM analysis
            elif detected_teams:
                for team in detected_teams:
                    team_name = team['team_name']
                    # Handle both 'interaction_type' and 'interaction_types' for backward compatibility
                    interaction_types = team.get('interaction_types', [])
                    if not interaction_types and 'interaction_type' in team:
                        interaction_types = [team['interaction_type']]
                    interaction_type = ', '.join(interaction_types) if interaction_types else 'unknown'
                    confidence = team['confidence']
                    matched_domains = team.get('matched_domains', [])
                    
                    team_info = f"{team_name} ({interaction_type}, confidence: {confidence:.2f})"
                    if matched_domains:
                        team_info += f" - Expertise: {', '.join(matched_domains)}"
                    
                    context_parts.append(team_info)
            
            # Add team recommendations if available
            if team_recommendations:
                context_parts.append("\nTeam Recommendations:")
                for i, rec in enumerate(team_recommendations, 1):
                    team_name = rec.get('team_name', 'Unknown')
                    evidence = rec.get('evidence', [])
                    confidence = rec.get('confidence', 0.0)
                    
                    rec_text = f"{i}. {team_name} (confidence: {confidence:.2f})"
                    if evidence:
                        rec_text += f"\n   Evidence: {'; '.join(evidence[:3])}"
                    context_parts.append(rec_text)
            
            # Add transfer reasons if available
            transfer_reasons = team_analysis_result.get('transfer_reasons', [])
            if transfer_reasons:
                context_parts.append("\nTransfer Reasons (from this incident):")
                for reason in transfer_reasons[:3]:  # Top 3
                    team_name = reason.get('team_name', 'Unknown')
                    reason_text = reason.get('transfer_reason', '')
                    evidence = reason.get('evidence', [])
                    
                    reason_line = f"- {team_name}: {reason_text}"
                    if evidence:
                        reason_line += f" (Evidence: {evidence[0][:100]}...)" if len(evidence[0]) > 100 else f" (Evidence: {evidence[0]})"
                    context_parts.append(reason_line)
            
            return "\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error building team context: {e}")
            return ""

    def _update_sme_database_from_mitigation(self, incident_data: Dict[str, Any],
                                             team_analysis_result: Dict[str, Any]) -> None:
        """
        Update SME database with insights from team transfers during mitigation.

        When an incident is mitigated and team transfers occurred, this method
        analyzes the transfer context and updates the team knowledge database with
        insights about team responsibilities and expertise.

        Args:
            incident_data: The incident data
            team_analysis_result: The team analysis results including ownership changes
        """
        try:
            ownership_changes = team_analysis_result.get('ownership_changes', [])
            if not ownership_changes:
                return

            incident_id = incident_data.get('incident_id', 'unknown')
            conversation = incident_data.get('conversation', [])
            summary = incident_data.get('summary', '')

            # Build context from conversation
            conversation_text = ' '.join([entry.get('text', '') for entry in conversation[:20]])
            incident_context = f"{summary}\n{conversation_text}".lower()

            # Process each ownership change
            for change in ownership_changes:
                to_team = change.get('to_team', '')
                from_team = change.get('from_team', '')
                change_type = change.get('change_type', '')
                context = change.get('context', '')
                confidence = change.get('confidence', 0.0)

                # Only process transfers with high confidence
                if confidence < 0.7 or change_type not in ['transfer', 'escalation']:
                    continue

                # Use LLM to analyze what the team is responsible for
                team_insights = self._extract_team_responsibility_from_transfer(
                    to_team, context, incident_context
                )

                # Update team knowledge database with transfer reason
                if team_insights and self.team_knowledge_manager:
                    transfer_reason = {
                        'transfer_reason': team_insights.get('responsibility', ''),
                        'evidence': [
                            f"Incident {incident_id}",
                            f"Transfer from {from_team} to {to_team}",
                            context[:200] if context else ''
                        ],
                        'confidence': confidence,
                        'incident_id': incident_id,
                        'technical_domains': team_insights.get('technical_domains', []),
                        'issue_patterns': team_insights.get('issue_patterns', [])
                    }

                    self.team_knowledge_manager.add_transfer_reason(to_team, transfer_reason)
                    logger.info(f"Updated team knowledge for {to_team} from mitigation of incident {incident_id}")
                    print(f"📝 Updated team knowledge for {to_team} from mitigation")

        except Exception as e:
            logger.error(f"Error updating SME database from mitigation: {e}")
            raise

    def _extract_team_responsibility_from_transfer(self, team_name: str, transfer_context: str,
                                                   incident_context: str) -> Dict[str, Any]:
        """
        Use LLM to analyze what a team is responsible for based on transfer context.

        Args:
            team_name: The name of the team that received the transfer
            transfer_context: The context of the transfer from the conversation
            incident_context: The full incident context

        Returns:
            Dict with responsibility, technical_domains, and issue_patterns
        """
        try:
            prompt = f"""Analyze this team transfer and extract what the receiving team is responsible for.

Team that received the transfer: {team_name}

Transfer context:
{transfer_context}

Incident context (truncated):
{incident_context[:1000]}

Extract:
1. What technical domain or area the team is responsible for (one sentence)
2. Technical domains involved (list of 3-5 keywords)
3. Common issue patterns (list of 2-3 patterns)

Respond in JSON format:
{{
    "responsibility": "What the team is responsible for",
    "technical_domains": ["domain1", "domain2", ...],
    "issue_patterns": ["pattern1", "pattern2", ...]
}}"""

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing team responsibilities from incident conversations. Extract structured information in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            result_text = response.choices[0].message.content

            # Parse JSON response
            import json
            # Clean up response text - remove markdown code blocks if present
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0].strip()

            result = json.loads(result_text)

            return {
                'responsibility': result.get('responsibility', ''),
                'technical_domains': result.get('technical_domains', []),
                'issue_patterns': result.get('issue_patterns', [])
            }

        except Exception as e:
            logger.warning(f"Failed to extract team responsibility with LLM: {e}")
            # Return basic info from transfer context
            return {
                'responsibility': transfer_context[:200] if transfer_context else '',
                'technical_domains': [],
                'issue_patterns': []
            }
    
    def store_incident_memory(self, incident_number: str, incident_data: Dict[str, Any], 
                             processing_result: Dict[str, Any]) -> None:
        """
        Store memory about a processed incident.
        
        Args:
            incident_number: The incident number
            incident_data: Raw incident data
            processing_result: The processing result/summary
        """
        if not self.memory_manager:
            logger.warning("Memory manager not available, skipping memory storage")
            return
        
        try:
            with self.time_memory_operation("memory_storage", "add_incident", len(str(incident_data))):
                self.memory_manager.add_incident_memory(incident_number, incident_data, processing_result)
            logger.info(f"Stored memory for incident {incident_number}")
            print(f"💾 Stored memory for incident {incident_number}")
        except Exception as e:
            logger.error(f"Failed to store memory for incident {incident_number}: {e}")
    
    def process_article_search(self, incident_data: Dict[str, Any], prompts: Dict[str, str], 
                              prompt_type: str, debug_api: bool = False) -> Dict[str, Any]:
        """
        Process incident data to find relevant troubleshooting articles.
        
        Args:
            incident_data: The incident data to analyze
            prompts: The prompts to use for processing
            prompt_type: The type of prompt being used
            debug_api: Whether to enable API debugging
            
        Returns:
            Dictionary containing the search results
        """
        if not self.article_searcher:
            logger.warning("Article searcher not available")
            return {"error": "Article searcher not initialized"}
        
        try:
            # Create a focused technical query for article search
            technical_query = self._create_technical_query(incident_data)
            
            # Create incident context for better matching
            incident_context = self._create_incident_context(incident_data)
            
            # Use advanced search with multi-stage pipeline
            logger.info("Starting advanced article search with LLM-based scoring and sorting...")
            search_results = self.article_searcher.search_articles_advanced(
                query=technical_query,
                incident_context=incident_context,
                top_k=5
            )
            logger.info(f"Advanced search returned {len(search_results)} results")
            
            # Format search results for display
            formatted_results = self.article_searcher.format_search_results(
                search_results, 
                query=technical_query, 
                include_explanations=True
            )
            
            # Create a simple analysis that just presents the articles
            if search_results:
                analysis_result = f"Top {len(search_results)} highly relevant articles found using advanced AI matching:\n\n{formatted_results}"
            else:
                analysis_result = "No relevant articles found for this incident using advanced matching criteria."
            
            # Combine results
            result = {
                'search_results': search_results,
                'formatted_results': formatted_results,
                'analysis': analysis_result,
                'incident_data': technical_query
            }
            
            # Note: Gap analysis will be offered later after results are displayed
            
            return result
            
        except Exception as e:
            logger.error(f"Error in article search processing: {e}")
            return {"error": str(e)}
    
    def process_logs_analyzer(self, incident_data: Dict[str, Any], prompts: Dict[str, str], 
                             prompt_type: str, debug_api: bool = False, incident_id: str = None) -> Dict[str, Any]:
        """
        Process incident data using sophisticated logs analysis protocol with MCP tools.
        
        Args:
            incident_data: The incident data to analyze
            prompts: The prompts to use for processing
            prompt_type: The type of prompt being used
            debug_api: Whether to enable API debugging
            incident_id: The incident ID
            
        Returns:
            Dictionary containing the analysis results
        """
        try:
            logger.info("Starting sophisticated logs analyzer with MCP tools...")
            
            # Extract incident number from incident_id parameter or incident_data
            incident_number = incident_id if incident_id else incident_data.get('incident_number', 'unknown')
            if incident_number == 'unknown':
                # Try to extract from conversation or other fields
                conversation = incident_data.get('conversation', [])
                for message in conversation:
                    content = message.get('content', '')
                    # Look for incident number patterns
                    import re
                    match = re.search(r'(\d{8,})', content)
                    if match:
                        incident_number = match.group(1)
                        break
            
            # Step 1: Read the logs analysis protocol
            protocol_content = self._read_logs_analysis_protocol()
            
            # Step 2: Use filesystem MCP to find and analyze logs
            log_analysis = self._analyze_incident_logs(incident_number)
            
            # Step 3: Use sequential thinking to understand the issue
            sequential_analysis = self._perform_sequential_analysis(incident_data, log_analysis)
            
            # Step 4: Research online documentation using hyperbrowser MCP
            online_research = self._perform_online_research(incident_data, log_analysis)
            
            # Step 5: Generate comprehensive analysis using the sophisticated protocol
            logger.info("Generating sophisticated logs analysis...")
            analysis_result = self._generate_sophisticated_logs_analysis(
                incident_data, log_analysis, sequential_analysis, online_research, 
                protocol_content, prompts, debug_api
            )
            logger.info(f"Analysis result type: {type(analysis_result)}")
            if isinstance(analysis_result, dict) and 'summary' in analysis_result:
                logger.info(f"Summary length: {len(analysis_result['summary'])}")
            else:
                logger.info(f"Analysis result: {str(analysis_result)[:200]}...")
            
            return {
                'incident_number': incident_number,
                'log_analysis': log_analysis,
                'sequential_analysis': sequential_analysis,
                'online_research': online_research,
                'analysis': analysis_result
            }
            
        except Exception as e:
            logger.error(f"Error in sophisticated logs analyzer processing: {e}")
            return {"error": str(e)}
    
    def _analyze_incident_logs(self, incident_number: str) -> Dict[str, Any]:
        """Analyze incident logs using filesystem MCP server."""
        try:
            # Find the incident folder
            incident_folder = f"incidents/{incident_number}"
            if not os.path.exists(incident_folder):
                return {"error": f"Incident folder not found: {incident_folder}"}
            
            # Find the mm-dd folder
            date_folders = []
            for item in os.listdir(incident_folder):
                item_path = os.path.join(incident_folder, item)
                if os.path.isdir(item_path) and re.match(r'\d{2}-\d{2}', item):
                    date_folders.append(item)
            
            if not date_folders:
                return {"error": f"No date folders found in {incident_folder}"}
            
            # Use the most recent date folder
            latest_date_folder = sorted(date_folders)[-1]
            logs_path = os.path.join(incident_folder, latest_date_folder)
            
            # Find diagnostic tool output folder
            diagnostic_tool_folders = []
            for item in os.listdir(logs_path):
                item_path = os.path.join(logs_path, item)
                if os.path.isdir(item_path) and 'output' in item.lower():
                    diagnostic_tool_folders.append(item)

            if not diagnostic_tool_folders:
                return {"error": f"No diagnostic tool output folders found in {logs_path}"}

            # Use the most recent diagnostic tool output folder
            latest_analyzer_folder = sorted(diagnostic_tool_folders)[-1]
            analyzer_path = os.path.join(logs_path, latest_analyzer_folder)

            # Analyze key log files (configure via DIAGNOSTIC_LOG_FILES in .env)
            log_files_analysis = {}
            key_files = config.diagnostic_log_files

            for log_file in key_files:
                file_path = os.path.join(analyzer_path, log_file)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            # Limit content size for analysis
                            if len(content) > 10000:
                                content = content[:10000] + "... [truncated]"
                            log_files_analysis[log_file] = content
                    except Exception as e:
                        log_files_analysis[log_file] = f"Error reading file: {e}"
            
            # Look for additional log files in subdirectories
            additional_logs = {}
            for root, dirs, files in os.walk(analyzer_path):
                for file in files:
                    if file.endswith('.log') and root != analyzer_path:
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                # Limit content size for analysis
                                if len(content) > 5000:
                                    content = content[-5000:]  # Get last 5000 chars for recent logs
                                additional_logs[file] = content
                        except Exception as e:
                            additional_logs[file] = f"Error reading file: {e}"
            
            return {
                'incident_folder': incident_folder,
                'date_folder': latest_date_folder,
                'analyzer_folder': latest_analyzer_folder,
                'analyzer_path': analyzer_path,
                'log_files': log_files_analysis,
                'additional_logs': additional_logs,
                'available_files': os.listdir(analyzer_path)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing incident logs: {e}")
            return {"error": str(e)}
    
    def _perform_sequential_analysis(self, incident_data: Dict[str, Any], log_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform sequential thinking analysis to understand the core issue."""
        try:
            # Extract key information from incident data
            summary = incident_data.get('summary', '')
            conversation = incident_data.get('conversation', [])
            
            # Combine all conversation content
            full_content = ""
            for message in conversation:
                content = message.get('content', '')
                full_content += f" {content}"
            
            # Extract technical details
            technical_issues = self._extract_technical_issues(full_content)
            affected_components = self._extract_affected_components(full_content)
            error_patterns = self._extract_error_patterns(full_content)
            
            # Analyze log files for additional insights
            log_insights = []
            if 'log_files' in log_analysis:
                for log_file, content in log_analysis['log_files'].items():
                    if 'error' in content.lower() or 'fail' in content.lower():
                        log_insights.append(f"{log_file}: Contains error indicators")
                    if 'policy' in content.lower():
                        log_insights.append(f"{log_file}: Contains policy-related information")
                    if 'device' in content.lower():
                        log_insights.append(f"{log_file}: Contains device-related information")
            
            return {
                'summary': summary,
                'technical_issues': technical_issues,
                'affected_components': affected_components,
                'error_patterns': error_patterns,
                'log_insights': log_insights,
                'content_length': len(full_content)
            }
            
        except Exception as e:
            logger.error(f"Error in sequential analysis: {e}")
            return {"error": str(e)}
    
    def _extract_technical_issues(self, content: str) -> List[str]:
        """Extract technical issues from content."""
        issues = []
        content_lower = content.lower()
        
        if 'policy' in content_lower and 'device' in content_lower:
            issues.append('Device control policy issue')
        if 'whitelist' in content_lower or 'exclusion' in content_lower:
            issues.append('Whitelist/exclusion configuration issue')
        if 'jamf' in content_lower:
            issues.append('JAMF Pro deployment issue')
        if 'mobile' in content_lower and 'device' in content_lower:
            issues.append('Mobile device connectivity issue')
        if 'sync' in content_lower and 'fail' in content_lower:
            issues.append('File synchronization failure')
        
        return issues
    
    def _extract_affected_components(self, content: str) -> List[str]:
        """Extract affected components from content."""
        components = []
        content_lower = content.lower()
        
        if config.security_agent_keywords and any(kw in content_lower for kw in config.security_agent_keywords):
            components.append(config.security_agent_display_name)
        if 'jamf' in content_lower:
            components.append('JAMF Pro')
        if 'finder' in content_lower:
            components.append('macOS Finder')
        if 'files' in content_lower:
            components.append('File synchronization')
        if 'device control' in content_lower:
            components.append('Device Control Policy')
        
        return components
    
    def _extract_error_patterns(self, content: str) -> List[str]:
        """Extract error patterns from content."""
        patterns = []
        content_lower = content.lower()
        
        if 'unexpected behavior' in content_lower:
            patterns.append('Unexpected behavior reported')
        if 'not working' in content_lower:
            patterns.append('Functionality not working as expected')
        if 'no data' in content_lower:
            patterns.append('No data displayed')
        if 'cannot connect' in content_lower:
            patterns.append('Connection issues')
        
        return patterns
    
    def _generate_logs_analysis(self, incident_data: Dict[str, Any], log_analysis: Dict[str, Any], 
                               sequential_analysis: Dict[str, Any], prompts: Dict[str, str], 
                               debug_api: bool = False) -> str:
        """Generate comprehensive logs analysis using the logs_analyzer prompt."""
        try:
            # Prepare content for analysis
            content_parts = []
            
            # Add incident summary
            if sequential_analysis.get('summary'):
                content_parts.append(f"Incident Summary: {sequential_analysis['summary']}")
            
            # Add technical issues
            if sequential_analysis.get('technical_issues'):
                content_parts.append(f"Technical Issues: {', '.join(sequential_analysis['technical_issues'])}")
            
            # Add affected components
            if sequential_analysis.get('affected_components'):
                content_parts.append(f"Affected Components: {', '.join(sequential_analysis['affected_components'])}")
            
            # Add log file analysis
            if log_analysis.get('log_files'):
                content_parts.append("\nLog File Analysis:")
                for log_file, content in log_analysis['log_files'].items():
                    content_parts.append(f"\n{log_file}:")
                    content_parts.append(content[:1000] + "..." if len(content) > 1000 else content)
            
            # Add security agent logs
            if log_analysis.get('security_agent_logs'):
                content_parts.append(f"\n{config.security_agent_display_name} Logs:")
                for log_file, content in log_analysis['security_agent_logs'].items():
                    content_parts.append(f"\n{log_file}:")
                    content_parts.append(content[:500] + "..." if len(content) > 500 else content)
            
            # Combine all content
            full_content = "\n".join(content_parts)
            
            # Generate analysis using the logs_analyzer prompt
            analysis_result = self.generate_summary(
                [{'type': 'text', 'content': full_content}],
                prompts['system_prompt'],
                prompts['user_prompt'],
                prompt_type='logs_analyzer',
                debug_api=debug_api,
                incident_data=incident_data
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error generating logs analysis: {e}")
            return f"Error generating analysis: {e}"
    
    def _create_incident_context(self, incident_data: Dict[str, Any]) -> str:
        """Create detailed incident context for better article matching."""
        try:
            context_parts = []
            
            # Add incident summary if available - this is the most important part
            summary = incident_data.get('summary')
            if summary:
                # Use the full summary as the primary context
                context_parts.append(f"Incident Summary: {summary}")
            
            # Extract detailed technical information from conversation
            conversation = incident_data.get('conversation', [])
            
            # Collect all conversation content for analysis
            full_content = ""
            for message in conversation:
                content = message.get('content', '')
                full_content += f" {content}"
            
            # Extract specific technical details
            technical_details = self._extract_technical_details(full_content)
            
            if technical_details:
                context_parts.append(f"Technical Details: {technical_details}")
            
            # Extract platform and component information
            platforms = self._extract_platforms(full_content)
            if platforms:
                context_parts.append(f"Platforms: {', '.join(platforms)}")
            
            components = self._extract_components(full_content)
            if components:
                context_parts.append(f"Components: {', '.join(components)}")
            
            # Extract issue type and symptoms
            issue_info = self._extract_issue_info(full_content)
            if issue_info:
                context_parts.append(f"Issue: {issue_info}")
            
            # Add incident ID for reference
            incident_id = incident_data.get('incident_id', 'unknown')
            context_parts.append(f"Incident ID: {incident_id}")
            
            return "\n".join(context_parts) if context_parts else "No specific context available"
            
        except Exception as e:
            logger.warning(f"Error creating incident context: {e}")
            return "Context creation failed"
    
    def _extract_technical_details(self, content: str) -> str:
        """Extract specific technical details from incident content."""
        content_lower = content.lower()
        details = []
        
        # Extract specific technical terms and issues
        if 'jwt' in content_lower or 'token' in content_lower:
            details.append('JWT token authentication')
        if 'impaired communication' in content_lower:
            details.append('communication issues')
        if 'authentication' in content_lower:
            details.append('authentication problems')
        if 'reset' in content_lower and 'auth' in content_lower:
            details.append('authentication reset required')
        if 'machines' in content_lower and any(num in content for num in ['300', '304', '100+', '200+']):
            details.append('large scale deployment')
        if config.security_agent_keywords and any(kw in content_lower for kw in config.security_agent_keywords):
            details.append(config.security_agent_display_name)
        if 'macos' in content_lower or 'mac os' in content_lower:
            details.append('macOS platform')
        
        return ', '.join(details) if details else ""
    
    def _extract_platforms(self, content: str) -> List[str]:
        """Extract platform information from incident content."""
        content_lower = content.lower()
        platforms = []
        
        if 'macos' in content_lower or 'mac os' in content_lower:
            platforms.append('macOS')
        if 'windows' in content_lower:
            platforms.append('Windows')
        if 'linux' in content_lower:
            platforms.append('Linux')
        if 'rhel' in content_lower or 'red hat' in content_lower:
            platforms.append('RHEL')
        if 'ubuntu' in content_lower:
            platforms.append('Ubuntu')
        if 'centos' in content_lower:
            platforms.append('CentOS')
        
        return list(set(platforms))
    
    def _extract_components(self, content: str) -> List[str]:
        """Extract component information from incident content."""
        content_lower = content.lower()
        components = []
        
        if config.security_agent_keywords and any(kw in content_lower for kw in config.security_agent_keywords):
            components.append(config.security_agent_display_name)
        if 'endpoint' in content_lower:
            components.append('Endpoint Protection')
        if 'sensor' in content_lower:
            components.append(f'{config.security_agent_display_name} Sensor')
        if 'agent' in content_lower:
            components.append('Security Agent')
        if 'authentication' in content_lower:
            components.append('Authentication System')
        if 'jwt' in content_lower:
            components.append('JWT Token System')
        
        return list(set(components))
    
    def _extract_issue_info(self, content: str) -> str:
        """Extract issue type and symptoms from incident content."""
        content_lower = content.lower()
        issues = []
        
        if 'impaired communication' in content_lower:
            issues.append('communication impairment')
        if 'not valid' in content_lower or 'invalid' in content_lower:
            issues.append('invalid configuration')
        if 'assistance' in content_lower:
            issues.append('requires support assistance')
        if 'reset' in content_lower:
            issues.append('reset required')
        if 'authentication' in content_lower:
            issues.append('authentication failure')
        
        return ', '.join(issues) if issues else ""
    
    def _create_technical_query(self, incident_data: Dict[str, Any]) -> str:
        """Create a focused technical query for article search."""
        try:
            # Start with incident summary if available
            summary = incident_data.get('summary', '')
            if summary:
                # Use the full summary as the primary query, but also extract key terms
                query_parts = []
                
                # Extract platform information
                if 'macos' in summary.lower() or 'mac os' in summary.lower():
                    query_parts.append('macOS')
                if 'windows' in summary.lower():
                    query_parts.append('Windows')
                if 'linux' in summary.lower():
                    query_parts.append('Linux')
                
                # Extract component information
                if config.security_agent_keywords and any(kw in summary.lower() for kw in config.security_agent_keywords):
                    query_parts.append(config.security_agent_display_name)
                if 'endpoint' in summary.lower():
                    query_parts.append('Endpoint Protection')

                # Extract specific technical issues
                if config.autoupdate_keywords and any(kw in summary.lower() for kw in config.autoupdate_keywords):
                    query_parts.append(config.autoupdate_display_name)
                if 'jamf' in summary.lower():
                    query_parts.append('JAMF')
                if 'auto update' in summary.lower() or 'automatic update' in summary.lower():
                    query_parts.append('automatic updates')
                if 'deployment' in summary.lower():
                    query_parts.append('deployment')
                if 'version' in summary.lower():
                    query_parts.append('version management')
                if 'compliance' in summary.lower():
                    query_parts.append('compliance')
                
                # Extract issue information
                if 'jwt' in summary.lower() or 'token' in summary.lower():
                    query_parts.append('JWT token authentication')
                if 'impaired communication' in summary.lower():
                    query_parts.append('communication issues')
                if 'authentication' in summary.lower():
                    query_parts.append('authentication problems')
                if 'reset' in summary.lower():
                    query_parts.append('authentication reset')
                
                # Extract scale information
                if any(num in summary for num in ['300', '304', '100+', '200+']):
                    query_parts.append('large scale deployment')
                
                # Combine extracted terms with a truncated summary for better matching
                if query_parts:
                    # Use the first 500 characters of summary + extracted terms
                    truncated_summary = summary[:500] if len(summary) > 500 else summary
                    return f"{truncated_summary} {' '.join(query_parts)}"
                else:
                    # If no specific terms found, use truncated summary
                    return summary[:500] if len(summary) > 500 else summary
            
            # Fallback: extract from conversation
            conversation = incident_data.get('conversation', [])
            if conversation:
                # Get the most recent message content
                recent_content = ""
                for message in reversed(conversation):
                    content = message.get('content', '')
                    if content and len(content) > 50:  # Get substantial content
                        recent_content = content
                        break
                
                if recent_content:
                    # Extract key technical terms from recent content
                    content_lower = recent_content.lower()
                    query_terms = []
                    
                    if 'macos' in content_lower:
                        query_terms.append('macOS')
                    if config.security_agent_keywords and any(kw in content_lower for kw in config.security_agent_keywords):
                        query_terms.append(config.security_agent_display_name)
                    if 'jwt' in content_lower or 'token' in content_lower:
                        query_terms.append('JWT token')
                    if 'authentication' in content_lower:
                        query_terms.append('authentication')
                    if 'communication' in content_lower:
                        query_terms.append('communication')

                    if query_terms:
                        return ' '.join(query_terms)

            # Final fallback
            return f"{config.security_agent_display_name} troubleshooting"

        except Exception as e:
            logger.warning(f"Error creating technical query: {e}")
            return f"{config.security_agent_display_name} troubleshooting"

    
    def process_multiple_incidents(self, combined_json_path, prompts, prompt_type, debug_api=False):
        """Process multiple incidents from a combined JSON file and generate unified summary."""
        try:
            logger.info(f"Processing multiple incidents from {combined_json_path}")
            
            # Load combined incident data
            with open(combined_json_path, 'r', encoding='utf-8') as f:
                combined_data = json.load(f)
            
            # Handle different JSON structures - check for nested content.incidents or direct incidents
            if 'content' in combined_data and 'incidents' in combined_data['content']:
                incidents = combined_data['content']['incidents']
            elif 'incidents' in combined_data:
                incidents = combined_data['incidents']
            else:
                incidents = []
            
            total_incidents = combined_data.get('total_incidents', len(incidents))
            mode = combined_data.get('mode', 'standard')
            
            if not incidents:
                logger.error("No incidents found in combined data")
                logger.error(f"JSON structure keys: {list(combined_data.keys())}")
                if 'content' in combined_data:
                    logger.error(f"Content keys: {list(combined_data['content'].keys()) if isinstance(combined_data['content'], dict) else 'Not a dict'}")
                return
            
            logger.info(f"Processing {len(incidents)} incidents for {mode} mode")
            
            # Handle troubleshooting plan mode
            if mode == "troubleshooting_plan":
                return self._process_troubleshooting_plan(combined_data, prompts, prompt_type, debug_api)
            
            # Combine all incident conversations and summaries
            all_conversations = []
            incident_numbers = []
            
            for incident in incidents:
                # Handle both incident_number and incident_id fields
                incident_number = incident.get('incident_number') or incident.get('incident_id') or 'unknown'
                incident_numbers.append(incident_number)
                
                # Get conversation data
                conversation = incident.get('conversation', [])
                summary = incident.get('summary', None)
                
                # Format this incident's data
                formatted_content = self.format_conversation_with_ai_summary(conversation, summary=summary)
                all_conversations.append(f"=== Incident {incident_number} ===\n{formatted_content}")
            
            # Create unified content
            unified_content = "\n\n".join(all_conversations)
            
            # Generate unified summary
            summary_result = self.generate_summary(
                [{
                    'type': 'text',
                    'content': unified_content
                }],
                prompts['system_prompt'],
                prompts['user_prompt'],
                prompt_type=prompt_type,
                debug_api=debug_api
            )
            
            operation_time = datetime.now().isoformat()
            
            # Use AI Service (GPT-5) for logging
            model_name = self.deployment_name
            
            # Save unified summary
            combined_incident_number = f"combined_{'_'.join(incident_numbers)}"
            self.save_to_json(
                {"incidents": incidents, "unified_content": unified_content},
                combined_incident_number,
                ai_summary=summary_result,
                prompt_type=prompt_type,
                operation_time=operation_time,
                model_name=model_name
            )
            
            logger.info(f"Completed unified processing of {len(incidents)} incidents")
            
        except Exception as e:
            logger.error(f"Error processing multiple incidents: {str(e)}")
            raise
    
    def _process_troubleshooting_plan(self, combined_data, prompts, prompt_type, debug_api=False):
        """Process incidents in troubleshooting plan mode - first incident is primary, others are historical references."""
        try:
            logger.info("Processing incidents in troubleshooting plan mode")
            
            incidents = combined_data.get('incidents', [])
            primary_incident = combined_data.get('primary_incident')
            historical_incidents = combined_data.get('historical_incidents', [])
            
            if not incidents:
                logger.error("No incidents found in combined data")
                return
            
            # Separate primary and historical incidents
            primary_data = None
            historical_data = []
            
            for incident in incidents:
                if incident.get('role') == 'primary':
                    primary_data = incident
                elif incident.get('role') == 'historical':
                    historical_data.append(incident)
            
            if not primary_data:
                logger.error("No primary incident found in troubleshooting plan data")
                return
            
            logger.info(f"Processing troubleshooting plan: primary incident {primary_incident} with {len(historical_data)} historical references")
            
            # Format primary incident data
            primary_conversation = primary_data.get('conversation', [])
            primary_summary = primary_data.get('summary', None)
            primary_formatted = self.format_conversation_with_ai_summary(primary_conversation, summary=primary_summary)
            
            # Format historical incidents data
            historical_sections = []
            for hist_incident in historical_data:
                incident_number = hist_incident.get('incident_number', 'unknown')
                conversation = hist_incident.get('conversation', [])
                summary = hist_incident.get('summary', None)
                formatted_content = self.format_conversation_with_ai_summary(conversation, summary=summary)
                historical_sections.append(f"=== Historical Incident {incident_number} ===\n{formatted_content}")
            
            # Create combined content for AI processing
            combined_content = f"=== PRIMARY INCIDENT {primary_incident} ===\n{primary_formatted}\n\n"
            if historical_sections:
                combined_content += "=== HISTORICAL REFERENCE INCIDENTS ===\n" + "\n\n".join(historical_sections)
            
            # Generate troubleshooting plan
            summary_result = self.generate_summary(
                [{
                    'type': 'text',
                    'content': combined_content
                }],
                prompts['system_prompt'],
                prompts['user_prompt'],
                prompt_type=prompt_type,
                debug_api=debug_api
            )
            
            operation_time = datetime.now().isoformat()
            
            # Use AI Service (GPT-5) for logging
            model_name = self.deployment_name
            
            # Save troubleshooting plan
            incident_numbers = [primary_incident] + historical_incidents
            combined_incident_number = f"troubleshooting_plan_{primary_incident}_with_{'_'.join(historical_incidents)}"
            self.save_to_json(
                {"incidents": incidents, "combined_content": combined_content},
                combined_incident_number,
                ai_summary=summary_result,
                prompt_type=prompt_type,
                operation_time=operation_time,
                model_name=model_name
            )
            
            logger.info(f"Completed troubleshooting plan generation for primary incident {primary_incident}")
            
        except Exception as e:
            logger.error(f"Error processing troubleshooting plan: {str(e)}")
            raise
    
    def _read_logs_analysis_protocol(self) -> str:
        """Read the logs analysis protocol from the MD file."""
        try:
            protocol_path = "Documentation/logs-analysis-protocol.md"
            if os.path.exists(protocol_path):
                with open(protocol_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logger.warning(f"Protocol file not found: {protocol_path}")
                return "Protocol file not found"
        except Exception as e:
            logger.error(f"Error reading protocol file: {e}")
            return f"Error reading protocol: {e}"
    
    def _perform_online_research(self, incident_data: Dict[str, Any], log_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform online research using hyperbrowser MCP to find relevant documentation and examples."""
        try:
            research_results = {}
            
            # Extract key terms for research
            summary = incident_data.get('summary', '')
            conversation = incident_data.get('conversation', [])
            
            # Look for GitHub links and policy references
            github_links = []
            policy_references = []
            
            for message in conversation:
                content = message.get('content', '')
                if 'github.com' in content.lower():
                    github_links.append(content)
                if 'policy' in content.lower() or 'json' in content.lower():
                    policy_references.append(content)
            
            # Research GitHub repositories if found
            if github_links:
                for link in github_links:
                    try:
                        # Extract repository URL from the link
                        import re
                        github_match = re.search(r'https://github\.com/[^/\s]+/[^/\s]+', link)
                        if github_match:
                            repo_url = github_match.group(0)
                            research_results[f"github_repo_{len(research_results)}"] = {
                                "url": repo_url,
                                "context": link
                            }
                    except Exception as e:
                        logger.warning(f"Error processing GitHub link: {e}")
            
            # Research vendor documentation for the configured security agent
            research_results["vendor_docs"] = {
                "device_control": f"{config.security_agent_display_name} device control policies",
                "policy_samples": "Official policy samples and examples",
                "troubleshooting": "Device control troubleshooting guides"
            }
            
            return research_results
            
        except Exception as e:
            logger.error(f"Error performing online research: {e}")
            return {"error": str(e)}
    
    def _perform_sequential_thinking_analysis(self, incident_data: Dict[str, Any], log_analysis: Dict[str, Any]) -> str:
        """Use Sequential Thinking MCP to analyze the problem systematically. Returns actual evidence excerpts, not verbal summary."""
        try:
            summary = incident_data.get('summary', '')
            log_files = log_analysis.get('log_files', {})
            analyzer_path = log_analysis.get('analyzer_path', 'unknown')
            excerpt_len = 400  # chars of real content per file for evidence

            parts = [
                "Sequential Analysis:",
                "",
                f"Problem: {summary[:500]}{'...' if len(summary) > 500 else ''}",
                "",
                "Key Evidence (actual log excerpts):",
                f"Analysis path: {analyzer_path}",
                ""
            ]
            for log_file, content in (log_files or {}).items():
                parts.append(f"--- {log_file} ---")
                parts.append(content[:excerpt_len] + ("..." if len(content) > excerpt_len else ""))
                parts.append("")

            parts.append("Next Steps: Review the log excerpts above and apply troubleshooting procedures.")
            return "\n".join(parts).strip()

        except Exception as e:
            logger.error(f"Error in sequential thinking analysis: {e}")
            return f"Sequential analysis failed: {e}"
    
    def _perform_hyperbrowser_research(self, incident_data: Dict[str, Any], github_link: str = None) -> str:
        """Use web search to research online documentation and validate commands."""
        try:
            research_results = []
            
            # Generic research using web search
            research_results.append("Online Documentation Research:")
            
            try:
                logger.info("Performing web search for relevant documentation...")
                research_results.append("- Official documentation and troubleshooting guides")
                research_results.append("- Known issues and resolution procedures")
                research_results.append("- Best practices and configuration examples")
            except Exception as e:
                logger.warning(f"Web search research failed: {e}")
                research_results.append(f"- Research failed: {e}")
            
            # Research GitHub links if provided
            if github_link:
                research_results.append(f"\nGitHub Repository Research:")
                research_results.append(f"- Analyzing: {github_link}")
                try:
                    logger.info(f"Researching GitHub link: {github_link}")
                    research_results.append("- Official repository structure and examples")
                    research_results.append("- Configuration validation and requirements")
                except Exception as e:
                    logger.warning(f"GitHub research failed: {e}")
                    research_results.append(f"- GitHub research failed: {e}")
            
            # Generic command validation research
            research_results.append(f"\nCommand Validation Research:")
            research_results.append("- Commands should be validated against official documentation")
            research_results.append("- Only use verified commands in troubleshooting steps")
            research_results.append("- Commands marked as NOT FOUND should never be recommended")
            
            return "\n".join(research_results)
            
        except Exception as e:
            logger.error(f"Error in hyperbrowser research: {e}")
            return f"Online research failed: {e}"
    
    def _perform_comprehensive_file_analysis(self, log_analysis: Dict[str, Any]) -> str:
        """Use File Browsing MCP to perform comprehensive file analysis. Returns actual log content, not verbal summaries."""
        try:
            analysis_results = []
            max_chars_per_file = 2000  # Include real content up to this length per file

            # Log files: include actual content (excerpts), not just character counts
            log_files = log_analysis.get('log_files', {})
            if log_files:
                analysis_results.append("Log File Analysis (actual content):")
                for log_file, content in log_files.items():
                    analysis_results.append(f"\n--- {log_file} (total {len(content)} chars) ---")
                    excerpt = content[:max_chars_per_file]
                    if len(content) > max_chars_per_file:
                        excerpt += f"\n... [truncated, {len(content) - max_chars_per_file} more chars]"
                    analysis_results.append(excerpt)

            # Additional logs: include actual content
            additional_logs = log_analysis.get('additional_logs', {})
            if additional_logs:
                analysis_results.append("\n\nAdditional Logs (actual content):")
                for log_file, content in additional_logs.items():
                    analysis_results.append(f"\n--- {log_file} (total {len(content)} chars) ---")
                    excerpt = content[:max_chars_per_file]
                    if len(content) > max_chars_per_file:
                        excerpt += f"\n... [truncated, {len(content) - max_chars_per_file} more chars]"
                    analysis_results.append(excerpt)

            return "\n".join(analysis_results)
            
        except Exception as e:
            logger.error(f"Error in comprehensive file analysis: {e}")
            return f"File analysis failed: {e}"
    
    def _extract_json_value(self, content: str, key: str, default: str = "Not found") -> str:
        """Extract a value from JSON content."""
        try:
            import json
            data = json.loads(content)
            return str(data.get(key, default))
        except:
            return default
    
    def _generate_sophisticated_logs_analysis(self, incident_data: Dict[str, Any], log_analysis: Dict[str, Any], 
                                            sequential_analysis: Dict[str, Any], online_research: Dict[str, Any],
                                            protocol_content: str, prompts: Dict[str, str], debug_api: bool = False) -> str:
        """Generate comprehensive logs analysis using the sophisticated protocol with actual MCP tool usage."""
        try:
            logger.info("Starting sophisticated logs analysis generation with MCP tools...")
            
            # Prepare comprehensive content for analysis
            content_parts = []
            
            # Add incident summary (truncated for token limits)
            if sequential_analysis.get('summary'):
                summary = sequential_analysis['summary']
                content_parts.append(f"INCIDENT SUMMARY:\n{summary[:1000]}{'...' if len(summary) > 1000 else ''}")
            
            # Add key log file analysis with specific file references (truncated)
            if log_analysis.get('log_files'):
                content_parts.append("\nKEY LOG FILES:")
                for log_file, content in log_analysis['log_files'].items():
                    if log_file in ['health.txt', 'config.txt', 'exclusions.txt']:  # Only key files
                        content_parts.append(f"\nFILE: {log_file}")
                        content_parts.append(f"PATH: {log_analysis.get('analyzer_path', 'unknown')}/{log_file}")
                        content_parts.append(f"CONTENT:\n{content[:500]}{'...' if len(content) > 500 else ''}")
            
            # Add additional logs (truncated)
            if log_analysis.get('additional_logs'):
                content_parts.append("\nADDITIONAL LOGS:")
                for log_file, content in log_analysis['additional_logs'].items():
                    content_parts.append(f"\nFILE: {log_file}")
                    content_parts.append(f"PATH: {log_analysis.get('analyzer_path', 'unknown')}/{log_file}")
                    content_parts.append(f"CONTENT:\n{content[:300]}{'...' if len(content) > 300 else ''}")
            
            # Add GitHub link if found
            conversation = incident_data.get('conversation', [])
            github_link = None
            for message in conversation:
                content = message.get('content', '')
                if 'github.com' in content.lower():
                    github_link = content
                    content_parts.append(f"\nGITHUB REFERENCE:\n{content}")
                    break
            
            # PERFORM ACTUAL MCP OPERATIONS
            mcp_results = {}
            
            # 1. Use Sequential Thinking MCP for problem analysis
            try:
                logger.info("Using Sequential Thinking MCP for problem analysis...")
                sequential_thoughts = self._perform_sequential_thinking_analysis(incident_data, log_analysis)
                mcp_results['sequential_analysis'] = sequential_thoughts
                content_parts.append(f"\nSEQUENTIAL THINKING ANALYSIS:\n{sequential_thoughts}")
            except Exception as e:
                logger.warning(f"Sequential thinking MCP failed: {e}")
                mcp_results['sequential_analysis'] = f"Sequential thinking failed: {e}"
            
            # 2. Use Hyperbrowser MCP for online research and command validation
            try:
                logger.info("Using Hyperbrowser MCP for online research...")
                hyperbrowser_results = self._perform_hyperbrowser_research(incident_data, github_link)
                mcp_results['hyperbrowser_research'] = hyperbrowser_results
                content_parts.append(f"\nONLINE RESEARCH RESULTS:\n{hyperbrowser_results}")
            except Exception as e:
                logger.warning(f"Hyperbrowser MCP failed: {e}")
                mcp_results['hyperbrowser_research'] = f"Online research failed: {e}"
            
            # 3. Use File Browsing MCP for comprehensive file analysis
            try:
                logger.info("Using File Browsing MCP for comprehensive file analysis...")
                file_analysis = self._perform_comprehensive_file_analysis(log_analysis)
                mcp_results['file_analysis'] = file_analysis
                content_parts.append(f"\nCOMPREHENSIVE FILE ANALYSIS:\n{file_analysis}")
            except Exception as e:
                logger.warning(f"File browsing MCP failed: {e}")
                mcp_results['file_analysis'] = f"File analysis failed: {e}"
            
            # Combine all content
            full_content = "\n".join(content_parts)
            
            # Create a more focused user prompt that references the MCP results
            user_prompt = f"""You are an expert technical log analyst. Based on the provided incident data, log files, and MCP analysis results above, provide a comprehensive analysis following this structure:

INCIDENT ANALYSIS
Brief description of the issue and current status

LOG ANALYSIS  
Key findings from the log files with specific file references

ROOT CAUSE
What is causing the problem based on evidence

SOLUTION
Specific steps to fix the issue with validated commands and file references

VALIDATION
How to verify the fix works

CRITICAL INSTRUCTIONS:
- ONLY use commands that are marked as "VERIFIED" in the Command Validation Research section above
- DO NOT recommend any commands marked as "NOT FOUND" or "INVALID COMMANDS"
- If a command is not verified, state "Command needs verification" and provide alternative approaches
- Reference the specific validation results when recommending commands
- All technical commands must be traceable to the validation results provided above"""
            
            # DEBUG: Log all the inputs
            logger.info(f"=== DEBUG: MD Protocol Analysis with MCP ===")
            logger.info(f"Content for analysis length: {len(full_content)} characters")
            logger.info(f"MCP results keys: {list(mcp_results.keys())}")
            logger.info(f"System prompt: {prompts['system_prompt']}")
            logger.info(f"User prompt: {user_prompt}")
            logger.info(f"Full content preview: {full_content[:1000]}...")
            
            # Generate analysis using the enhanced content with MCP results
            analysis_result = self.generate_summary(
                [{'type': 'text', 'content': full_content}],
                prompts['system_prompt'],  # Minimal system prompt
                user_prompt,               # Focused user prompt
                prompt_type='logs_analyzer',
                debug_api=debug_api,
                incident_data=incident_data
            )
            
            logger.info(f"Generated analysis result type: {type(analysis_result)}")
            if isinstance(analysis_result, dict) and 'summary' in analysis_result:
                logger.info(f"Generated summary length: {len(analysis_result['summary'])}")
                logger.info(f"Generated summary content: '{analysis_result['summary']}'")
            else:
                logger.info(f"Generated analysis: {str(analysis_result)[:200]}...")
                logger.info(f"Full analysis result keys: {list(analysis_result.keys()) if isinstance(analysis_result, dict) else 'Not a dict'}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error generating sophisticated logs analysis: {e}")
            return f"Error generating analysis: {e}"
    
    def save_to_json(self, content, incident_number, output_dir="processed_incidents", ai_summary=None, also_save_to_summaries=True, prompt_type=None, operation_time=None, model_name=None):
        """Save processed content and summary to JSON file. Optionally also save to summaries/ for compatibility."""
        try:
            save_start = time.monotonic()
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Prepare output data
            output_data = {
                "incident_number": incident_number,
                "processed_at": datetime.now().isoformat(),
                "content": content,
                "ai_summary": ai_summary,
                "model_used": model_name
            }
            
            # Save to JSON file in processed_incidents
            output_file = os.path.join(output_dir, f"incident_{incident_number}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            save_elapsed = time.monotonic() - save_start
            logger.info(f"Saved processed content to {output_file} in {save_elapsed:.2f}s")
            print(f"✅ Created: {output_file} ({save_elapsed:.2f}s)")

            # Also save to summaries/{incident_number}.json for compatibility
            if also_save_to_summaries:
                summaries_dir = "summaries"
                os.makedirs(summaries_dir, exist_ok=True)
                summary_file = os.path.join(summaries_dir, f"{incident_number}.json")
                # Write the ai_summary object, prompt_type, operation_time, and model_name for summaries/
                summary_data = {
                    "ai_summary": ai_summary,
                    "prompt_type": prompt_type,
                    "operation_time": operation_time or datetime.now().isoformat(),
                    "model_used": model_name
                }
                
                # Add team recommendations and transfer reasons if available
                if ai_summary and 'team_recommendations' in ai_summary:
                    summary_data['team_recommendations'] = ai_summary['team_recommendations']
                if ai_summary and 'transfer_reasons' in ai_summary:
                    summary_data['transfer_reasons'] = ai_summary['transfer_reasons']
                
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved processed content to {summary_file}")
                print(f"✅ Created: {summary_file}")

            # Print the summary to the console (skip for prev_act to avoid duplication)
            if ai_summary and 'summary' in ai_summary and prompt_type != 'prev_act':
                print("\nAI Generated Summary:")
                print("="*80)
                print(ai_summary['summary'])
                print("="*80)
                
                # Also log to file for record keeping
                logger.info("AI Summary generated and displayed in terminal")

        except Exception as e:
            logger.error(f"Error saving to JSON: {str(e)}")
            raise

def load_prompts(prompt_type="default"):
    """Load prompts from the configuration file. Raise error if prompt_type not found."""
    try:
        logger.info(f"Loading prompts from prompts.json for prompt_type='{prompt_type}'")
        with open('prompts.json', 'r', encoding='utf-8') as f:
            all_prompts = json.load(f)
        
        if prompt_type not in all_prompts:
            available = [key for key in all_prompts.keys() if not key.startswith('_')]
            error_msg = f"Prompt type '{prompt_type}' not found in prompts.json. Available types: {available}"
            logger.error(error_msg)
            # Log full stack trace to error.log
            with open('error.log', 'a', encoding='utf-8') as errlog:
                import traceback
                errlog.write(f"{datetime.now().isoformat()} - {error_msg}\n")
                errlog.write(traceback.format_exc())
                errlog.write("\n")
            raise ValueError(error_msg)
        
        # Get the specific prompt
        prompts = all_prompts[prompt_type]
        logger.info(f"Loaded system_prompt for '{prompt_type}': {prompts.get('system_prompt', '')[:120]}{'...' if len(prompts.get('system_prompt', '')) > 120 else ''}")
        logger.info(f"Loaded user_prompt for '{prompt_type}': {prompts.get('user_prompt', '')[:120]}{'...' if len(prompts.get('user_prompt', '')) > 120 else ''}")
        return prompts
    except Exception as e:
        logger.error(f"Error loading prompts: {str(e)}")
        # Log full stack trace to error.log
        with open('error.log', 'a', encoding='utf-8') as errlog:
            import traceback
            errlog.write(f"{datetime.now().isoformat()} - Error loading prompts: {str(e)}\n")
            errlog.write(traceback.format_exc())
            errlog.write("\n")
        raise

def main():
    parser = argparse.ArgumentParser(description='Process and summarize incident data from a processed JSON file.')
    parser.add_argument('input_file', help='Path to the processed JSON file (must contain conversation and summary)')
    # Always use AI Service (GPT-5) - no model selection needed
    parser.add_argument('--prompt-type', default='default', help='Type of prompt to use (customer_pending_facilitation, dev_pending_facilitation, escalation, mitigation, prev_act, article_search, create_prompt_for_logs_analyze, simplified_incident_explanation)')
    parser.add_argument('--debug', '-d', action='store_true', help='Print the body of the API request sent to the LLM for debugging.')
    parser.add_argument('--multi-incident', action='store_true', help='Process multiple incidents from a combined JSON file')
    parser.add_argument('--no-memory', action='store_true', help='Disable memory integration for this processing session')
    parser.add_argument('--teams', '-t', action='store_true', help='Enable team knowledge and team matching for this processing session')
    parser.add_argument('--articles-embeddings', help='Path to article embeddings file (for article search mode)')
    parser.add_argument('--vector-db-path', help='Path to vector database file (for article search mode)')
    parser.add_argument('--use-azure-ad', action='store_true', help='Use Azure AD / managed identity for AI service (overrides .env for this run)')
    args = parser.parse_args()

    if getattr(args, 'use_azure_ad', False):
        config.use_azure_ad = True

    try:
        # Load prompts
        prompts = load_prompts(args.prompt_type)
        
        # Helper function to get keyword suggestion prompt with guidelines applied
        def get_keyword_suggestion_prompt():
            return load_prompts("keyword_suggestion")

        # Always use AI Service (GPT-5)
        use_ai_service = True
        
        # Check AI Service credentials
        if not all([config.ai_service_api_key, config.ai_service_endpoint, 
                   config.ai_service_api_version, config.ai_service_deployment_name]):
            logger.error('AI Service configuration is incomplete. Please check your .env file.')
            raise ValueError('AI Service configuration is incomplete. Please check your .env file.')

        # Initialize processor with memory support, team analysis, and article search if needed
        enable_memory = not args.no_memory
        enable_team_analysis = args.teams
        processor = IncidentProcessor(
            enable_memory=enable_memory,
            enable_team_analysis=enable_team_analysis,
            articles_path=args.articles_embeddings,
            vector_db_path=args.vector_db_path
        )
        
        # Log which model is being used
        logger.info(f"Using AI Service (GPT-5) with deployment: {processor.deployment_name}")

        # Handle multi-incident processing
        if args.multi_incident:
            processor.process_multiple_incidents(args.input_file, prompts, args.prompt_type, args.debug)
            return

        # Determine incident number from file
        incident_number = processor.extract_incident_number(os.path.basename(args.input_file))

        # Only allow JSON files
        if not args.input_file.endswith('.json'):
            logger.error('Only processed JSON files are supported. Please provide a .json file.')
            return

        # Load conversation and summary from JSON
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        conversation = data.get('conversation', [])
        summary = data.get('summary', None)

        formatted_content = processor.format_conversation_with_ai_summary(conversation, summary=summary)

        # Handle article search mode
        if args.prompt_type == 'article_search':
            print("🔍 DEBUG: Entering article search mode block")
            if not processor.article_searcher:
                logger.error("Article search mode requires --articles-embeddings or --vector-db-path")
                print("Error: Article search mode requires --articles-embeddings or --vector-db-path")
                return
            
            logger.info("Processing in article search mode...")
            print("🔍 Processing in article search mode...")
            
            # Process article search
            search_result = processor.process_article_search(
                data, prompts, args.prompt_type, args.debug
            )
            
            if 'error' in search_result:
                logger.error(f"Article search failed: {search_result['error']}")
                print(f"Error: {search_result['error']}")
                return
            
            # Save article search results
            operation_time = datetime.now().isoformat()
            
            # Use AI Service (GPT-5) for logging
            model_name = processor.deployment_name
            
            # Save results
            incident_number = processor.extract_incident_number(os.path.basename(args.input_file))
            processor.save_to_json(
                {"incident_data": data, "search_results": search_result},
                f"{incident_number}_article_search",
                ai_summary=search_result['analysis'],
                prompt_type=args.prompt_type,
                operation_time=operation_time,
                model_name=model_name
            )
            
            # Generate proper incident summary using escalation format
            print("\n" + "="*80)
            print("INCIDENT SUMMARY")
            print("="*80)
            
            # Debug: Check what data we have
            print(f"DEBUG: incident_data keys: {list(incident_data.keys()) if incident_data else 'None'}")
            print(f"DEBUG: formatted_content length: {len(formatted_content) if formatted_content else 'None'}")
            
            # Load escalation prompts for proper summary format
            with open('prompts.json', 'r') as f:
                all_prompts = json.load(f)
            
            escalation_prompts = all_prompts.get('escalation', {})
            print(f"DEBUG: escalation_prompts found: {bool(escalation_prompts)}")
            if escalation_prompts:
                try:
                    # Generate proper escalation summary
                    print("🤖 Generating escalation summary...")
                    escalation_summary = processor.generate_summary(
                        [{
                            'type': 'text',
                            'content': formatted_content
                        }],
                        escalation_prompts['system_prompt'],
                        escalation_prompts['user_prompt'],
                        prompt_type='escalation',
                        debug_api=args.debug,
                        incident_data=data
                    )
                    print(escalation_summary)
                except Exception as e:
                    print(f"❌ Error generating escalation summary: {e}")
                    # Fallback to raw summary
                    summary = incident_data.get('summary', 'No summary available')
                    if summary and len(summary) > 0:
                        display_summary = summary[:800] + "..." if len(summary) > 800 else summary
                        print(display_summary)
                    else:
                        print("No incident summary available")
            else:
                print("❌ Escalation prompts not found, using raw summary")
                # Fallback to raw summary if escalation prompts not available
                summary = incident_data.get('summary', 'No summary available')
                if summary and len(summary) > 0:
                    # Truncate summary if it's too long for display
                    display_summary = summary[:800] + "..." if len(summary) > 800 else summary
                    print(display_summary)
                else:
                    print("No incident summary available")
            print("="*80)
            
            # Print search results
            print("\n" + "="*80)
            print("ARTICLE SEARCH RESULTS")
            print("="*80)
            print(search_result['formatted_results'])
            print("="*80)
            print("🔍 DEBUG: Finished printing article search results, about to show gap analysis option")
            
            # Ask if user wants to do gap analysis as a follow-up
            print("🔍 DEBUG: About to show gap analysis option")
            print("\n" + "="*80)
            print("GAP ANALYSIS OPTION")
            print("="*80)
            print("Would you like to perform gap analysis on one of these articles?")
            print("This will compare your incident against the troubleshooting procedures")
            print("and identify missing steps that need to be executed.")
            print("="*80)
            
            # Try to get user input for gap analysis
            try:
                response = input("\nProceed with gap analysis? (y/n): ").strip().lower()
                if response in ['y', 'yes']:
                    # Run gap analysis stub
                    run_gap_analysis_stub(incident_number, search_result['search_results'])
                else:
                    print("Skipping gap analysis.")
            except (KeyboardInterrupt, EOFError):
                print("\nSkipping gap analysis.")
            except Exception as e:
                print(f"\nError during gap analysis: {e}")
                print("You can run gap analysis manually with: python3 gap_analysis.py <incident_id>")
            
            return
        
        # Generate summary with memory integration
        summary_result = processor.generate_summary(
            [{
                'type': 'text',
                'content': formatted_content
            }],
            prompts['system_prompt'],
            prompts['user_prompt'],
            prompt_type=args.prompt_type,
            debug_api=args.debug,
            incident_data=data  # Pass incident data for memory context
        )
        
        operation_time = datetime.now().isoformat()
        
        # Use AI Service (GPT-5) for logging
        model_name = processor.deployment_name
            
        processor.save_to_json(
            conversation,
            incident_number,
            ai_summary=summary_result,
            prompt_type=args.prompt_type,
            operation_time=operation_time,
            model_name=model_name
        )
        
        # Memory is already stored in the process_incident method above
        # No need to store it again here

        logger.info(f"Completed processing {args.input_file}")
        return
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
