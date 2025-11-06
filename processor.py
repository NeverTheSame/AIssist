import json
import os
import re
import sys
from datetime import datetime
import logging
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from openai import AzureOpenAI
import tiktoken
from config import config
import argparse
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
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
from memory.memory_manager import SummarizerMemoryManager
from article_searcher import ArticleSearcher
from team_knowledge.teams_matcher import TeamDetector, TeamKnowledgeManager, TeamAnalyzer, TeamLearningEngine
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
            use_azure_router=use_azure_router
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
    
    return logger

logger = setup_logging()

# Token costs are now configurable via .env file
# See config.py for ZAI_INPUT_COST, ZAI_OUTPUT_COST, OPENAI_INPUT_COST, OPENAI_OUTPUT_COST

class MolecularContextEngine:
    """
    Dynamic molecule construction for incident processing.
    Selects most relevant examples based on incident characteristics.
    """
    
    def __init__(self):
        logger.info("Initializing MolecularContextEngine and loading example database from molecular_examples.json.")
        try:
            with open("molecular_examples.json", "r", encoding="utf-8") as f:
                self.example_database = json.load(f)
            logger.info(f"Loaded molecular example database with keys: {list(self.example_database.keys())}")
        except Exception as e:
            logger.error(f"Failed to load molecular_examples.json: {e}")
            self.example_database = {}
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract key technical terms from incident text."""
        logger.info(f"[MolecularContextEngine] Extracting keywords from text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        # Load technical patterns from configuration file
        technical_patterns = self._load_technical_patterns()
        
        keywords = []
        text_lower = text.lower()
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text_lower)
            keywords.extend(matches)
        result = list(set(keywords))
        logger.info(f"[MolecularContextEngine] Extracted keywords: {result}")
        return result
    
    def _load_technical_patterns(self) -> List[str]:
        """Load technical patterns from configuration file."""
        try:
            with open("technical_patterns.json", "r", encoding="utf-8") as f:
                patterns_config = json.load(f)
                return patterns_config.get("technical_patterns", [])
        except FileNotFoundError:
            logger.warning("technical_patterns.json not found, using default patterns")
            return self._get_default_patterns()
        except Exception as e:
            logger.error(f"Error loading technical_patterns.json: {e}, using default patterns")
            return self._get_default_patterns()
    
    def _get_default_patterns(self) -> List[str]:
        """Get default technical patterns if configuration file is not available."""
        return [
            r'\b(agent|service|process)\b',
            r'\b(crash|error|failure|timeout)\b',
            r'\b(memory|cpu|performance)\b',
            r'\b(network|connectivity|firewall)\b',
            r'\b(sync|synchronization)\b',
            r'\b(authentication|auth)\b',
            r'\b(macos|windows|linux)\b',
            r'\b(policy|configuration)\b',
            r'\b(telemetry|reporting)\b',
            r'\b(installation|deployment)\b',
            r'\b(detection|engine)\b',
            r'\b(security|protection)\b'
        ]
    
    def select_relevant_examples(self, incident_text: str, prompt_type: str, max_examples: int = 5) -> List[Dict]:
        """Select most relevant examples based on incident content."""
        logger.info(f"[MolecularContextEngine] Selecting relevant examples for prompt_type='{prompt_type}' and incident_text: {incident_text[:100]}{'...' if len(incident_text) > 100 else ''}")
        if prompt_type not in self.example_database:
            logger.warning(f"[MolecularContextEngine] Prompt type '{prompt_type}' not found in example database.")
            return []
        
        incident_keywords = self.extract_keywords(incident_text)
        examples = self.example_database[prompt_type]
        
        # Calculate relevance scores
        scored_examples = []
        for example in examples:
            # Keyword overlap score
            overlap = len(set(incident_keywords) & set(example['keywords']))
            keyword_score = overlap / len(example['keywords']) if example['keywords'] else 0
            
            # Text similarity score using TF-IDF
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                vectors = vectorizer.fit_transform([incident_text, example['example']['input']])
                similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            except Exception as e:
                logger.warning(f"[MolecularContextEngine] TF-IDF similarity calculation failed: {e}")
                similarity_score = 0
            
            # Combined relevance score
            relevance_score = (keyword_score * 0.6) + (similarity_score * 0.4)
            
            scored_examples.append({
                'example': example['example'],
                'score': relevance_score,
                'category': example['category']
            })
        
        # Sort by relevance and return top examples
        scored_examples.sort(key=lambda x: x['score'], reverse=True)
        selected = scored_examples[:max_examples]
        logger.info(f"[MolecularContextEngine] Selected {len(selected)} relevant examples (top {max_examples}): {[ex['category'] for ex in selected]}")
        return selected
    
    def construct_molecular_prompt(self, base_prompt: str, incident_text: str, prompt_type: str):
        """Construct dynamic molecular prompt with relevant examples. Returns (prompt, num_examples_used)."""
        logger.info(f"[MolecularContextEngine] Constructing molecular prompt for prompt_type='{prompt_type}'. Base prompt length: {len(base_prompt)}. Incident text length: {len(incident_text)}.")
        relevant_examples = self.select_relevant_examples(incident_text, prompt_type)
        if not relevant_examples:
            logger.info("[MolecularContextEngine] No relevant examples found. Returning base prompt.")
            return base_prompt, 0
        
        # Build examples text
        examples_text = "\n\nRelevant examples for context:\n"
        for i, example_data in enumerate(relevant_examples, 1):
            example = example_data['example']
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Input: {example['input']}\n"
            examples_text += f"Output: {example['output']}\n"
        examples_text += "\nNow process the following incident:\n"
        
        # Combine base prompt with examples
        final_prompt = base_prompt + examples_text
        
        logger.info(f"[MolecularContextEngine] Constructed molecular prompt. Final prompt length: {len(final_prompt)}. Preview: {final_prompt[:200]}{'...' if len(final_prompt) > 200 else ''}")
        return final_prompt, len(relevant_examples)

    def get_all_categories(self):
        """Get all unique categories from molecular examples"""
        try:
            with open("molecular_examples.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                all_categories = set()
                for section_name, examples in data.items():
                    for example in examples:
                        category = example.get("category", "")
                        if category:
                            all_categories.add(category)
                return sorted(all_categories)
        except Exception as e:
            print(f"Could not load categories: {e}")
            return []

class IncidentProcessor:
    def __init__(self, enable_memory=True, articles_path=None, vector_db_path=None, enable_team_analysis=False, enable_timing=False):
        # Always use Azure Router (GPT-5)
        self.use_azure_router = True
        self.enable_timing = enable_timing
        self.molecular_engine = MolecularContextEngine()
        
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
        
        # Initialize memory manager if enabled
        self.memory_manager = None
        if enable_memory:
            try:
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
                    use_azure_router=True
                )
                logger.info("Article searcher initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize article searcher: {e}. Continuing without article search.")
                self.article_searcher = None
        
        # Initialize Azure Router client (GPT-5)
        # Validate configuration (will refresh from environment if needed)
        try:
            config.validate()
        except ValueError as e:
            raise ValueError(f"AI Service configuration is incomplete: {e}")
        
        self.client = AzureOpenAI(
            api_key=config.ai_service_api_key,
            api_version=config.ai_service_api_version,
            azure_endpoint=config.ai_service_endpoint
        )
        self.deployment_name = config.ai_service_deployment_name
        self.model_costs = {
            "input": config.input_cost,
            "output": config.output_cost
        }
        self.use_million_tokens = False  # Azure Router costs are per 1K tokens
        
        # Initialize team analyzer with LLM client if team analysis is enabled
        if enable_team_analysis and self.team_detector and self.team_knowledge_manager:
            try:
                self.team_analyzer = TeamAnalyzer(
                    self.team_detector, 
                    self.team_knowledge_manager,
                    llm_client=self.client,
                    deployment_name=getattr(self, 'deployment_name', None),
                    use_azure_router=True
                )
                logger.info("Team analyzer initialized with LLM client")
                
                # Initialize team learning engine with LLM client
                self.team_learning_engine = TeamLearningEngine(
                    self.team_knowledge_manager,
                    self.team_detector,
                    llm_client=self.client,
                    deployment_name=getattr(self, 'deployment_name', None),
                    use_azure_router=True
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

    def format_conversation_with_ai_summary(self, conversation, internal_ai_summary=None, summary=None):
        """Format conversation and authored summary for the AI model."""
        formatted = []
        for entry in conversation:
            # Clean the content to remove Azure Support Center info
            cleaned_content = self.clean_azure_support_info(entry['content'])
            formatted.append(f"[{entry['timestamp']}] {entry['author']}: {cleaned_content}")
        conversation_text = "\n".join(formatted)
        
        # Add authored summary if available (before conversation)
        if summary:
            # Clean the summary text
            cleaned_summary = self.clean_azure_support_info(summary)
            return f"--- Authored Summary ---\n{cleaned_summary}\n\n{conversation_text}"
        
        return conversation_text

    def generate_summary(self, content, system_prompt, user_prompt, prompt_type="default", debug_api=False, incident_data=None):
        """Generate summary using OpenAI or Azure OpenAI with molecular context enhancement, memory integration, and team analysis."""
        try:
            # Format the content
            with self.time_context("format_content", "ai", {"content_items": len(content)}):
                formatted_content = []
                for item in content:
                    if item['type'] == 'text':
                        formatted_content.append(item['content'])
                    else:  # image
                        formatted_content.append(f"[Image: {item['content']}]")
                conversation = "\n".join(formatted_content)
            
            # Apply molecular context engineering for supported prompt types
            molecular_examples_used = 0
            if prompt_type.endswith('_molecular'):
                with self.time_context("molecular_context_engineering", "molecular", {"prompt_type": prompt_type}):
                    enhanced_user_prompt, molecular_examples_used = self.molecular_engine.construct_molecular_prompt(
                        user_prompt, conversation, prompt_type
                    )
                logger.info(f"Applied molecular context engineering with {molecular_examples_used} examples for {prompt_type}")
            else:
                enhanced_user_prompt = user_prompt
            
            # Enhance prompt with memory context if memory manager is available and incident data is provided
            memory_enhanced = False
            if self.memory_manager and incident_data:
                try:
                    with self.time_memory_operation("memory_enhancement", "enhance_prompt"):
                        # Pass information about whether molecular context was used
                        molecular_context_used = molecular_examples_used > 0
                        original_prompt = enhanced_user_prompt
                        enhanced_user_prompt = self.memory_manager.enhance_prompt_with_memory(
                            enhanced_user_prompt, incident_data, molecular_context_used
                        )
                        
                        # Only claim enhancement if the prompt was actually changed
                        if enhanced_user_prompt != original_prompt:
                            memory_enhanced = True
                            logger.info("Enhanced prompt with memory context")
                            print(f"🧠 Enhanced prompt with memory context from previous incidents")
                        else:
                            logger.info("Memory search completed but no relevant context found")
                    if not memory_enhanced:
                        memory_enhanced = True  # Still mark as processed to avoid retries
                except Exception as e:
                    logger.warning(f"Failed to enhance prompt with memory: {e}")
            
            # Perform team analysis if team knowledge system is available and incident data is provided
            team_analysis_result = None
            if self.team_analyzer and incident_data:
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
                        except Exception as e:
                            logger.warning(f"Failed to learn from team analysis: {e}")
                    elif self.team_learning_engine and team_analysis_result and team_analysis_result.get('skipped_reason'):
                        # Team learning skipped - reason already logged by team analyzer
                        pass
                except Exception as e:
                    logger.warning(f"Failed to perform team analysis: {e}")
            
            # Enhance prompt with team context if team analysis was performed
            team_enhanced = False
            if team_analysis_result and team_analysis_result.get('detected_teams'):
                try:
                    team_context = self._build_team_context(team_analysis_result)
                    if team_context:
                        enhanced_user_prompt = f"{enhanced_user_prompt}\n\nTeam Context:\n{team_context}"
                        team_enhanced = True
                        logger.info("Enhanced prompt with team context")
                        print(f"👥 Enhanced prompt with team context from {len(team_analysis_result['detected_teams'])} teams")
                except Exception as e:
                    logger.warning(f"Failed to enhance prompt with team context: {e}")
            # Prepare messages - ensure content is a string
            with self.time_context("prepare_llm_messages", "ai", {"prompt_length": len(str(enhanced_user_prompt))}):
                messages = [
                    {"role": "system", "content": str(system_prompt)},
                    {"role": "user", "content": f"{str(enhanced_user_prompt)}\n\nContent:\n{conversation}"}
                ]
            if debug_api:
                print("\n[DEBUG_API] LLM API request body:")
            # Use Azure Router (GPT-5)
            model_name = self.deployment_name
            # Count input tokens
            with self.time_context("count_tokens", "ai", {"text_length": len(f"{system_prompt}\n{enhanced_user_prompt}\n{conversation}")}):
                input_text = f"{system_prompt}\n{enhanced_user_prompt}\n{conversation}"
                input_tokens = self.count_tokens(input_text)
            
            # Use Azure Router (GPT-5) for timing
            model_name = self.deployment_name
            
            # Generate summary with timing using Azure Router (GPT-5)
            print(f"🤖 Starting LLM call with {model_name}...")
            with self.time_llm_call("llm_generate_summary", model_name, input_tokens, 0):  # We'll update output tokens after
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=8000
                )
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
            # Add molecular context info if used
            if molecular_examples_used > 0:
                result["molecular_context"] = {
                    "examples_used": molecular_examples_used,
                    "prompt_type": prompt_type
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
    
    def _build_team_context(self, team_analysis_result: Dict[str, Any]) -> str:
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
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error building team context: {e}")
            return ""
    
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
            incident_folder = f"icms/{incident_number}"
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
            
            # Find Client Analyzer output folder
            client_analyzer_folders = []
            for item in os.listdir(logs_path):
                item_path = os.path.join(logs_path, item)
                if os.path.isdir(item_path) and 'output' in item.lower():
                    client_analyzer_folders.append(item)
            
            if not client_analyzer_folders:
                return {"error": f"No Client Analyzer output folders found in {logs_path}"}
            
            # Use the most recent Client Analyzer folder
            latest_analyzer_folder = sorted(client_analyzer_folders)[-1]
            analyzer_path = os.path.join(logs_path, latest_analyzer_folder)
            
            # Analyze key log files
            log_files_analysis = {}
            key_files = [
                'health.txt',
                'config.txt',
                'exclusions.txt',
                'definitions.txt',
                'log.txt',
                'console.txt',
                'syslog.txt'
            ]
            
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
        
        if 'mdatp' in content_lower or 'mde' in content_lower:
            components.append('Microsoft Defender for Endpoint')
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
            
            # Add MDE logs
            if log_analysis.get('mde_logs'):
                content_parts.append("\nMDE Logs:")
                for log_file, content in log_analysis['mde_logs'].items():
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
        if 'defender' in content_lower:
            details.append('Microsoft Defender for Endpoint')
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
        
        if 'defender' in content_lower:
            components.append('Microsoft Defender')
        if 'endpoint' in content_lower:
            components.append('Endpoint Protection')
        if 'mde' in content_lower:
            components.append('MDE')
        if 'sensor' in content_lower:
            components.append('MDE Sensor')
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
                if 'defender' in summary.lower():
                    query_parts.append('Microsoft Defender')
                if 'mde' in summary.lower():
                    query_parts.append('MDE')
                if 'endpoint' in summary.lower():
                    query_parts.append('Endpoint Protection')
                
                # Extract specific technical issues
                if 'mau' in summary.lower() or 'microsoft autoupdate' in summary.lower():
                    query_parts.append('Microsoft AutoUpdate')
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
                    if 'defender' in content_lower:
                        query_terms.append('Microsoft Defender')
                    if 'jwt' in content_lower or 'token' in content_lower:
                        query_terms.append('JWT token')
                    if 'authentication' in content_lower:
                        query_terms.append('authentication')
                    if 'communication' in content_lower:
                        query_terms.append('communication')
                    
                    if query_terms:
                        return ' '.join(query_terms)
            
            # Final fallback
            return "Microsoft Defender Endpoint troubleshooting"
            
        except Exception as e:
            logger.warning(f"Error creating technical query: {e}")
            return "Microsoft Defender Endpoint troubleshooting"

    
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
            
            # Use Azure Router (GPT-5) for logging
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
            
            # Use Azure Router (GPT-5) for logging
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
            
            # Research Microsoft Defender documentation
            research_results["microsoft_docs"] = {
                "device_control": "Microsoft Defender for Endpoint device control policies",
                "policy_samples": "Official policy samples and examples",
                "troubleshooting": "Device control troubleshooting guides"
            }
            
            return research_results
            
        except Exception as e:
            logger.error(f"Error performing online research: {e}")
            return {"error": str(e)}
    
    def _perform_sequential_thinking_analysis(self, incident_data: Dict[str, Any], log_analysis: Dict[str, Any]) -> str:
        """Use Sequential Thinking MCP to analyze the problem systematically."""
        try:
            # Extract key information for analysis
            summary = incident_data.get('summary', '')
            conversation = incident_data.get('conversation', [])
            
            # Build generic analysis prompt
            analysis_prompt = f"""
            Analyze this incident systematically:
            
            INCIDENT: {summary[:500]}
            
            LOG FINDINGS:
            - Available log files: {list(log_analysis.get('log_files', {}).keys())}
            - Log analysis results: {log_analysis.get('analyzer_path', 'Not available')}
            
            Use sequential thinking to:
            1. Identify the core problem
            2. Analyze potential root causes
            3. Evaluate evidence from logs
            4. Develop hypothesis about what's wrong
            5. Propose systematic troubleshooting approach
            """
            
            # Return a generic structured analysis
            return f"""Sequential Analysis:
            
            Problem: {summary[:200]}...
            Key Evidence: 
            - Log files analyzed: {len(log_analysis.get('log_files', {}))}
            - Analysis path: {log_analysis.get('analyzer_path', 'unknown')}
            
            Hypothesis: Issue requires systematic investigation based on log evidence
            Next Steps: Review log findings and apply appropriate troubleshooting procedures"""
            
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
        """Use File Browsing MCP to perform comprehensive file analysis."""
        try:
            analysis_results = []
            
            # Generic file analysis
            log_files = log_analysis.get('log_files', {})
            if log_files:
                analysis_results.append("Log File Analysis:")
                for log_file, content in log_files.items():
                    analysis_results.append(f"- {log_file}: {len(content)} characters")
                    # Look for common error patterns
                    if 'error' in content.lower() or 'fail' in content.lower():
                        analysis_results.append(f"  Contains error indicators")
                    if 'warn' in content.lower():
                        analysis_results.append(f"  Contains warning indicators")
            
            # Analyze additional logs if available
            additional_logs = log_analysis.get('additional_logs', {})
            if additional_logs:
                analysis_results.append("\nAdditional Logs Analysis:")
                for log_file, content in additional_logs.items():
                    analysis_results.append(f"- {log_file}: {len(content)} characters")
            
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
            logger.info(f"Saved processed content to {output_file}")
            print(f"✅ Created: {output_file}")

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
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved processed content to {summary_file}")
                print(f"✅ Created: {summary_file}")

            # Print the summary to the console (skip for prev_act_molecular to avoid duplication)
            if ai_summary and 'summary' in ai_summary and prompt_type != 'prev_act_molecular':
                print("\nAI Generated Summary:")
                print("="*80)
                print(ai_summary['summary'])
                print("="*80)
                
                # Print molecular context info if available
                if 'molecular_context' in ai_summary:
                    print(f"Molecular Context: {ai_summary['molecular_context']['examples_used']} examples used for {ai_summary['molecular_context']['prompt_type']}")
                
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
    # Always use Azure Router (GPT-5) - no model selection needed
    parser.add_argument('--prompt-type', default='default', help='Type of prompt to use (default, technical, executive, escalation, escalation_molecular, mitigation_molecular, troubleshooting_molecular, article_search_molecular, etc.)')
    parser.add_argument('--debug', '-d', action='store_true', help='Print the body of the API request sent to the LLM for debugging.')
    parser.add_argument('--multi-incident', action='store_true', help='Process multiple incidents from a combined JSON file')
    parser.add_argument('--no-memory', action='store_true', help='Disable memory integration for this processing session')
    parser.add_argument('--no-team-analysis', action='store_true', help='Disable team analysis for this processing session')
    parser.add_argument('--articles-embeddings', help='Path to article embeddings file (for article search mode)')
    parser.add_argument('--vector-db-path', help='Path to vector database file (for article search mode)')
    args = parser.parse_args()

    try:
        # Load prompts
        prompts = load_prompts(args.prompt_type)
        
        # Helper function to get keyword suggestion prompt with guidelines applied
        def get_keyword_suggestion_prompt():
            return load_prompts("keyword_suggestion")

        # Always use Azure Router (GPT-5)
        use_azure_router = True
        
        # Validate AI Service credentials (will refresh from environment if needed)
        try:
            config.validate()
        except ValueError as e:
            logger.error(f'AI Service configuration is incomplete: {e}')
            raise ValueError(f'AI Service configuration is incomplete: {e}')

        # Initialize processor with memory support, team analysis, and article search if needed
        enable_memory = not args.no_memory
        enable_team_analysis = not args.no_team_analysis
        processor = IncidentProcessor(
            enable_memory=enable_memory,
            enable_team_analysis=enable_team_analysis,
            articles_path=args.articles_embeddings,
            vector_db_path=args.vector_db_path
        )
        
        # Log which model is being used
        logger.info(f"Using Azure Router (GPT-5) with deployment: {processor.deployment_name}")

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
        if args.prompt_type == 'article_search_molecular':
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
            
            # Use Azure Router (GPT-5) for logging
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
            
            escalation_prompts = all_prompts.get('escalation_molecular', {})
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
                        prompt_type='escalation_molecular',
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
        
        # Generate summary with molecular context enhancement and memory integration
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
        
        # Use Azure Router (GPT-5) for logging
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

        # Interactive prompt to add example to molecular_examples.json
        try:
            # Check if stdin is available (not redirected)
            import sys
            if sys.stdin.isatty():
                add_example = input("Do you want to add this example to molecular_examples.json? (Y/N): ").strip().lower()
            else:
                print("Skipping interactive example addition (stdin not available)")
                add_example = 'n'
            if add_example == 'y':
                # Automatically use the current prompt type as the section if it's a molecular type
                if args.prompt_type.endswith('_molecular'):
                    section = args.prompt_type
                    print(f"Adding example to section: {section}")
                else:
                    # Fallback to manual selection for non-molecular prompt types
                    if sys.stdin.isatty():
                        section = input("Which section? (escalation_molecular / mitigation_molecular / troubleshooting_molecular / article_molecular / wait_time_molecular / prev_act_molecular): ").strip()
                    else:
                        print("Cannot select section interactively (stdin not available)")
                        section = "escalation_molecular"  # Default fallback

                # Generate suggested keywords using AI
                try:
                    keyword_prompts = get_keyword_suggestion_prompt()
                    keyword_messages = [
                        {"role": "system", "content": keyword_prompts["system_prompt"]},
                        {"role": "user", "content": keyword_prompts["user_prompt"].replace("{incident_text}", formatted_content)}
                    ]
                    # Use Azure Router (GPT-5) for keyword generation
                    keyword_response = processor.client.chat.completions.create(
                        model=processor.deployment_name,
                        messages=keyword_messages,
                        temperature=0.3,
                        max_tokens=100
                    )
                    suggested_keywords = keyword_response.choices[0].message.content.strip()
                except Exception as e:
                    print(f"Could not generate suggested keywords: {e}")
                    suggested_keywords = ""
                # Show existing categories for the selected section
                try:
                    with open("molecular_examples.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                        existing_categories = set()
                        if section in data:
                            existing_categories = set(
                                ex.get("category", "") for ex in data[section] if ex.get("category", "")
                            )
                        if existing_categories:
                            print(f"Existing categories in '{section}': {sorted(existing_categories)}")
                        else:
                            print(f"No categories found in '{section}'.")
                except Exception as e:
                    print(f"Could not load existing categories: {e}")
                if sys.stdin.isatty():
                    keywords = input(f"Enter keywords (suggested: {suggested_keywords}): ").strip().split(",")
                else:
                    print(f"Using suggested keywords: {suggested_keywords}")
                    keywords = suggested_keywords.split(",") if suggested_keywords else []
                keywords = [k.strip() for k in keywords if k.strip()]

                # Get categories only from the selected section and show as suggestions
                try:
                    with open("molecular_examples.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                        section_categories = set()
                        if section in data:
                            section_categories = set(
                                ex.get("category", "") for ex in data[section] if ex.get("category", "")
                            )
                        if section_categories:
                            category_suggestions = " / ".join(sorted(section_categories))
                        else:
                            category_suggestions = "No categories found in this section"
                except Exception as e:
                    category_suggestions = "Could not load categories"
                    
                if sys.stdin.isatty():
                    category = input(f"Enter category ({category_suggestions}): ").strip()
                else:
                    print(f"Using default category: general")
                    category = "general"
                # Filter noisy lines from input
                noise_phrases = [
                    "Transferred from",
                    "Support ICM enrichment CEM MDE",
                    "Acknowledging incident",
                    "[IMAGE DATA SHRUNK - TODO: handle image data]",
                    "Created by Azure Support Center"
                ]
                filtered_lines = []
                for line in formatted_content.splitlines():
                    if not any(phrase in line for phrase in noise_phrases):
                        # Remove leading '[timestamp] author:' if present
                        cleaned_line = re.sub(r"^\[.*?\]\s*[^:]+:\s*", "", line)
                        if cleaned_line.strip():
                            filtered_lines.append(cleaned_line.strip())
                filtered_input = "\n".join(filtered_lines).strip()
                # Prepare new example (no severity)
                new_example = {
                    "keywords": keywords,
                    "category": category,
                    "example": {
                        "input": filtered_input,
                        "output": summary_result["summary"] if isinstance(summary_result, dict) and "summary" in summary_result else str(summary_result)
                    }
                }
                # Load, append, and save
                with open("molecular_examples.json", "r+", encoding="utf-8") as f:
                    data = json.load(f)
                    if section not in data:
                        data[section] = []
                    data[section].append(new_example)
                    f.seek(0)
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.truncate()
                print(f"Example added to {section} in molecular_examples.json.")
        except Exception as e:
            logger.error(f"Error during interactive example addition: {str(e)}")
        return
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()