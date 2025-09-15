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
import openai
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
        def create(self, model, messages, temperature=0.7, max_tokens=2000):
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
from memory_manager import SummarizerMemoryManager
from article_searcher import ArticleSearcher

def run_gap_analysis_inline(incident_id: str, articles: List[Dict[str, Any]]):
    """Run interactive gap analysis after article search."""
    try:
        # Import the gap analysis functions
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from simple_gap_analysis import display_articles, get_article_content, create_gap_analysis_prompt
        
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
        
        # Get article content
        article_content = get_article_content(selected_article)
        
        # Create gap analysis prompt
        gap_analysis_prompt = create_gap_analysis_prompt(incident_data, article_content)
        
        print("\n" + "="*80)
        print("EXECUTING GAP ANALYSIS")
        print("="*80)
        print(f"Incident: {incident_id}")
        print(f"Article: {selected_article.get('title', 'Unknown')}")
        print("="*80)
        
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            azure_endpoint=config.azure_openai_endpoint
        )
        
        # Load prompts from prompts.json
        with open("prompts.json", "r") as f:
            all_prompts = json.load(f)
        
        gap_analysis_prompts = all_prompts.get("troubleshooting_gap_analysis", {})
        system_prompt = gap_analysis_prompts.get("system_prompt", "")
        user_prompt = gap_analysis_prompts.get("user_prompt", "")
        
        # Create the full prompt by combining user prompt with incident and article data
        # Create a more focused prompt that explicitly references the incident data
        focused_user_prompt = f"""
IMPORTANT: You must analyze the specific incident data provided below. Do NOT generate generic troubleshooting steps.

INCIDENT DATA TO ANALYZE:
{incident_data.get('summary', 'No summary available')}

TROUBLESHOOTING ARTICLE CONTENT:
{article_content}

Your task is to:
1. Analyze the SPECIFIC incident data above (Device Control policy issue on macOS)
2. Compare it against the troubleshooting article procedures
3. Identify gaps and create an execution plan

Focus ONLY on the Device Control policy issue described in the incident data. Do NOT analyze network connectivity or any other generic issues.
"""
        
        # Execute gap analysis with AI
        print("🤖 Executing gap analysis with AI...")
        response = client.chat.completions.create(
            model=config.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": focused_user_prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        # Handle different response structures
        try:
            gap_analysis_result = response.choices[0].message.content
        except (AttributeError, IndexError) as e:
            try:
                gap_analysis_result = response.choices[0].content
            except (AttributeError, IndexError):
                try:
                    gap_analysis_result = response.content
                except AttributeError:
                    gap_analysis_result = str(response)
                    print(f"⚠️ Warning: Could not extract content from response, using string representation: {e}")
        
        # Save the gap analysis result
        result_file = f"gap_analysis_{incident_id}_{selected_article.get('title', 'unknown').replace(' ', '_').replace('/', '_')}.txt"
        with open(result_file, 'w') as f:
            f.write(f"GAP ANALYSIS RESULTS\n")
            f.write(f"Incident: {incident_id}\n")
            f.write(f"Article: {selected_article.get('title', 'Unknown')}\n")
            f.write("="*80 + "\n\n")
            f.write(gap_analysis_result)
        
        print(f"\n✅ Gap analysis completed and saved to: {result_file}")
        
        # Display the results
        print("\n" + "="*80)
        print("GAP ANALYSIS RESULTS")
        print("="*80)
        print(gap_analysis_result)
        print("="*80)
        
    except Exception as e:
        logging.error(f"Error in interactive gap analysis: {e}")
        print(f"❌ Error during gap analysis: {e}")
        print("You can run gap analysis manually with: python3 simple_gap_analysis.py <incident_id>")

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
    def __init__(self, use_azure=False, use_zai=False, use_azure_5=False, enable_memory=True, 
                 articles_path=None, vector_db_path=None):
        # Set default behavior: if no specific model is specified, use Azure OpenAI 5
        if not use_azure and not use_zai and not use_azure_5:
            use_azure_5 = True
            
        self.use_azure = use_azure
        self.use_zai = use_zai
        self.use_azure_5 = use_azure_5
        self.molecular_engine = MolecularContextEngine()
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
        
        # Initialize article searcher if paths are provided
        self.article_searcher = None
        if articles_path or vector_db_path:
            try:
                self.article_searcher = ArticleSearcher(
                    articles_path=articles_path,
                    vector_db_path=vector_db_path,
                    use_azure=use_azure,
                    use_azure_5=use_azure_5,
                    use_zai=use_zai
                )
                logger.info("Article searcher initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize article searcher: {e}. Continuing without article search.")
                self.article_searcher = None
        
        if use_zai:
            if not all([config.zai_api_key, config.zai_base_url]):
                raise ValueError("ZAI configuration is incomplete. Please check your .env file.")
            
            self.client = ZaiClient(
                api_key=config.zai_api_key,
                base_url=config.zai_base_url
            )
            self.model_costs = {
                "input": config.zai_input_cost,
                "output": config.zai_output_cost
            }
            self.use_million_tokens = True  # ZAI costs are per 1M tokens
        elif use_azure_5:
            if not all([config.azure_openai_5_api_key, config.azure_openai_5_endpoint, 
                       config.azure_openai_5_api_version, config.azure_openai_5_deployment_name]):
                raise ValueError("Azure OpenAI 5 configuration is incomplete. Please check your .env file.")
            
            self.client = AzureOpenAI(
                api_key=config.azure_openai_5_api_key,
                api_version=config.azure_openai_5_api_version,
                azure_endpoint=config.azure_openai_5_endpoint
            )
            self.deployment_name = config.azure_openai_5_deployment_name
            # For Azure OpenAI 5, we'll use the same cost structure as OpenAI for now
            # You can add Azure-specific cost configs later if needed
            self.model_costs = {
                "input": config.openai_input_cost,
                "output": config.openai_output_cost
            }
            self.use_million_tokens = False  # OpenAI/Azure costs are per 1K tokens
        elif use_azure:
            if not all([config.azure_openai_api_key, config.azure_openai_endpoint, 
                       config.azure_openai_api_version, config.azure_openai_deployment_name]):
                raise ValueError("Azure OpenAI configuration is incomplete. Please check your .env file.")
            
            self.client = AzureOpenAI(
                api_key=config.azure_openai_api_key,
                api_version=config.azure_openai_api_version,
                azure_endpoint=config.azure_openai_endpoint
            )
            self.deployment_name = config.azure_openai_deployment_name
            # For Azure, we'll use the same cost structure as OpenAI for now
            # You can add Azure-specific cost configs later if needed
            self.model_costs = {
                "input": config.openai_input_cost,
                "output": config.openai_output_cost
            }
            self.use_million_tokens = False  # OpenAI/Azure costs are per 1K tokens
        else:
            if not config.openai_api_key:
                raise ValueError("OpenAI API key is not set. Please check your .env file.")
            
            self.client = openai.OpenAI(
                api_key=config.openai_api_key
            )
            self.model_costs = {
                "input": config.openai_input_cost,
                "output": config.openai_output_cost
            }
            self.use_million_tokens = False  # OpenAI costs are per 1K tokens
    
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
        """Generate summary using OpenAI or Azure OpenAI with molecular context enhancement and memory integration."""
        try:
            # Format the content
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
                    # Pass information about whether molecular context was used
                    molecular_context_used = molecular_examples_used > 0
                    enhanced_user_prompt = self.memory_manager.enhance_prompt_with_memory(
                        enhanced_user_prompt, incident_data, molecular_context_used
                    )
                    memory_enhanced = True
                    logger.info("Enhanced prompt with memory context")
                    print(f"🧠 Enhanced prompt with memory context from previous incidents")
                except Exception as e:
                    logger.warning(f"Failed to enhance prompt with memory: {e}")
            # Prepare messages - ensure content is a string
            messages = [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": f"{str(enhanced_user_prompt)}\n\nContent:\n{conversation}"}
            ]
            if debug_api:
                print("\n[DEBUG_API] LLM API request body:")
                # Determine the actual model that will be used
                if self.use_zai:
                    model_name = "z-ai"
                elif self.use_azure_5:
                    model_name = self.deployment_name
                elif self.use_azure:
                    model_name = self.deployment_name
                else:
                    model_name = config.openai_model_name
                print(json.dumps({
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2000
                }, indent=2, ensure_ascii=False))
            # Count input tokens
            input_text = f"{system_prompt}\n{enhanced_user_prompt}\n{conversation}"
            input_tokens = self.count_tokens(input_text)
            # Generate summary
            if self.use_zai:
                response = self.client.chat.completions.create(
                    model=config.zai_model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
            elif self.use_azure_5:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    max_completion_tokens=16384
                )
            elif self.use_azure:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
            else:
                response = self.client.chat.completions.create(
                    model=config.openai_model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
            # Get output tokens and calculate cost
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
            return result
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise
    
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
            # Format the incident data for search
            conversation = incident_data.get('conversation', [])
            summary = incident_data.get('summary', None)
            formatted_content = self.format_conversation_with_ai_summary(conversation, summary=summary)
            
            # Search for relevant articles
            logger.info("Searching for relevant articles...")
            search_results = self.article_searcher.search_articles(formatted_content, top_k=5, use_azure_openai=False)
            logger.info(f"Embedding search returned {len(search_results)} results")
            
            if not search_results:
                # Fallback to TF-IDF search
                logger.info("No results from embedding search, trying TF-IDF...")
                search_results = self.article_searcher.search_articles_tfidf(formatted_content, top_k=5)
                logger.info(f"TF-IDF search returned {len(search_results)} results")
            
            # Format search results for display
            formatted_results = self.article_searcher.format_search_results(
                search_results, 
                query=formatted_content, 
                include_explanations=True
            )
            
            # Create a simple analysis that just presents the articles
            if search_results:
                analysis_result = f"Top {len(search_results)} relevant articles found for this incident:\n\n{formatted_results}"
            else:
                analysis_result = "No relevant articles found for this incident."
            
            # Combine results
            result = {
                'search_results': search_results,
                'formatted_results': formatted_results,
                'analysis': analysis_result,
                'incident_data': formatted_content
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in article search processing: {e}")
            return {"error": str(e)}
    

    
    def process_multiple_incidents(self, combined_json_path, prompts, prompt_type, debug_api=False):
        """Process multiple incidents from a combined JSON file and generate unified summary."""
        try:
            logger.info(f"Processing multiple incidents from {combined_json_path}")
            
            # Load combined incident data
            with open(combined_json_path, 'r', encoding='utf-8') as f:
                combined_data = json.load(f)
            
            incidents = combined_data.get('incidents', [])
            total_incidents = combined_data.get('total_incidents', len(incidents))
            mode = combined_data.get('mode', 'standard')
            
            if not incidents:
                logger.error("No incidents found in combined data")
                return
            
            logger.info(f"Processing {len(incidents)} incidents for {mode} mode")
            
            # Handle troubleshooting plan mode
            if mode == "troubleshooting_plan":
                return self._process_troubleshooting_plan(combined_data, prompts, prompt_type, debug_api)
            
            # Combine all incident conversations and summaries
            all_conversations = []
            incident_numbers = []
            
            for incident in incidents:
                incident_number = incident.get('incident_number', 'unknown')
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
            
            # Determine model name for logging
            if self.use_zai:
                model_name = config.zai_model_name
            elif self.use_azure:
                model_name = self.deployment_name
            else:
                model_name = config.openai_model_name
            
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
            
            # Determine model name for logging
            if self.use_zai:
                model_name = config.zai_model_name
            elif self.use_azure:
                model_name = self.deployment_name
            else:
                model_name = config.openai_model_name
            
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

            # Print the summary to the console
            if ai_summary and 'summary' in ai_summary:
                print("\nAI Generated Summary:")
                print("="*80)
                print(ai_summary['summary'])
                print("="*80)
                
                # Print molecular context info if available
                if 'molecular_context' in ai_summary:
                    print(f"\nMolecular Context: {ai_summary['molecular_context']['examples_used']} examples used for {ai_summary['molecular_context']['prompt_type']}")
                
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
        
        # Load and apply guidelines from AGENTS.md
        try:
            with open('AGENTS.md', 'r', encoding='utf-8') as f:
                agents_content = f.read()
            
            # Extract writing guidelines from AGENTS.md
            writing_guidelines = []
            
            # Extract guidelines from the Writing Guidelines section
            if "## Writing Guidelines" in agents_content:
                guidelines_section = agents_content.split("## Writing Guidelines")[1].split("## ")[0]
                
                # Extract guidelines from each subsection
                for line in guidelines_section.split('\n'):
                    line = line.strip()
                    if line.startswith('- ') and not line.startswith('- Use abbreviated names'):
                        # Skip the first guideline about abbreviations as it's already handled
                        writing_guidelines.append(line[2:])  # Remove the "- " prefix
            
            # Apply writing guidelines to system prompt
            if writing_guidelines:
                guidelines_text = "\n\nCRITICAL WRITING GUIDELINES - MUST FOLLOW:\n" + "\n".join([f"- {guideline}" for guideline in writing_guidelines])
                prompts["system_prompt"] = guidelines_text + "\n\n" + str(prompts["system_prompt"])
                logger.info(f"Applied {len(writing_guidelines)} writing guidelines from AGENTS.md to '{prompt_type}'")
        
        except Exception as e:
            logger.warning(f"Could not load guidelines from AGENTS.md: {e}")
        
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
    parser.add_argument('--azure', action='store_true', help='Use Azure OpenAI (GPT-4) instead of default Azure OpenAI 5')
    parser.add_argument('--openai', action='store_true', help='Use OpenAI API instead of Azure OpenAI (requires OPENAI_API_KEY)')
    parser.add_argument('--zai', action='store_true', help='Use ZAI API (requires ZAI_API_KEY and ZAI_BASE_URL)')
    parser.add_argument('--azure-5', action='store_true', help='Use Azure OpenAI 5 (GPT-5) - this is now the default')
    parser.add_argument('--prompt-type', default='default', help='Type of prompt to use (default, technical, executive, escalation, escalation_molecular, mitigation_molecular, troubleshooting_molecular, article_search_molecular, etc.)')
    parser.add_argument('--debug', '-d', action='store_true', help='Print the body of the API request sent to the LLM for debugging.')
    parser.add_argument('--multi-incident', action='store_true', help='Process multiple incidents from a combined JSON file')
    parser.add_argument('--no-memory', action='store_true', help='Disable memory integration for this processing session')
    parser.add_argument('--articles-path', help='Path to directory containing troubleshooting articles (for article search mode)')
    parser.add_argument('--vector-db-path', help='Path to vector database file (for article search mode)')
    args = parser.parse_args()

    try:
        # Load prompts
        prompts = load_prompts(args.prompt_type)
        
        # Helper function to get keyword suggestion prompt with guidelines applied
        def get_keyword_suggestion_prompt():
            return load_prompts("keyword_suggestion")

        # Determine which LLM to use
        use_openai = args.openai
        use_zai = args.zai
        use_azure_5 = args.azure_5
        use_azure = args.azure  # This now means GPT-4
        
        if use_zai:
            # Check ZAI credentials
            if not all([config.zai_api_key, config.zai_base_url]):
                logger.error('ZAI API key or base URL is not set. Please check your .env file.')
                raise ValueError('ZAI API key or base URL is not set. Please check your .env file.')
        elif use_openai:
            # Only require OpenAI API key if --openai is set
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                logger.error('OpenAI API key is not set. Please check your .env file or environment variables.')
                raise ValueError('OpenAI API key is not set. Please check your .env file or environment variables.')
        elif use_azure:
            # Check Azure OpenAI (GPT-4) credentials
            if not all([config.azure_openai_api_key, config.azure_openai_endpoint, 
                       config.azure_openai_api_version, config.azure_openai_deployment_name]):
                logger.error('Azure OpenAI (GPT-4) configuration is incomplete. Please check your .env file.')
                raise ValueError('Azure OpenAI (GPT-4) configuration is incomplete. Please check your .env file.')
        else:
            # Azure OpenAI 5 is default - check credentials
            if not all([config.azure_openai_5_api_key, config.azure_openai_5_endpoint, 
                       config.azure_openai_5_api_version, config.azure_openai_5_deployment_name]):
                logger.error('Azure OpenAI 5 configuration is incomplete. Please check your .env file.')
                raise ValueError('Azure OpenAI 5 configuration is incomplete. Please check your .env file.')

        # Initialize processor with memory support and article search if needed
        enable_memory = not args.no_memory
        processor = IncidentProcessor(
            use_azure=use_azure, 
            use_zai=use_zai, 
            use_azure_5=use_azure_5, 
            enable_memory=enable_memory,
            articles_path=args.articles_path,
            vector_db_path=args.vector_db_path
        )
        
        # Log which model is being used
        if processor.use_azure_5:
            logger.info(f"Using Azure OpenAI 5 (GPT-5) with deployment: {processor.deployment_name}")
        elif processor.use_azure:
            logger.info(f"Using Azure OpenAI (GPT-4) with deployment: {processor.deployment_name}")
        elif processor.use_zai:
            logger.info(f"Using ZAI with model: {config.zai_model_name}")
        else:
            logger.info(f"Using OpenAI with model: {config.openai_model_name}")

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
            if not processor.article_searcher:
                logger.error("Article search mode requires --articles-path or --vector-db-path")
                print("Error: Article search mode requires --articles-path or --vector-db-path")
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
            
            # Determine model name for logging
            if processor.use_zai:
                model_name = config.zai_model_name
            elif processor.use_azure_5:
                model_name = processor.deployment_name
            elif processor.use_azure:
                model_name = processor.deployment_name
            else:
                model_name = config.openai_model_name
            
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
            
            # Print search results
            print("\n" + "="*80)
            print("ARTICLE SEARCH RESULTS")
            print("="*80)
            print(search_result['formatted_results'])
            print("="*80)
            
            # Only offer gap analysis for article_search_molecular prompt type
            if args.prompt_type == 'article_search_molecular':
                print("\n" + "="*80)
                print("GAP ANALYSIS OPTION")
                print("="*80)
                print("Would you like to perform gap analysis on one of these articles?")
                print("This will compare your incident against the troubleshooting procedures")
                print("and identify missing steps that need to be executed.")
                print("="*80)
                
                try:
                    response = input("\nProceed with gap analysis? (y/n): ").strip().lower()
                    if response in ['y', 'yes']:
                        # Run gap analysis inline
                        run_gap_analysis_inline(incident_number, search_result['search_results'])
                except (KeyboardInterrupt, EOFError):
                    print("\nSkipping gap analysis.")
                except Exception as e:
                    print(f"\nError during gap analysis: {e}")
                    print("You can run gap analysis manually with: python3 simple_gap_analysis.py <incident_id>")
            
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
        
        # Determine model name for logging
        if processor.use_zai:
            model_name = config.zai_model_name
        elif processor.use_azure_5:
            model_name = processor.deployment_name
        elif processor.use_azure:
            model_name = processor.deployment_name
        else:
            model_name = config.openai_model_name
            
        processor.save_to_json(
            conversation,
            incident_number,
            ai_summary=summary_result,
            prompt_type=args.prompt_type,
            operation_time=operation_time,
            model_name=model_name
        )
        
        # Store memory about this incident if memory is enabled
        if processor.memory_manager:
            try:
                processor.store_incident_memory(incident_number, data, summary_result)
                logger.info(f"Stored memory for incident {incident_number}")
            except Exception as e:
                logger.error(f"Failed to store memory for incident {incident_number}: {e}")
        
        logger.info(f"Completed processing {args.input_file}")

        # Interactive prompt to add example to molecular_examples.json
        try:
            add_example = input("Do you want to add this example to molecular_examples.json? (Y/N): ").strip().lower()
            if add_example == 'y':
                # Automatically use the current prompt type as the section if it's a molecular type
                if args.prompt_type.endswith('_molecular'):
                    section = args.prompt_type
                    print(f"Adding example to section: {section}")
                else:
                    # Fallback to manual selection for non-molecular prompt types
                    section = input("Which section? (escalation_molecular / mitigation_molecular / troubleshooting_molecular / article_molecular / wait_time_molecular / prev_act_molecular): ").strip()

                # Generate suggested keywords using AI
                try:
                    keyword_prompts = get_keyword_suggestion_prompt()
                    keyword_messages = [
                        {"role": "system", "content": keyword_prompts["system_prompt"]},
                        {"role": "user", "content": keyword_prompts["user_prompt"].replace("{incident_text}", formatted_content)}
                    ]
                    if processor.use_zai:
                        keyword_response = processor.client.chat.completions.create(
                            model=config.zai_model_name,
                            messages=keyword_messages,
                            temperature=0.3,
                            max_tokens=100
                        )
                    elif processor.use_azure_5:
                        keyword_response = processor.client.chat.completions.create(
                            model=processor.deployment_name,
                            messages=keyword_messages,
                            max_completion_tokens=16384
                        )
                    elif processor.use_azure:
                        keyword_response = processor.client.chat.completions.create(
                            model=processor.deployment_name,
                            messages=keyword_messages,
                            temperature=0.3,
                            max_tokens=100
                        )
                    else:
                        keyword_response = processor.client.chat.completions.create(
                            model=config.openai_model_name,
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
                keywords = input(f"Enter keywords (suggested: {suggested_keywords}): ").strip().split(",")
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
                    
                category = input(f"Enter category ({category_suggestions}): ").strip()
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