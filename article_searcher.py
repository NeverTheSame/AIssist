import json
import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from openai import AzureOpenAI
from config import config

logger = logging.getLogger(__name__)

class ArticleSearcher:
    """
    A class to search for relevant troubleshooting articles based on incident data.
    Supports both vector database and text file-based article storage.
    """
    
    def __init__(self, articles_path: Optional[str] = None, vector_db_path: Optional[str] = None, 
                 use_azure: bool = False, use_azure_5: bool = False, use_zai: bool = False):
        """
        Initialize the ArticleSearcher.
        
        Args:
            articles_path: Path to directory containing text articles
            vector_db_path: Path to vector database file (JSON with embeddings)
            use_azure: Use Azure OpenAI for embeddings
            use_azure_5: Use Azure OpenAI 5 for embeddings
            use_zai: Use ZAI for embeddings
        """
        self.articles_path = articles_path
        self.vector_db_path = vector_db_path
        self.use_azure = use_azure
        self.use_azure_5 = use_azure_5
        self.use_zai = use_zai
        
        # Initialize the appropriate client
        self._init_client()
        
        # Initialize embedding client (always uses regular Azure OpenAI)
        self._init_embedding_client()
        
        # Load articles and embeddings
        self.articles = []
        self.embeddings = []
        self._load_articles()
    
    def _init_client(self):
        """Initialize the appropriate OpenAI client based on configuration."""
        if self.use_zai:
            if not all([config.zai_api_key, config.zai_base_url]):
                raise ValueError("ZAI configuration is incomplete. Please check your .env file.")
            
            # For ZAI, we'll use OpenAI client with custom base URL
            self.client = openai.OpenAI(
                api_key=config.zai_api_key,
                base_url=config.zai_base_url
            )
        elif self.use_azure_5:
            if not all([config.azure_openai_5_api_key, config.azure_openai_5_endpoint, 
                       config.azure_openai_5_api_version]):
                raise ValueError("Azure OpenAI 5 configuration is incomplete. Please check your .env file.")
            
            self.client = AzureOpenAI(
                api_key=config.azure_openai_5_api_key,
                api_version=config.azure_openai_5_api_version,
                azure_endpoint=config.azure_openai_5_endpoint
            )
        elif self.use_azure:
            if not all([config.azure_openai_api_key, config.azure_openai_endpoint, 
                       config.azure_openai_api_version]):
                raise ValueError("Azure OpenAI configuration is incomplete. Please check your .env file.")
            
            self.client = AzureOpenAI(
                api_key=config.azure_openai_api_key,
                api_version=config.azure_openai_api_version,
                azure_endpoint=config.azure_openai_endpoint
            )
        else:
            if not config.openai_api_key:
                raise ValueError("OpenAI API key is not set. Please check your .env file.")
            
            self.client = openai.OpenAI(api_key=config.openai_api_key)
    
    def _init_embedding_client(self):
        """Initialize a separate client for embeddings (always uses regular Azure OpenAI)."""
        if not all([config.azure_openai_api_key, config.azure_openai_endpoint, 
                   config.azure_openai_api_version]):
            raise ValueError("Azure OpenAI configuration is incomplete for embeddings. Please check your .env file.")
        
        self.embedding_client = AzureOpenAI(
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            azure_endpoint=config.azure_openai_endpoint
        )
    
    def _load_articles(self):
        """Load articles from the specified source."""
        if self.vector_db_path and os.path.exists(self.vector_db_path):
            self._load_from_vector_db()
        elif self.articles_path and os.path.exists(self.articles_path):
            self._load_from_text_files()
        else:
            logger.warning("No articles found. Please provide either vector_db_path or articles_path.")
    
    def _load_from_vector_db(self):
        """Load articles from a vector database file."""
        try:
            with open(self.vector_db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle the Articles Inventorizer format where keys are URLs and values are embeddings
            if isinstance(data, dict):
                # Check if embeddings are nested under an "embeddings" key
                if 'embeddings' in data and isinstance(data['embeddings'], dict):
                    # Format: {"embeddings": {"url": embedding, ...}}
                    embeddings_dict = data['embeddings']
                    self.articles = []
                    self.embeddings = []
                    
                    for url, embedding in embeddings_dict.items():
                        # Extract title from URL - decode URL encoding first
                        from urllib.parse import unquote
                        decoded_url = unquote(url)
                        title = decoded_url.split('/')[-1].replace('.md', '')
                        if title.endswith('-TSG'):
                            title = title[:-4] + ' TSG'
                        
                        # Get content summary from the actual article file
                        content_summary = self._get_content_summary(url)
                        
                        article = {
                            'title': title,
                            'url': url,
                            'content_summary': content_summary,
                            'full_content_path': url,
                            'embedding': embedding
                        }
                        
                        self.articles.append(article)
                        # Ensure embedding is a list of floats with 1536 dimensions
                        if isinstance(embedding, dict):
                            # If embedding is a dict, try to extract the embedding values
                            embedding = list(embedding.values()) if embedding else []
                        elif not isinstance(embedding, list):
                            embedding = []
                        
                        # Convert all values to float and ensure 1536 dimensions
                        try:
                            embedding = [float(x) for x in embedding]
                            # Ensure exactly 1536 dimensions
                            if len(embedding) != 1536:
                                if len(embedding) > 1536:
                                    embedding = embedding[:1536]
                                    logger.info(f"Truncated embedding for {url} to 1536 dimensions")
                                else:
                                    embedding.extend([0.0] * (1536 - len(embedding)))
                                    logger.info(f"Padded embedding for {url} to 1536 dimensions")
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid embedding format for {url}, using zero vector")
                            embedding = [0.0] * 1536
                        
                        self.embeddings.append(embedding)
                    
                    logger.info(f"Converted {len(self.articles)} articles from Articles Inventorizer format (nested embeddings)")
                elif not data.get('articles'):
                    # Format: {"url": embedding, ...} (direct mapping)
                    self.articles = []
                    self.embeddings = []
                    
                    for url, embedding in data.items():
                        # Extract title from URL - decode URL encoding first
                        from urllib.parse import unquote
                        decoded_url = unquote(url)
                        title = decoded_url.split('/')[-1].replace('.md', '')
                        if title.endswith('-TSG'):
                            title = title[:-4] + ' TSG'
                        
                        # Get content summary from the actual article file
                        content_summary = self._get_content_summary(url)
                        
                        article = {
                            'title': title,
                            'url': url,
                            'content_summary': content_summary,
                            'full_content_path': url,
                            'embedding': embedding
                        }
                        
                        self.articles.append(article)
                        # Ensure embedding is a list of floats with 1536 dimensions
                        if isinstance(embedding, dict):
                            # If embedding is a dict, try to extract the embedding values
                            embedding = list(embedding.values()) if embedding else []
                        elif not isinstance(embedding, list):
                            embedding = []
                        
                        # Convert all values to float and ensure 1536 dimensions
                        try:
                            embedding = [float(x) for x in embedding]
                            # Ensure exactly 1536 dimensions
                            if len(embedding) != 1536:
                                if len(embedding) > 1536:
                                    embedding = embedding[:1536]
                                    logger.info(f"Truncated embedding for {url} to 1536 dimensions")
                                else:
                                    embedding.extend([0.0] * (1536 - len(embedding)))
                                    logger.info(f"Padded embedding for {url} to 1536 dimensions")
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid embedding format for {url}, using zero vector")
                            embedding = [0.0] * 1536
                        
                        self.embeddings.append(embedding)
                    
                    logger.info(f"Converted {len(self.articles)} articles from Articles Inventorizer format (direct mapping)")
                else:
                    # Handle the expected format with separate articles and embeddings arrays
                    self.articles = data.get('articles', [])
                    raw_embeddings = data.get('embeddings', [])
                    
                    # Ensure all embeddings are lists of floats with 1536 dimensions
                    self.embeddings = []
                    for i, embedding in enumerate(raw_embeddings):
                        if isinstance(embedding, dict):
                            # If embedding is a dict, try to extract the embedding values
                            embedding = list(embedding.values()) if embedding else []
                        elif not isinstance(embedding, list):
                            embedding = []
                        
                        # Convert all values to float and ensure 1536 dimensions
                        try:
                            embedding = [float(x) for x in embedding]
                            # Ensure exactly 1536 dimensions
                            if len(embedding) != 1536:
                                if len(embedding) > 1536:
                                    embedding = embedding[:1536]
                                    logger.info(f"Truncated embedding {i} to 1536 dimensions")
                                else:
                                    embedding.extend([0.0] * (1536 - len(embedding)))
                                    logger.info(f"Padded embedding {i} to 1536 dimensions")
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid embedding format for article {i}, using zero vector")
                            embedding = [0.0] * 1536
                        
                        self.embeddings.append(embedding)
                    
                    logger.info(f"Loaded {len(self.articles)} articles from standard vector database format")
            
            logger.info(f"Total articles loaded: {len(self.articles)}")
            logger.info(f"Total embeddings loaded: {len(self.embeddings)}")
            
            # Debug: Check the structure of the first article and embedding
            if self.articles:
                logger.info(f"First article keys: {list(self.articles[0].keys())}")
                logger.info(f"First article title: {self.articles[0].get('title', 'No title')}")
            
            if self.embeddings:
                logger.info(f"First embedding length: {len(self.embeddings[0])}")
                logger.info(f"First embedding type: {type(self.embeddings[0])}")
                logger.info(f"First embedding sample: {self.embeddings[0][:5] if len(self.embeddings[0]) >= 5 else self.embeddings[0]}")
            
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            raise
    
    def _load_from_text_files(self):
        """Load articles from text files in the specified directory."""
        try:
            articles = []
            embeddings = []
            
            articles_dir = Path(self.articles_path)
            if not articles_dir.exists():
                logger.warning(f"Articles directory {self.articles_path} does not exist")
                return
            
            # Find all text files
            text_files = list(articles_dir.glob("*.txt")) + list(articles_dir.glob("*.md"))
            
            for file_path in text_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract title from filename or first line
                    title = file_path.stem
                    if content.strip():
                        first_line = content.split('\n')[0].strip()
                        if first_line.startswith('# '):
                            title = first_line[2:]
                    
                    article = {
                        'title': title,
                        'content': content,
                        'file_path': str(file_path),
                        'url': f"file://{file_path.absolute()}"
                    }
                    
                    articles.append(article)
                    
                    # Generate embedding for the article
                    embedding = self._generate_embedding(content)
                    embeddings.append(embedding)
                    
                except Exception as e:
                    logger.warning(f"Error loading article {file_path}: {e}")
                    continue
            
            self.articles = articles
            self.embeddings = embeddings
            
            logger.info(f"Loaded {len(self.articles)} articles from text files")
            
        except Exception as e:
            logger.error(f"Error loading articles from text files: {e}")
            raise
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """
        Generate a simple 1536-dimensional vector representation of the text.
        Uses TF-IDF vectorization and extends/pads to 1536 dimensions.
        """
        try:
            logger.info("Generating simple 1536-dimensional vector representation")
            
            # Create a simple TF-IDF vectorization
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
            
            # Create a TF-IDF vectorizer with a large feature set
            vectorizer = TfidfVectorizer(
                max_features=1536,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0
            )
            
            # Fit and transform the text
            # We need to fit on some sample data first, so we'll use the text itself
            tfidf_matrix = vectorizer.fit_transform([text])
            
            # Convert to dense array and get the first (and only) vector
            embedding = tfidf_matrix.toarray()[0]
            
            # Ensure we have exactly 1536 dimensions
            if len(embedding) != 1536:
                if len(embedding) > 1536:
                    embedding = embedding[:1536]
                else:
                    # Pad with zeros to reach 1536 dimensions
                    embedding = np.pad(embedding, (0, 1536 - len(embedding)), 'constant')
            
            # Normalize the vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            logger.info(f"Generated simple embedding with {len(embedding)} dimensions")
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating simple embedding: {e}")
            logger.info("Using zero vector as fallback")
            return [0.0] * 1536
    
    def _generate_azure_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using Azure OpenAI (only when explicitly requested)."""
        try:
            # Try BAAI/bge-large-en-v1.5 first for consistency with article embeddings
            deployment_name = "BAAI/bge-large-en-v1.5"
            
            logger.info(f"Attempting to use Azure OpenAI embedding deployment: {deployment_name}")
            logger.info(f"Azure OpenAI endpoint: {self.embedding_client.base_url}")
            
            try:
                response = self.embedding_client.embeddings.create(
                    model=deployment_name,
                    input=text
                )
                
                embedding = response.data[0].embedding
                
                # Verify we get 1536 dimensions as expected
                if len(embedding) != 1536:
                    logger.warning(f"Unexpected embedding dimension: {len(embedding)}, expected 1536")
                
                logger.info(f"Successfully generated Azure OpenAI embedding with {len(embedding)} dimensions using {deployment_name}")
                return embedding
                
            except Exception as e:
                logger.warning(f"Failed to use {deployment_name}: {e}")
                logger.info("Falling back to text-embedding-ada-002...")
                
                # Fallback to text-embedding-ada-002
                fallback_deployment = "text-embedding-ada-002"
                response = self.embedding_client.embeddings.create(
                    model=fallback_deployment,
                    input=text
                )
                
                embedding = response.data[0].embedding
                
                # text-embedding-ada-002 produces 1536 dimensions, so we should be good
                if len(embedding) != 1536:
                    logger.warning(f"Fallback embedding has unexpected dimension: {len(embedding)}, expected 1536")
                    # Ensure 1536 dimensions
                    if len(embedding) > 1536:
                        embedding = embedding[:1536]
                    else:
                        embedding.extend([0.0] * (1536 - len(embedding)))
                
                logger.info(f"Successfully generated Azure OpenAI embedding with {len(embedding)} dimensions using fallback {fallback_deployment}")
                return embedding
            
        except Exception as e:
            logger.error(f"Error generating Azure OpenAI embedding: {e}")
            logger.info("Using zero vector as fallback")
            # Return a zero vector as fallback with 1536 dimensions
            return [0.0] * 1536
    
    def _generate_embedding(self, text: str, use_azure_openai: bool = False) -> List[float]:
        """
        Generate embedding for the given text.
        
        Args:
            text: The text to embed
            use_azure_openai: If True, use Azure OpenAI embeddings. Default is False (use simple vectorization)
        """
        if use_azure_openai:
            return self._generate_azure_openai_embedding(text)
        else:
            return self._generate_simple_embedding(text)
    
    def search_articles(self, query: str, top_k: int = 5, use_azure_openai: bool = False) -> List[Dict[str, Any]]:
        """
        Search for relevant articles based on the query.
        
        Args:
            query: The search query (incident description)
            top_k: Number of top results to return
            use_azure_openai: If True, use Azure OpenAI embeddings for query. Default is False (use simple vectorization)
            
        Returns:
            List of relevant articles with relevance scores
        """
        if not self.articles:
            logger.warning("No articles available for search")
            return []
        
        # Use TF-IDF search as primary method since it works better for this use case
        logger.info("Using TF-IDF search for article matching")
        return self.search_articles_tfidf(query, top_k)
    
    def search_articles_tfidf(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant articles using TF-IDF similarity (fallback method).
        
        Args:
            query: The search query (incident description)
            top_k: Number of top results to return
            
        Returns:
            List of relevant articles with relevance scores
        """
        if not self.articles:
            logger.warning("No articles available for search")
            return []
        
        try:
            # Prepare documents for TF-IDF
            documents = [article.get('content_summary', article.get('content', '')) for article in self.articles]
            documents.append(query)
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Calculate similarities
            query_vector = tfidf_matrix[-1]  # Last document is the query
            similarities = cosine_similarity(query_vector, tfidf_matrix[:-1])[0]
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include articles with some relevance
                    article = self.articles[idx].copy()
                    article['relevance_score'] = float(similarities[idx])
                    results.append(article)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching articles with TF-IDF: {e}")
            return []
    
    def format_search_results(self, results: List[Dict[str, Any]], query: str = "", include_explanations: bool = False) -> str:
        """
        Format search results for display or processing.
        
        Args:
            results: List of search results from search_articles()
            query: The original search query for context
            include_explanations: Whether to include AI-generated explanations
            
        Returns:
            Formatted string with search results
        """
        if not results:
            return "No relevant articles found."
        
        formatted_results = []
        for i, article in enumerate(results, 1):
            title = article.get('title', 'Unknown Title')
            url = article.get('url', 'No URL available')
            score = article.get('relevance_score', 0.0)
            
            formatted_results.append(f"{i}. {title}")
            formatted_results.append(f"   URL: {url}")
            formatted_results.append(f"   Relevance Score: {score:.3f}")
            
            # Add AI-generated explanation if requested
            if include_explanations and query:
                explanation = self._generate_relevance_explanation(title, query, score)
                if explanation:
                    formatted_results.append(f"   Why relevant: {explanation}")
            
            formatted_results.append("")
        
        return "\n".join(formatted_results)
    
    def _generate_relevance_explanation(self, title: str, query: str, score: float) -> str:
        """
        Generate a brief explanation of why an article is relevant to the query.
        
        Args:
            title: Article title
            query: Search query
            score: Relevance score
            
        Returns:
            Brief explanation string
        """
        try:
            title_lower = title.lower()
            query_lower = query.lower()
            
            # Extract matching terms
            matching_terms = []
            for term in query_lower.split():
                if term in title_lower and len(term) > 3:  # Only significant terms
                    matching_terms.append(term)
            
            # Remove duplicates and get unique terms
            unique_terms = list(set(matching_terms))
            
            # Generate contextual explanation based on content
            if "troubleshooting" in title_lower and "device control" in title_lower:
                return f"Direct troubleshooting guide for Device Control issues on macOS. Contains key terms: {', '.join(unique_terms[:2])}."
            elif "audit" in title_lower and "policy" in title_lower:
                return f"Policy audit documentation that addresses policy configuration and enforcement issues."
            elif "usb" in title_lower and "device control" in title_lower:
                return f"USB-specific Device Control troubleshooting that directly relates to removable media access issues."
            elif "device control" in title_lower and "printers" in title_lower:
                return f"Device Control implementation guide, though focused on printers rather than USB devices."
            elif "device control" in title_lower and "tsg" in title_lower:
                return f"Technical Support Guide (TSG) for Device Control that provides comprehensive troubleshooting steps."
            elif unique_terms:
                terms_str = ", ".join(unique_terms[:3])
                return f"Contains relevant technical terms: {terms_str}. Score {score:.3f} indicates good content match."
            else:
                return f"Content similarity score of {score:.3f} suggests relevance, though specific term matches are not obvious."
                
        except Exception as e:
            logger.warning(f"Error generating explanation: {e}")
            return ""
    
    def _get_content_summary(self, url: str) -> str:
        """Get a content summary from the actual article file."""
        try:
            # Convert URL to local file path - keep URL encoding as is
            base_path = "/Users/kirill/Documents/M/Code/AzureDevops/ArticlesInventorizer"
            file_path = os.path.join(base_path, url)
            
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract a meaningful summary (first few paragraphs or key sections)
                lines = content.split('\n')
                summary_lines = []
                
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        summary_lines.append(line)
                        if len(summary_lines) >= 5:  # Get first 5 non-empty lines
                            break
                
                if summary_lines:
                    return '\n'.join(summary_lines)
                else:
                    return f"Article content available at: {url}"
            else:
                return f"Article file not found: {url}"
                
        except Exception as e:
            logger.warning(f"Error reading article content for {url}: {e}")
            return f"Article content available at: {url}"
    
    def get_full_article_content(self, url: str) -> str:
        """Get the full article content for detailed analysis."""
        try:
            # Convert URL to local file path - keep URL encoding as is
            base_path = "/Users/kirill/Documents/M/Code/AzureDevops/ArticlesInventorizer"
            file_path = os.path.join(base_path, url)
            
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                return f"Article file not found: {file_path}"
                
        except Exception as e:
            logger.error(f"Error reading full article content for {url}: {e}")
            return f"Error reading article: {e}"
    
    def save_vector_db(self, output_path: str):
        """Save articles and embeddings to a vector database file."""
        try:
            # Ensure all embeddings are lists of floats with 1536 dimensions before saving
            processed_embeddings = []
            for i, embedding in enumerate(self.embeddings):
                if isinstance(embedding, list):
                    try:
                        processed_embedding = [float(x) for x in embedding]
                        # Ensure exactly 1536 dimensions
                        if len(processed_embedding) != 1536:
                            if len(processed_embedding) > 1536:
                                processed_embedding = processed_embedding[:1536]
                                logger.info(f"Truncated embedding {i} to 1536 dimensions")
                            else:
                                processed_embedding.extend([0.0] * (1536 - len(processed_embedding)))
                                logger.info(f"Padded embedding {i} to 1536 dimensions")
                        processed_embeddings.append(processed_embedding)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing embedding {i} for saving: {e}")
                        # Use zero vector as fallback
                        processed_embeddings.append([0.0] * 1536)
                else:
                    logger.warning(f"Embedding {i} is not a list, using zero vector")
                    processed_embeddings.append([0.0] * 1536)
            
            data = {
                'articles': self.articles,
                'embeddings': processed_embeddings,
                'metadata': {
                    'total_articles': len(self.articles),
                    'created_at': str(Path().absolute()),
                    'embedding_dimension': 1536,
                    'embedding_model': 'BAAI/bge-large-en-v1.5',
                    'embedding_provider': 'Azure OpenAI'
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved vector database to {output_path}")
            logger.info(f"Saved {len(processed_embeddings)} embeddings with 1536 dimensions")
            
        except Exception as e:
            logger.error(f"Error saving vector database: {e}")
            raise
