import json
import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
import openai
from openai import AzureOpenAI
from config import config
from functools import cmp_to_key
import re

logger = logging.getLogger(__name__)

class ArticleSearcher:
    """
    Unified article searcher supporting both basic and advanced search methods.
    
    Features:
    - Multi-stage semantic search pipeline with AI
    - LLM-based relevance scoring and sorting
    - Intelligent explanation generation
    - Hybrid search combining semantic and keyword matching
    - TF-IDF fallback for simple searches
    - Backward compatible with legacy code
    
    Use search_articles() for simple TF-IDF search (fast, backward compatible)
    Use search_articles_advanced() for AI-powered multi-stage search (slower, more accurate)
    """
    
    def __init__(self, articles_path: Optional[str] = None, vector_db_path: Optional[str] = None, 
                 use_azure_router: bool = True):
        """
        Initialize the AdvancedArticleSearcher.
        
        Args:
            articles_path: Path to directory containing text articles
            vector_db_path: Path to vector database file (JSON with embeddings)
            use_azure_router: Use Azure Router for embeddings and LLM calls (always True)
        """
        self.articles_path = articles_path
        self.vector_db_path = vector_db_path
        self.use_azure_router = use_azure_router
        
        # Initialize the appropriate client
        self._init_client()
        
        # Initialize embedding client (always uses regular Azure OpenAI)
        self._init_embedding_client()
        
        # Load articles and embeddings
        self.articles = []
        self.embeddings = []
        self._load_articles()
        
        # Relevance scoring threshold
        self.relevance_threshold = 5.0
        
        # Maximum candidates for LLM processing
        self.max_candidates = 20
        
        # Final results count
        self.final_results_count = 5
        
        # Local embedder (initialized lazily)
        self.local_embedder = None
        
        # Always use all-MiniLM-L6-v2 for consistent embeddings (384 dimensions)
        try:
            self._ensure_local_embedder()
            logger.info("Using all-MiniLM-L6-v2 (384 dimensions) for consistent semantic search across all articles")
        except Exception as e:
            logger.warning(f"Failed to initialize all-MiniLM-L6-v2: {e}")
    
    def _ensure_local_embedder(self):
        """Initialize a local embedding model if available and not yet created."""
        if self.local_embedder is None and SentenceTransformer is not None:
            try:
                # Small, fast local model (384 dimensions) - consistent with article embeddings
                self.local_embedder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Initialized local SentenceTransformer: all-MiniLM-L6-v2 (384 dimensions)")
            except Exception as e:
                logger.warning(f"Failed to initialize local SentenceTransformer: {e}")
                self.local_embedder = None
    
    def _read_article_full_text(self, article: Dict[str, Any]) -> str:
        """Read full text for an article from available sources."""
        if 'content' in article and article['content']:
            return article['content']
        # Try to read from local file based on url path
        try:
            base_path = "/Users/kirill/Documents/M/Code/AzureDevops/ArticlesInventorizer"
            url = article.get('url') or article.get('full_content_path')
            if url:
                file_path = os.path.join(base_path, url)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
        except Exception:
            pass
        # Fallback to summary
        return article.get('content_summary', '') or ''

    def _recompute_embeddings_locally(self):
        """Recompute all article embeddings using local SentenceTransformer."""
        if self.local_embedder is None:
            return
        texts: List[str] = []
        for art in self.articles:
            texts.append(self._read_article_full_text(art))
        try:
            vectors = self.local_embedder.encode(texts, normalize_embeddings=False)
            new_embeddings: List[List[float]] = []
            for vec in vectors:
                if isinstance(vec, np.ndarray):
                    emb = vec.tolist()
                else:
                    emb = list(vec)
                new_embeddings.append(emb)
            self.embeddings = new_embeddings
            logger.info(f"Recomputed {len(new_embeddings)} embeddings locally")
        except Exception as e:
            logger.warning(f"Failed to recompute embeddings locally: {e}")

    def _init_client(self):
        """Initialize Azure Router client (GPT-5)."""
        try:
            self.client = AzureOpenAI(
                api_key=config.ai_service_api_key,
                azure_endpoint=config.ai_service_endpoint,
                api_version=config.ai_service_api_version
            )
            self.deployment_name = config.ai_service_deployment_name
            logger.info("Initialized Azure Router client (GPT-5)")
        except Exception as e:
            logger.error(f"Failed to initialize Azure Router client: {e}")
            raise

    def _init_embedding_client(self):
        """Initialize embedding client (uses Azure Router)."""
        try:
            self.embedding_client = AzureOpenAI(
                api_key=config.ai_service_api_key,
                azure_endpoint=config.ai_service_endpoint,
                api_version=config.ai_service_api_version
            )
            logger.info("Initialized embedding client with Azure Router")
        except Exception as e:
            logger.warning(f"Failed to initialize embedding client: {e}")
            self.embedding_client = None

    def _load_articles(self):
        """Load articles from the specified path."""
        if not self.vector_db_path or not os.path.exists(self.vector_db_path):
            logger.warning(f"Vector database not found at {self.vector_db_path}")
            return
        
        try:
            with open(self.vector_db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # Direct list of articles
                self.articles = data
                logger.info(f"Loaded {len(self.articles)} articles from vector database")
            elif isinstance(data, dict) and 'articles' in data:
                # Dictionary with articles key
                self.articles = data['articles']
                logger.info(f"Loaded {len(self.articles)} articles from vector database")
            elif isinstance(data, dict) and 'embeddings' in data:
                # Dictionary with only embeddings (no articles)
                self.articles = []  # No articles available, only embeddings
                logger.info("Vector database contains only embeddings, no articles available")
            else:
                logger.error(f"Unexpected data format in vector database: {type(data)}")
                return
            
            # Load embeddings if available
            if isinstance(data, dict) and 'embeddings' in data:
                embeddings_data = data['embeddings']
                self.embeddings = {}
                self.article_paths = []
                
                for article_path, embedding in embeddings_data.items():
                    if isinstance(embedding, list) and len(embedding) > 0:
                        self.embeddings[article_path] = np.array(embedding, dtype=np.float32)
                        self.article_paths.append(article_path)
                        logger.debug(f"Loaded embedding for: {article_path}")
                    else:
                        logger.warning(f"Invalid embedding format for {article_path}: {type(embedding)}")
                
                logger.info(f"Loaded {len(self.embeddings)} embeddings from vector database")
                logger.info(f"Article paths: {len(self.article_paths)}")
                if self.article_paths:
                    logger.info(f"First few article paths: {self.article_paths[:3]}")
            else:
                # Generate embeddings for articles
                self._generate_embeddings()
                
        except Exception as e:
            logger.error(f"Error loading articles from vector database: {e}")

    def _generate_embeddings(self):
        """Generate embeddings for all articles."""
        if not self.articles:
            return
        
        logger.info("Generating embeddings for articles...")
        self.embeddings = []
        
        for i, article in enumerate(self.articles):
            try:
                # Use full text for embedding
                text = self._read_article_full_text(article)
                if not text:
                    text = article.get('content_summary', '')
                
                embedding = self._generate_embedding(text)
                if embedding:
                    self.embeddings.append(embedding)
                else:
                    # Fallback to zero vector
                    self.embeddings.append([0.0] * 1536)  # Default embedding dimension
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated embeddings for {i + 1}/{len(self.articles)} articles")
                    
            except Exception as e:
                logger.warning(f"Error generating embedding for article {i}: {e}")
                self.embeddings.append([0.0] * 1536)
        
        logger.info(f"Generated {len(self.embeddings)} embeddings")

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text, matching the dimensions of stored article embeddings."""
        try:
            # Check what dimensions the article embeddings have
            article_dimensions = None
            if self.article_paths and self.embeddings:
                sample_article_embedding = self.embeddings[self.article_paths[0]]
                article_dimensions = len(sample_article_embedding)
                logger.debug(f"Article embeddings have {article_dimensions} dimensions")
            
            # Always use all-MiniLM-L6-v2 for consistent 384-dimensional embeddings
            # This ensures proper semantic matching with existing article embeddings
            if self.local_embedder is not None:
                local_embedding = self.local_embedder.encode([text], normalize_embeddings=False)
                if isinstance(local_embedding, np.ndarray):
                    embedding_list = local_embedding[0].tolist()
                else:
                    embedding_list = list(local_embedding[0])
                
                logger.debug(f"Generated all-MiniLM-L6-v2 embedding with {len(embedding_list)} dimensions")
                return embedding_list
            else:
                logger.warning("No all-MiniLM-L6-v2 embedder available for consistent embeddings")
                return None
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def search_articles_advanced(self, query: str, incident_context: str = "", top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Advanced article search using multi-stage AI pipeline.
        
        Args:
            query: Search query
            incident_context: Additional context about the incident
            top_k: Number of results to return
            
        Returns:
            List of relevant articles with advanced scoring and explanations
        """
        if not self.embeddings:
            logger.warning("No embeddings available for search")
            return []
        
        try:
            # Stage 1: Semantic search to get initial candidates
            logger.info("Stage 1: Performing semantic search...")
            candidates = self._semantic_search(query, top_k * 3)  # Get more candidates for LLM processing
            
            # If semantic search fails, fall back to TF-IDF
            if not candidates:
                logger.info("Semantic search failed, falling back to TF-IDF search...")
                candidates = self._fallback_tfidf_search(query, top_k * 3)
            
            if not candidates:
                logger.warning("No candidates found in semantic search")
                return []
            
            # Stage 2: LLM-based relevance scoring
            logger.info("Stage 2: Performing LLM-based relevance scoring...")
            scored_candidates = self._llm_relevance_scoring(candidates, query, incident_context)
            
            # Stage 3: LLM-based re-ranking
            logger.info("Stage 3: Performing LLM-based re-ranking...")
            ranked_candidates = self._llm_re_ranking(scored_candidates, query, incident_context)
            
            # Stage 4: Generate intelligent explanations
            logger.info("Stage 4: Generating intelligent explanations...")
            final_results = self._generate_intelligent_explanations(ranked_candidates, query, incident_context)
            
            # Return top results
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in advanced article search: {e}")
            return []

    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Stage 1: Semantic search using all-MiniLM-L6-v2 embeddings."""
        try:
            # Generate query embedding using all-MiniLM-L6-v2
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []
            
            # Using all-MiniLM-L6-v2 embeddings with 384 dimensions for consistency
            article_dimensions = len(query_embedding)  # Should be 384 for all-MiniLM-L6-v2
            logger.info(f"Using all-MiniLM-L6-v2 embeddings with {article_dimensions} dimensions for consistent semantic search")
            
            # Calculate similarities with regenerated article embeddings
            similarities = []
            logger.info(f"Query embedding length: {len(query_embedding)}")
            logger.info(f"Number of article paths: {len(self.article_paths)}")
            
            for i, article_path in enumerate(self.article_paths):
                # Generate fresh embedding for this article using all-MiniLM-L6-v2
                article_content = self._get_article_summary(article_path)
                if not article_content:
                    logger.debug(f"No content available for article {i}: {article_path}")
                    continue
                
                article_embedding = self._generate_embedding(article_content)
                if not article_embedding:
                    logger.debug(f"Failed to generate embedding for article {i}: {article_path}")
                    continue
                
                if len(article_embedding) != len(query_embedding):
                    logger.warning(f"Dimension mismatch for article {i}: query={len(query_embedding)}, article={len(article_embedding)}")
                    continue
                
                similarity = cosine_similarity(
                    [query_embedding], 
                    [article_embedding]
                )[0][0]
                
                similarities.append((i, similarity))
                logger.debug(f"Article {i} similarity: {similarity:.4f}")
            
            logger.info(f"Calculated similarities for {len(similarities)} articles")
            
            # Sort by similarity and get top candidates
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Debug logging
            if similarities:
                logger.info(f"Top 5 similarities: {[s[1] for s in similarities[:5]]}")
                logger.info(f"Max similarity: {similarities[0][1]:.4f}")
            else:
                logger.warning("No similarities calculated")
            
            candidates = []
            for i, similarity in similarities[:top_k]:
                if similarity > 0.1:  # Reasonable similarity threshold
                    article_path = self.article_paths[i]
                    candidate = {
                        'article_path': article_path,
                        'semantic_similarity': similarity,
                        'title': self._extract_title_from_path(article_path),
                        'content_summary': self._get_article_summary(article_path)
                    }
                    candidates.append(candidate)
            
            logger.info(f"Semantic search returned {len(candidates)} candidates above threshold 0.1")
            return candidates
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def _llm_relevance_scoring(self, candidates: List[Dict[str, Any]], query: str, incident_context: str) -> List[Dict[str, Any]]:
        """Stage 2: LLM-based relevance scoring."""
        scored_candidates = []
        llm_failed = False
        
        for candidate in candidates:
            try:
                # Create detailed prompt for relevance scoring
                scoring_prompt = self._create_relevance_scoring_prompt(candidate, query, incident_context)
                
                # Call LLM for scoring
                params = {
                    "model": self._get_model_name(),
                    "messages": [
                        {"role": "system", "content": "You are an expert technical support analyst specializing in incident troubleshooting and knowledge base article relevance assessment."},
                        {"role": "user", "content": scoring_prompt}
                    ]
                }
                # Use Azure Router (GPT-5) parameters
                params["max_tokens"] = 100
                params["temperature"] = 0.1
                
                logger.debug(f"Calling LLM for scoring article: {candidate.get('title', 'Unknown')}")
                response = self.client.chat.completions.create(**params)
                
                # Extract score from response
                response_text = response.choices[0].message.content
                
                # Debug Azure 5 empty responses
                if not response_text or response_text.strip() == "":
                    logger.debug(f"Azure 5 returned empty response for {candidate.get('title')}. Raw response: {response}")
                    llm_failed = True
                    break
                
                score = self._extract_relevance_score(response_text)
                
                # Validate score extraction
                if score == 5.0 and "Score:" not in response_text and not any(str(i) in response_text for i in range(11)):
                    logger.warning(f"LLM response may be invalid for {candidate.get('title')}: {response_text[:100]}...")
                    llm_failed = True
                    break
                
                candidate['llm_relevance_score'] = score
                candidate['llm_reasoning'] = response_text
                scored_candidates.append(candidate)
                
                logger.debug(f"Scored article '{candidate['title']}' with score {score}")
                
            except Exception as e:
                logger.error(f"Error scoring article {candidate.get('title')}: {e}")
                llm_failed = True
                break
        
        # If LLM failed for any article, use fallback for all
        if llm_failed or not scored_candidates:
            logger.info("LLM scoring failed, using heuristic fallback for all candidates")
            return self._apply_heuristic_scoring(candidates, query, incident_context)
        
        return scored_candidates

    def _llm_re_ranking(self, candidates: List[Dict[str, Any]], query: str, incident_context: str) -> List[Dict[str, Any]]:
        """Stage 3: LLM-based re-ranking using pairwise comparisons."""
        if len(candidates) <= 1:
            return candidates
        
        try:
            # Sort by LLM relevance score first
            candidates.sort(key=lambda x: x.get('llm_relevance_score', 0), reverse=True)
            
            # Use pairwise comparison for fine-tuning
            def compare_articles(a, b):
                try:
                    comparison_prompt = self._create_pairwise_comparison_prompt(a, b, query, incident_context)
                    
                    params = {
                        "model": self._get_model_name(),
                        "messages": [
                            {"role": "system", "content": "You are an expert technical support analyst. Compare articles and determine which is more relevant to the incident."},
                            {"role": "user", "content": comparison_prompt}
                        ]
                    }
                    # Use Azure Router (GPT-5) parameters
                    params["max_tokens"] = 50
                    params["temperature"] = 0.1
                    
                    response = self.client.chat.completions.create(**params)
                    result = self._extract_comparison_result(response.choices[0].message.content)
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"Error in pairwise comparison: {e}")
                    return 0  # No change in order
            
            # Apply pairwise comparison sorting
            from functools import cmp_to_key
            candidates.sort(key=cmp_to_key(compare_articles), reverse=True)
            
            logger.info("LLM-based re-ranking completed")
            return candidates
            
        except Exception as e:
            logger.warning(f"LLM re-ranking failed: {e}, using original order")
            return candidates

    def _generate_intelligent_explanations(self, candidates: List[Dict[str, Any]], query: str, incident_context: str) -> List[Dict[str, Any]]:
        """Stage 4: Generate intelligent explanations for why each article is relevant."""
        final_results = []
        explanation_failed = False
        
        for candidate in candidates:
            try:
                # Create explanation prompt
                explanation_prompt = self._create_explanation_prompt(candidate, query, incident_context)
                
                # Call LLM for explanation
                params = {
                    "model": self._get_model_name(),
                    "messages": [
                        {"role": "system", "content": "You are an expert technical support analyst specializing in incident troubleshooting and knowledge base article relevance assessment. Provide clear, technical explanations for why articles are relevant to specific incidents."},
                        {"role": "user", "content": explanation_prompt}
                    ]
                }
                # Use Azure Router (GPT-5) parameters
                params["max_tokens"] = 200
                params["temperature"] = 0.3
                
                logger.debug(f"Calling LLM for explanation: {candidate.get('title', 'Unknown')}")
                response = self.client.chat.completions.create(**params)
                
                # Add intelligent explanation
                explanation = response.choices[0].message.content.strip()
                
                # Debug Azure 5 empty responses
                if not explanation or explanation == "":
                    logger.debug(f"Azure 5 returned empty explanation for {candidate.get('title')}. Raw response: {response}")
                    explanation_failed = True
                    break
                
                if len(explanation) < 10:
                    logger.warning(f"LLM returned short explanation for {candidate.get('title')}: {explanation}")
                    explanation_failed = True
                    break
                
                candidate['intelligent_explanation'] = explanation
                candidate['relevance_score'] = candidate.get('llm_relevance_score', 0)
                final_results.append(candidate)
                
                logger.debug(f"Generated explanation for article '{candidate['title']}': {explanation[:50]}...")
                
            except Exception as e:
                logger.error(f"Error generating explanation for {candidate.get('title')}: {e}")
                explanation_failed = True
                break
        
        # If explanation generation failed, use heuristic explanations for all
        if explanation_failed or not final_results:
            logger.info("LLM explanation generation failed, using heuristic explanations for all candidates")
            return self._apply_heuristic_explanations(candidates, query, incident_context)
        
        return final_results

    def _apply_heuristic_scoring(self, candidates: List[Dict[str, Any]], query: str, incident_context: str) -> List[Dict[str, Any]]:
        """Apply heuristic scoring when LLM fails."""
        # Use semantic similarities and normalize them
        sims = [c.get('semantic_similarity', 0.0) for c in candidates]
        norm = self._normalize_scores(sims, 3.0, 9.0) # Normalize to a range for better distribution
        
        for i, candidate in enumerate(candidates):
            # Add token overlap as tie-breaker
            overlap = self._token_overlap_score(query + "\n" + incident_context, candidate.get('content_summary', ''))
            score = norm[i] + min(1.0, overlap * 2.0) # Add a bonus for token overlap
            
            candidate['llm_relevance_score'] = round(min(10.0, score), 2)
            candidate['llm_reasoning'] = f"Heuristic: semantic similarity {candidate.get('semantic_similarity', 0):.3f} + token overlap {overlap:.3f}"
            
            logger.debug(f"Heuristic score for '{candidate['title']}': {candidate['llm_relevance_score']}")
        
        return candidates

    def _apply_heuristic_explanations(self, candidates: List[Dict[str, Any]], query: str, incident_context: str) -> List[Dict[str, Any]]:
        """Apply heuristic explanations when LLM fails."""
        for candidate in candidates:
            # Generate explanation based on technical term matches and content analysis
            title = candidate.get('title', '')
            summary = candidate.get('content_summary', '')
            score = candidate.get('llm_relevance_score', 0)
            
            # Extract key terms from query and context (stopwords removed)
            sw = self._stopwords()
            def terms(text: str) -> List[str]:
                raw = re.findall(r'[a-z0-9_\-]+', text.lower())
                return [t for t in raw if len(t) > 2 and t not in sw]
            query_terms = set(terms(query + " " + incident_context))
            title_terms = set(terms(title))
            summary_terms = set(terms(summary))
            
            # Find matching technical terms
            matches = list(query_terms.intersection(title_terms.union(summary_terms)))
            matches.sort(key=lambda t: (-len(t), t))
            match_list = matches[:5]
            
            if match_list:
                explanation = f"Article matches incident based on technical terms: {', '.join(match_list)}. "
            else:
                explanation = "Article relevant based on semantic similarity. "
            
            # Add platform/component context if available
            if any(term in summary.lower() for term in ['linux','windows','debian','rhel','ubuntu','centos','redhat','macos']):
                explanation += "Platform-specific troubleshooting guidance. "
            if any(term in summary.lower() for term in ['defender','mde','endpoint','security','edr','av','antivirus']):
                explanation += "Security product troubleshooting. "
            if any(term in summary.lower() for term in ['policy','configuration','deployment','onboarding','selinux','kernel']):
                explanation += "Configuration and deployment guidance. "
            
            explanation += f"Relevance score: {score:.1f}/10."
            
            candidate['intelligent_explanation'] = explanation
            candidate['relevance_score'] = score
            
            logger.debug(f"Heuristic explanation for '{candidate['title']}': {explanation[:50]}...")
        
        return candidates

    def _stopwords(self) -> set:
        return {
            'the','and','for','with','this','that','from','into','your','their','have','has','had','not','are','was','were','but','you','will','can','should','could','would','about','such','than','then','them','they','its','also','more','most','some','any','all','each','other','only','very','over','under','out','our','how','why','what','when','where','which','whose','after','before','during','while','among','between','against','without','within','per','via','of','to','in','on','at','by','as','is','it','be','or','an','a'
        }

    def _token_overlap_score(self, a: str, b: str) -> float:
        """Simple Jaccard token overlap between two texts (0..1), stopwords removed."""
        try:
            sw = self._stopwords()
            ta = {t for t in re.findall(r"[a-z0-9_]+", a.lower()) if len(t) > 2 and t not in sw}
            tb = {t for t in re.findall(r"[a-z0-9_]+", b.lower()) if len(t) > 2 and t not in sw}
            if not ta or not tb:
                return 0.0
            inter = len(ta & tb)
            union = len(ta | tb)
            return float(inter) / float(union) if union else 0.0
        except Exception:
            return 0.0

    def _extract_title_from_path(self, article_path: str) -> str:
        """Derive a human-friendly title from an article path or filename."""
        try:
            name = os.path.basename(article_path)
            # Strip extension
            if "." in name:
                name = name.rsplit(".", 1)[0]
            # Replace separators with spaces
            name = name.replace("_", " ").replace("-", " ")
            # Compact multiple spaces
            name = re.sub(r"\s+", " ", name).strip()
            # Title case but keep acronyms
            title = " ".join([w if w.isupper() else w.capitalize() for w in name.split(" ")])
            return title or article_path
        except Exception:
            return article_path

    def _get_article_summary(self, article_path: str, max_chars: int = 600) -> str:
        """Read a brief summary from the beginning of the article file if available."""
        try:
            base_path = "/Users/kirill/Documents/M/Code/AzureDevops/ArticlesInventorizer"
            file_path = os.path.join(base_path, article_path)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read(max_chars * 3)  # read a chunk, then trim
                    text = text.strip()
                    if len(text) > max_chars:
                        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
                    return text
        except Exception:
            pass
        return ""

    def _fallback_tfidf_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback retrieval using TF-IDF over local article files when embeddings are incompatible."""
        try:
            if not hasattr(self, "article_paths") or not self.article_paths:
                logger.warning("TF-IDF fallback: no article paths available")
                return []

            base_path = "/Users/kirill/Documents/M/Code/AzureDevops/ArticlesInventorizer"
            documents: List[str] = []
            valid_paths: List[str] = []
            for p in self.article_paths:
                file_path = os.path.join(base_path, p)
                try:
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            if content and content.strip():
                                documents.append(content)
                                valid_paths.append(p)
                except Exception:
                    continue

            if not documents:
                logger.warning("TF-IDF fallback: no readable documents")
                return []

            vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
            doc_matrix = vectorizer.fit_transform(documents)
            query_vec = vectorizer.transform([query])
            sims = (doc_matrix @ query_vec.T).toarray().ravel()

            ranked = np.argsort(-sims)
            candidates: List[Dict[str, Any]] = []
            for idx in ranked[:top_k * 2]:
                sim = float(sims[idx])
                if sim <= 0:
                    continue
                article_path = valid_paths[idx]
                candidates.append({
                    "article_path": article_path,
                    "semantic_similarity": sim,
                    "title": self._extract_title_from_path(article_path),
                    "content_summary": self._get_article_summary(article_path)
                })

            logger.info(f"TF-IDF fallback returned {len(candidates[:top_k])} candidates")
            return candidates[:top_k]
        except Exception as e:
            logger.error(f"Error in TF-IDF fallback: {e}")
            return []

    def _create_relevance_scoring_prompt(self, candidate: Dict[str, Any], query: str, incident_context: str) -> str:
        """Create prompt for LLM relevance scoring."""
        return f"""
# Technical Support Article Relevance Assessment

## Incident Query
{query}

## Incident Context
{incident_context}

## Article to Evaluate
**Title:** {candidate.get('title', 'Unknown')}
**URL:** {candidate.get('url', 'Unknown')}
**Content Summary:** {candidate.get('content_summary', 'No summary available')}

## Scoring Criteria
Rate the relevance of this article to the incident on a scale of 0-10:

- **10**: Directly addresses the exact issue described in the incident
- **8-9**: Highly relevant troubleshooting guide for similar issues
- **6-7**: Moderately relevant, contains useful information for the problem domain
- **4-5**: Somewhat relevant, may contain tangential information
- **2-3**: Barely relevant, minimal connection to the incident
- **0-1**: Not relevant, no connection to the incident

Consider:
- Technical compatibility (platforms, versions, components)
- Problem domain alignment (security, performance, configuration, etc.)
- Troubleshooting methodology applicability
- Specificity of the solution to the described issue

## Response Format
Provide your score as a single integer between 0 and 10, followed by a brief explanation of your reasoning.

Score: [0-10]
Reasoning: [Brief explanation of why this score was assigned]
"""
    
    def _create_pairwise_comparison_prompt(self, candidate_a: Dict[str, Any], candidate_b: Dict[str, Any], query: str, incident_context: str) -> str:
        """Create prompt for pairwise comparison of candidates."""
        return f"""
# Technical Support Article Comparison

## Incident Query
{query}

## Incident Context
{incident_context}

## Article A
**Title:** {candidate_a.get('title', 'Unknown')}
**Content Summary:** {candidate_a.get('content_summary', 'No summary available')}
**Current Score:** {candidate_a.get('llm_relevance_score', 0)}

## Article B
**Title:** {candidate_b.get('title', 'Unknown')}
**Content Summary:** {candidate_b.get('content_summary', 'No summary available')}
**Current Score:** {candidate_b.get('llm_relevance_score', 0)}

## Task
Compare these two articles and determine which is more relevant to the incident.

Consider:
- Technical specificity to the described problem
- Troubleshooting methodology alignment
- Platform/component compatibility
- Solution completeness and applicability

## Response Format
Respond with:
- "1" if Article A is more relevant
- "-1" if Article B is more relevant  
- "0" if both articles are equally relevant

Response: [1, -1, or 0]
"""
    
    def _create_explanation_prompt(self, candidate: Dict[str, Any], query: str, incident_context: str) -> str:
        """Create prompt for generating intelligent explanations."""
        return f"""
# Technical Support Article Relevance Explanation

## Incident Query
{query}

## Incident Context
{incident_context}

## Article
**Title:** {candidate.get('title', 'Unknown')}
**URL:** {candidate.get('url', 'Unknown')}
**Content Summary:** {candidate.get('content_summary', 'No summary available')}
**Relevance Score:** {candidate.get('llm_relevance_score', 0)}/10

## Task
Explain why this article is relevant to the incident in 2-3 sentences. Focus on:
- Specific technical aspects that match the incident
- Troubleshooting steps that apply to the described problem
- Platform, component, or domain alignment
- How this article would help resolve the incident

Be specific and technical, avoiding generic statements.

## Response
Provide a clear, concise explanation of the relevance:
"""
    
    def _extract_relevance_score(self, response: str) -> float:
        """Extract relevance score from LLM response."""
        try:
            # Look for score patterns
            score_patterns = [
                r'Score:\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)/10',
                r'(\d+(?:\.\d+)?)\s*out\s*of\s*10',
                r'Relevance:\s*(\d+(?:\.\d+)?)',
                r'^(\d+(?:\.\d+)?)\s*$'
            ]
            
            for pattern in score_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    score = float(match.group(1))
                    return max(0, min(10, score))  # Clamp between 0 and 10
            
            # If no pattern matches, try to extract any number
            numbers = re.findall(r'\d+(?:\.\d+)?', response)
            if numbers:
                score = float(numbers[0])
                return max(0, min(10, score))
            
            # Default fallback
            return 5.0
            
        except Exception as e:
            logger.warning(f"Error extracting relevance score: {e}")
            return 5.0
    
    def _normalize_scores(self, values: List[float], lo: float = 3.0, hi: float = 9.0) -> List[float]:
        """Min-max normalize a list of values to [lo, hi]."""
        if not values:
            return []
        vmin = min(values)
        vmax = max(values)
        if vmax - vmin < 1e-8:
            mid = (lo + hi) / 2.0
            return [mid for _ in values]
        return [lo + (x - vmin) * (hi - lo) / (vmax - vmin) for x in values]
    
    def _extract_comparison_result(self, response: str) -> int:
        """Extract comparison result from LLM response."""
        try:
            response = response.strip().lower()
            
            if '1' in response and '-1' not in response:
                return 1
            elif '-1' in response:
                return -1
            elif '0' in response or 'equal' in response:
                return 0
            else:
                # Default to 0 if unclear
                return 0
                
        except Exception as e:
            logger.warning(f"Error extracting comparison result: {e}")
            return 0
    
    def _get_model_name(self) -> str:
        """Get the Azure Router model name for LLM calls."""
        if hasattr(self, 'deployment_name') and self.deployment_name:
            return self.deployment_name
        return config.ai_service_deployment_name
    
    def format_search_results(self, results: List[Dict[str, Any]], query: str = "", include_explanations: bool = True) -> str:
        """
        Format search results for display with advanced explanations.
        
        Args:
            results: List of search results from search_articles_advanced()
            query: The original search query for context
            include_explanations: Whether to include intelligent explanations
            
        Returns:
            Formatted string with search results
        """
        if not results:
            return "No relevant articles found."
        
        formatted_results = []
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Unknown Title')
            url = result.get('url', 'No URL available')
            score = result.get('relevance_score', result.get('llm_relevance_score', 0))
            explanation = result.get('intelligent_explanation', 'No explanation available')
            
            # Format the result
            result_text = f"{i}. {title}\n"
            result_text += f"   URL: {url}\n"
            result_text += f"   Relevance Score: {score:.1f}/10\n"
            
            if include_explanations and explanation:
                result_text += f"   Why relevant: {explanation}\n"
            
            formatted_results.append(result_text)
        
        return "\n".join(formatted_results)
    
    # ========================================================================
    # BACKWARD COMPATIBILITY METHODS
    # Simple TF-IDF search for legacy code (e.g., gap_analysis.py)
    # ========================================================================
    
    def search_articles(self, query: str, top_k: int = 5, use_azure_openai: bool = False) -> List[Dict[str, Any]]:
        """
        Simple article search using TF-IDF (backward compatible method).
        
        This is the simple, fast search method that doesn't use LLM ranking.
        For better results with AI-powered ranking, use search_articles_advanced().
        
        Args:
            query: The search query (incident description)
            top_k: Number of top results to return
            use_azure_openai: Ignored for backward compatibility (uses TF-IDF always)
            
        Returns:
            List of relevant articles with relevance scores
        """
        logger.info("Using simple TF-IDF search (backward compatible mode)")
        return self.search_articles_tfidf(query, top_k)
    
    def search_articles_tfidf(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant articles using TF-IDF similarity.
        Fast fallback method that doesn't require LLM calls.
        
        Args:
            query: The search query (incident description)
            top_k: Number of top results to return
            
        Returns:
            List of relevant articles with relevance scores
        """
        if not self.article_paths:
            logger.warning("No articles available for TF-IDF search")
            return []
        
        try:
            base_path = "/Users/kirill/Documents/M/Code/AzureDevops/ArticlesInventorizer"
            documents: List[str] = []
            valid_paths: List[str] = []
            
            # Load article content
            for p in self.article_paths:
                file_path = os.path.join(base_path, p)
                try:
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            if content and content.strip():
                                documents.append(content)
                                valid_paths.append(p)
                except Exception as e:
                    logger.debug(f"Error reading {p}: {e}")
                    continue
            
            if not documents:
                logger.warning("No readable documents for TF-IDF search")
                return []
            
            # Add query to documents
            documents.append(query)
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Calculate similarities
            query_vector = tfidf_matrix[-1]  # Last document is the query
            similarities = cosine_similarity(query_vector, tfidf_matrix[:-1])[0]
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include articles with some relevance
                    article_path = valid_paths[idx]
                    result = {
                        'title': self._extract_title_from_path(article_path),
                        'url': article_path,
                        'article_path': article_path,
                        'content_summary': self._get_article_summary(article_path),
                        'relevance_score': float(similarities[idx]) * 10,  # Scale to 0-10
                        'semantic_similarity': float(similarities[idx])
                    }
                    results.append(result)
            
            logger.info(f"TF-IDF search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in TF-IDF search: {e}")
            return []
    
    def get_full_article_content(self, url: str) -> str:
        """
        Get the full article content for detailed analysis.
        Backward compatible method from legacy ArticleSearcher.
        
        Args:
            url: Article URL/path
            
        Returns:
            Full article content as string
        """
        try:
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


# Backward compatibility alias
AdvancedArticleSearcher = ArticleSearcher
