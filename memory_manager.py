import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from mem0 import Memory
from openai import AzureOpenAI
from config import config

logger = logging.getLogger(__name__)

class SummarizerMemoryManager:
    """
    Memory manager for the incident summarizer using mem0.
    Provides persistent memory across processing sessions for better context awareness.
    """
    
    def __init__(self, user_id: str = "summarizer_user"):
        """
        Initialize the memory manager.
        
        Args:
            user_id: Unique identifier for the summarizer user/session
        """
        self.user_id = user_id
        
        # Configure mem0 to use Azure OpenAI for embeddings
        if config.azure_openai_embedding_deployment_name:
            # Use full mem0 with Azure OpenAI embeddings
            import os
            # Embedding environment variables
            os.environ['EMBEDDING_AZURE_OPENAI_API_KEY'] = config.azure_openai_api_key
            os.environ['EMBEDDING_AZURE_ENDPOINT'] = config.azure_openai_endpoint
            os.environ['EMBEDDING_AZURE_API_VERSION'] = config.azure_openai_api_version
            os.environ['EMBEDDING_AZURE_DEPLOYMENT'] = config.azure_openai_embedding_deployment_name
            
            # LLM environment variables
            os.environ['LLM_AZURE_OPENAI_API_KEY'] = config.azure_openai_api_key
            os.environ['LLM_AZURE_ENDPOINT'] = config.azure_openai_endpoint
            os.environ['LLM_AZURE_API_VERSION'] = config.azure_openai_api_version
            os.environ['LLM_AZURE_DEPLOYMENT'] = config.azure_openai_deployment_name
            
            # Initialize mem0 with Azure OpenAI configuration
            from mem0.configs.base import MemoryConfig
            from mem0.embeddings.configs import EmbedderConfig
            from mem0.llms.configs import LlmConfig
            
            memory_config = MemoryConfig(
                embedder=EmbedderConfig(
                    provider="azure_openai"
                ),
                llm=LlmConfig(
                    provider="azure_openai"
                )
            )
            
            self.memory = Memory(memory_config)
            self.memories = []  # Not needed with mem0
            logger.info("Initialized mem0 with Azure OpenAI embeddings")
            print(f"✅ Using mem0 with Azure OpenAI embeddings for memory storage and semantic search")
        else:
            # Fallback to file-based memory system
            self.memory = None
            self.memories = []
            self._load_memories()
            logger.info("Using file-based memory system (no embedding model configured)")
            print(f"📁 Using file-based memory system (no embedding model configured)")
        
        # Initialize Azure OpenAI client for other operations
        self.openai_client = AzureOpenAI(
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            azure_endpoint=config.azure_openai_endpoint
        )
        
        # Load memory configuration
        self.memory_config = self._load_memory_config()
        
        logger.info(f"Initialized SummarizerMemoryManager for user: {user_id}")
    
    def _load_memory_config(self) -> Dict[str, Any]:
        """Load memory configuration from file."""
        try:
            # Create memory directory if it doesn't exist
            memory_dir = "memory"
            os.makedirs(memory_dir, exist_ok=True)
            
            config_path = os.path.join(memory_dir, "memory_config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                logger.info("Loaded memory configuration")
                return config
        except FileNotFoundError:
            logger.warning("memory_config.json not found, using default configuration")
            return {
                "memory_integration": {
                    "enabled": True,
                    "priority": "complementary",
                    "max_memory_context_length": 1000,
                    "memory_search_limit": 3,
                    "enable_memory_for_molecular": True
                },
                "molecular_integration": {
                    "preserve_molecular_examples": True,
                    "molecular_examples_priority": "high",
                    "allow_memory_enhancement": True
                }
            }
        except Exception as e:
            logger.error(f"Error loading memory configuration: {e}, using default")
            return {"memory_integration": {"enabled": True}, "molecular_integration": {"preserve_molecular_examples": True}}
    
    def _load_memories(self):
        """Load memories from file."""
        try:
            # Create memory directory if it doesn't exist
            memory_dir = "memory"
            os.makedirs(memory_dir, exist_ok=True)
            
            memory_file = os.path.join(memory_dir, f"memory_{self.user_id}.json")
            if os.path.exists(memory_file):
                with open(memory_file, "r", encoding="utf-8") as f:
                    self.memories = json.load(f)
                logger.info(f"Loaded {len(self.memories)} memories from file")
            else:
                self.memories = []
                logger.info("No existing memories found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
            self.memories = []
    
    def _save_memories(self):
        """Save memories to file."""
        try:
            memory_dir = "memory"
            os.makedirs(memory_dir, exist_ok=True)
            
            memory_file = os.path.join(memory_dir, f"memory_{self.user_id}.json")
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.memories)} memories to file")
        except Exception as e:
            logger.error(f"Error saving memories: {e}")
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts using word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def add_incident_memory(self, incident_number: str, incident_data: Dict[str, Any], 
                           processing_result: Dict[str, Any]) -> None:
        """
        Store memory about a processed incident.
        
        Args:
            incident_number: The incident number
            incident_data: Raw incident data
            processing_result: The processing result/summary
        """
        try:
            if self.memory:
                # Use mem0 for storage
                memory_content = {
                    "incident_number": incident_number,
                    "timestamp": datetime.now().isoformat(),
                    "incident_type": incident_data.get("type", "unknown"),
                    "severity": incident_data.get("severity", "unknown"),
                    "summary": processing_result.get("summary", ""),
                    "key_findings": processing_result.get("key_findings", []),
                    "technical_details": processing_result.get("technical_details", {}),
                    "recommendations": processing_result.get("recommendations", [])
                }
                
                # Convert to string for storage
                memory_text = json.dumps(memory_content, indent=2)
                
                # Add to mem0 memory
                self.memory.add(
                    messages=[{"role": "system", "content": f"Incident processing result: {memory_text}"}],
                    user_id=self.user_id
                )
                
                logger.info(f"Added memory for incident {incident_number} using mem0")
            else:
                # Use file-based storage
                memory_entry = {
                    "incident_number": incident_number,
                    "timestamp": datetime.now().isoformat(),
                    "incident_type": incident_data.get("type", "unknown"),
                    "severity": incident_data.get("severity", "unknown"),
                    "description": incident_data.get("description", ""),
                    "summary": processing_result.get("summary", ""),
                    "key_findings": processing_result.get("key_findings", []),
                    "technical_details": processing_result.get("technical_details", {}),
                    "recommendations": processing_result.get("recommendations", [])
                }
                
                # Add to memories list
                self.memories.append(memory_entry)
                
                # Save to file
                self._save_memories()
                
                logger.info(f"Added memory for incident {incident_number} using file-based storage")
            
        except Exception as e:
            logger.error(f"Failed to add memory for incident {incident_number}: {e}")
    
    def search_relevant_memories(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Search for relevant memories based on a query.
        
        Args:
            query: Search query
            limit: Maximum number of memories to return (uses config default if None)
            
        Returns:
            List of relevant memories
        """
        try:
            # Use configured limit if not specified
            if limit is None:
                limit = self.memory_config.get("memory_integration", {}).get("memory_search_limit", 3)
            
            if self.memory:
                # Use mem0 for search
                results = self.memory.search(
                    query=query,
                    user_id=self.user_id,
                    limit=limit
                )
                
                logger.info(f"Found {len(results.get('results', []))} relevant memories for query: {query}")
                return results.get('results', [])
            else:
                # Use file-based search
                if not self.memories:
                    return []
                
                # Calculate similarity scores for all memories
                scored_memories = []
                for memory in self.memories:
                    # Create searchable text from memory
                    searchable_text = f"{memory.get('incident_type', '')} {memory.get('severity', '')} {memory.get('description', '')} {memory.get('summary', '')}"
                    
                    # Calculate similarity
                    similarity = self._simple_similarity(query, searchable_text)
                    scored_memories.append((similarity, memory))
                
                # Sort by similarity score (highest first) and return top results
                scored_memories.sort(key=lambda x: x[0], reverse=True)
                
                # Return top results
                results = [memory for score, memory in scored_memories[:limit] if score > 0.1]  # Only return if similarity > 10%
                
                logger.info(f"Found {len(results)} relevant memories for query: {query}")
                return results
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    def get_context_for_incident(self, incident_data: Dict[str, Any]) -> str:
        """
        Get relevant context from previous incidents for processing a new incident.
        
        Args:
            incident_data: The incident data to find context for
            
        Returns:
            Formatted context string
        """
        try:
            # Extract key information for context search
            incident_type = incident_data.get("type", "")
            severity = incident_data.get("severity", "")
            description = incident_data.get("description", "")
            
            # Create search query
            search_query = f"{incident_type} {severity} {description[:200]}"
            
            # Search for relevant memories
            relevant_memories = self.search_relevant_memories(search_query, limit=3)
            
            if not relevant_memories:
                return ""
            
            # Format context
            context_parts = ["Previous similar incidents:"]
            for i, memory in enumerate(relevant_memories, 1):
                if self.memory:
                    # mem0 format
                    try:
                        memory_data = json.loads(memory.get('memory', '{}'))
                        context_parts.append(
                            f"{i}. Incident {memory_data.get('incident_number', 'unknown')}: "
                            f"{memory_data.get('summary', 'No summary available')}"
                        )
                    except json.JSONDecodeError:
                        # Fallback to raw memory text
                        context_parts.append(f"{i}. {memory.get('memory', 'No details available')}")
                else:
                    # File-based format
                    context_parts.append(
                        f"{i}. Incident {memory.get('incident_number', 'unknown')}: "
                        f"{memory.get('summary', 'No summary available')}"
                    )
            
            context = "\n".join(context_parts)
            logger.info(f"Generated context for incident processing: {len(context)} characters")
            return context
            
        except Exception as e:
            logger.error(f"Failed to get context for incident: {e}")
            return ""
    
    def enhance_prompt_with_memory(self, base_prompt: str, incident_data: Dict[str, Any], 
                                  molecular_context_used: bool = False) -> str:
        """
        Enhance a base prompt with relevant memory context.
        
        Args:
            base_prompt: The original prompt
            incident_data: The incident data
            molecular_context_used: Whether molecular context was already applied
            
        Returns:
            Enhanced prompt with memory context
        """
        try:
            # Check if memory integration is enabled
            if not self.memory_config.get("memory_integration", {}).get("enabled", True):
                logger.info("Memory integration disabled in configuration")
                return base_prompt
            
            # Check if memory should be used with molecular context
            if molecular_context_used and not self.memory_config.get("memory_integration", {}).get("enable_memory_for_molecular", True):
                logger.info("Memory integration disabled for molecular prompts in configuration")
                return base_prompt
            
            context = self.get_context_for_incident(incident_data)
            
            if not context:
                return base_prompt
            
            # Limit context length based on configuration
            max_length = self.memory_config.get("memory_integration", {}).get("max_memory_context_length", 1000)
            if len(context) > max_length:
                context = context[:max_length] + "..."
            
            # If molecular context was already used, add memory as additional context
            if molecular_context_used:
                enhanced_prompt = f"{base_prompt}\n\nAdditional context from previous similar incidents:\n{context}\n\nUse this additional context alongside the provided examples for more informed analysis."
            else:
                enhanced_prompt = f"{base_prompt}\n\nContext from previous similar incidents:\n{context}\n\nUse this context to provide more informed and consistent analysis."
            
            logger.info(f"Enhanced prompt with memory context: {len(enhanced_prompt)} characters")
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Failed to enhance prompt with memory: {e}")
            return base_prompt
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored memories.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            if self.memory:
                # mem0 system
                return {
                    "user_id": self.user_id,
                    "memory_system": "mem0",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # File-based system
                return {
                    "user_id": self.user_id,
                    "memory_system": "file-based",
                    "memory_count": len(self.memories),
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            return {}
    
    def clear_memories(self) -> bool:
        """
        Clear all memories for the current user.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.memory:
                # mem0 system - clear memories
                # Note: This would require mem0 API support for clearing memories
                logger.warning("Memory clearing not implemented for mem0 - would require API support")
                return False
            else:
                # File-based system - clear memories
                self.memories = []
                self._save_memories()
                logger.info("Cleared all file-based memories")
                return True
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return False
