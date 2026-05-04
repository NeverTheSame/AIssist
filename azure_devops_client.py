import requests
import base64
import json
import logging
import re
from urllib.parse import quote
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AzureDevOpsClient:
    """Client for interacting with Azure DevOps REST API for work item operations."""
    
    def __init__(self, org: str, project: str, pat: str):
        """
        Initialize Azure DevOps client.

        Args:
            org: Azure DevOps organization name (e.g., "my-org")
            project: Project name (e.g., "My Project")
            pat: Personal Access Token
        """
        self.org = org
        self.project = project
        self.pat = pat

        if not self.pat:
            raise ValueError("Azure DevOps PAT (AZURE_DEVOPS_PAT) is required but not found in environment variables.")
        
        # Encode PAT for Basic authentication
        credentials = f":{self.pat}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        self.headers = {
            'Authorization': f'Basic {encoded_credentials}',
            'Content-Type': 'application/json'
        }
        
        # Base URL for API calls
        self.base_url = f"https://dev.azure.com/{self.org}/{quote(project)}"
        self.api_version = "7.1"
        
        logger.info(f"Initialized Azure DevOps client for {self.org}/{self.project}")
    
    def _make_request(self, method: str, url: str, data: Optional[Dict] = None, headers: Optional[Dict] = None, timeout: int = 60) -> requests.Response:
        """Make an HTTP request to Azure DevOps API."""
        request_headers = {**self.headers}
        if headers:
            request_headers.update(headers)
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=request_headers, timeout=timeout)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=request_headers, json=data, timeout=timeout)
            elif method.upper() == 'PATCH':
                response = requests.patch(url, headers=request_headers, json=data, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout as e:
            logger.error(f"Azure DevOps API request timed out after {timeout}s: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Azure DevOps API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise
    
    def search_work_items(self, query_text: str, work_item_type: Optional[str] = None, max_results: int = 20) -> List[Dict]:
        """
        Search for work items using WIQL (Work Item Query Language).
        Uses a simplified query to avoid timeouts.
        
        Args:
            query_text: Keywords to search for in title
            work_item_type: Optional work item type filter (e.g., "Task", "Bug", "Feature")
            max_results: Maximum number of results to return
            
        Returns:
            List of work item dictionaries with id, title, and url
        """
        # Simplify query - search only by title first to avoid timeouts
        # Use a simple, selective query that's more likely to complete quickly
        # Escape single quotes in query text
        safe_query = query_text.replace("'", "''")
        
        # Build simplified WIQL query - search only in title for better performance
        # Limit to recent items to reduce scan size
        wiql_query = f"SELECT [System.Id], [System.Title] FROM WorkItems WHERE "
        wiql_query += f"[System.TeamProject] = '{self.project}'"
        wiql_query += f" AND [System.Title] CONTAINS WORDS '{safe_query}'"
        
        # Add work item type filter if specified
        if work_item_type:
            wiql_query += f" AND [System.WorkItemType] = '{work_item_type}'"
        
        # Order by changed date (most recent first) and limit to recent items
        wiql_query += f" ORDER BY [System.ChangedDate] DESC"
        
        url = f"{self.base_url}/_apis/wit/wiql?api-version={self.api_version}"
        data = {"query": wiql_query}
        
        try:
            # Use shorter timeout for WIQL queries (they have a 30s server limit)
            response = self._make_request('POST', url, data=data, timeout=25)
            result = response.json()
            
            work_items = []
            if 'workItems' in result and result['workItems']:
                # Get work item IDs from WIQL result (limit to max_results)
                work_item_ids = [item['id'] for item in result['workItems'][:max_results]]
                
                if work_item_ids:
                    # Fetch full work item details in batches if needed
                    # Azure DevOps allows up to 200 IDs per request
                    batch_size = 200
                    all_items = []
                    
                    for i in range(0, len(work_item_ids), batch_size):
                        batch_ids = work_item_ids[i:i + batch_size]
                        ids_string = ','.join(map(str, batch_ids))
                        details_url = f"{self.base_url}/_apis/wit/workitems?ids={ids_string}&$expand=all&api-version={self.api_version}"
                        
                        try:
                            details_response = self._make_request('GET', details_url, timeout=30)
                            details_result = details_response.json()
                            
                            if 'value' in details_result:
                                all_items.extend(details_result['value'])
                        except Exception as e:
                            logger.warning(f"Error fetching batch of work items: {e}")
                            continue
                    
                    # Process all fetched items
                    for item in all_items:
                        fields = item.get('fields', {})
                        work_items.append({
                            'id': item.get('id'),
                            'title': fields.get('System.Title', ''),
                            'description': fields.get('System.Description', ''),
                            'work_item_type': fields.get('System.WorkItemType', ''),
                            'url': item.get('url', '').replace('/_apis/wit/workitems', '/_workitems/edit')
                        })
            
            logger.info(f"Found {len(work_items)} work items matching '{query_text}'")
            return work_items
            
        except requests.exceptions.Timeout:
            logger.warning(f"WIQL query timed out for '{query_text}'. Search may be too complex or dataset too large.")
            return []
        except Exception as e:
            logger.error(f"Error searching work items: {e}")
            return []
    
    def get_active_preventative_actions(self, assigned_to: str, custom_field_value: str = "KK", max_results: int = 50) -> List[Dict]:
        """
        Query work items that are active preventative actions.
        Returns only work items of type "Task" assigned to the specified user where Custom field 1 equals the specified value.
        
        Args:
            assigned_to: Full name of the assignee (e.g., "Kirill Kuklin")
            custom_field_value: Value to filter by in Custom field 1 (default: "KK")
            max_results: Maximum number of results to return
            
        Returns:
            List of work item dictionaries with id, title, state, url, etc.
        """
        # Build WIQL query for active preventative actions
        # Type = Task, Assigned to specific user, Custom field 1 = specified value, State not in (Closed, Inactive, Resolved)
        safe_name = assigned_to.replace("'", "''")
        safe_value = custom_field_value.replace("'", "''")
        
        # Try different possible field reference names for custom fields
        # Common patterns: CustomField1, CustomField01, Custom_Field_1, etc.
        # We'll try multiple patterns, starting with the most common
        field_references = [
            "[Custom.CustomField1]",
            "[Custom.CustomField01]", 
            "[Custom.Custom field 1]",
            "[Custom.Custom_Field_1]",
            "[Custom.CustomField 1]"
        ]
        
        # First, try to get work item #19394 to see the actual field reference
        actual_field_ref = None
        try:
            test_work_item = self.get_work_item(19394)
            if test_work_item and 'raw' in test_work_item:
                fields = test_work_item['raw'].get('fields', {})
                # Look for fields that have value "KK"
                for field_name, field_value in fields.items():
                    if isinstance(field_value, str) and field_value.strip() == "KK":
                        actual_field_ref = field_name
                        logger.info(f"Found custom field reference from work item 19394: {actual_field_ref} (value: '{field_value}')")
                        break
                
                # If not found by exact value match, log all custom fields for debugging
                if not actual_field_ref:
                    logger.debug("Field with value 'KK' not found. Searching all custom fields:")
                    for field_name, field_value in fields.items():
                        if "Custom" in field_name or "custom" in field_name.lower():
                            logger.debug(f"  {field_name} = '{field_value}' (type: {type(field_value).__name__})")
                            if isinstance(field_value, str) and field_value.strip() == "KK":
                                actual_field_ref = field_name
                                logger.info(f"Found custom field reference by name pattern: {actual_field_ref} (value: '{field_value}')")
                                break
        except Exception as e:
            logger.warning(f"Could not fetch work item 19394 to determine field reference: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Use the found field reference, or try common patterns
        if actual_field_ref:
            # The field reference name from API is already the correct format (e.g., "Custom.CustomField1")
            field_ref = f"[{actual_field_ref}]"
            logger.info(f"Using field reference: {field_ref}")
        else:
            # Default to first pattern if we couldn't find it
            field_ref = field_references[0]
            logger.warning(f"Could not determine custom field reference from work item 19394, will try patterns: {field_references}")
        
        wiql_query = f"SELECT [System.Id], [System.Title], [System.State] FROM WorkItems WHERE "
        wiql_query += f"[System.TeamProject] = '{self.project}'"
        wiql_query += f" AND [System.WorkItemType] = 'Task'"
        # Use CONTAINS to match the display name (handles both "Name" and "Name <email>" formats)
        wiql_query += f" AND [System.AssignedTo] CONTAINS '{safe_name}'"
        # Use the determined field reference
        wiql_query += f" AND {field_ref} = '{safe_value}'"
        wiql_query += f" AND [System.State] <> 'Closed'"
        wiql_query += f" AND [System.State] <> 'Inactive'"
        wiql_query += f" AND [System.State] <> 'Resolved'"
        wiql_query += f" ORDER BY [System.ChangedDate] DESC"
        
        url = f"{self.base_url}/_apis/wit/wiql?api-version={self.api_version}"
        
        # Try the query with the determined field reference, and if it fails, try alternatives
        last_error = None
        field_refs_to_try = [field_ref] if actual_field_ref else field_references
        successful_query = None
        
        for attempt_field_ref in field_refs_to_try:
            # Build query with current field reference
            current_wiql_query = f"SELECT [System.Id], [System.Title], [System.State] FROM WorkItems WHERE "
            current_wiql_query += f"[System.TeamProject] = '{self.project}'"
            current_wiql_query += f" AND [System.WorkItemType] = 'Task'"
            current_wiql_query += f" AND [System.AssignedTo] CONTAINS '{safe_name}'"
            current_wiql_query += f" AND {attempt_field_ref} = '{safe_value}'"
            current_wiql_query += f" AND [System.State] <> 'Closed'"
            current_wiql_query += f" AND [System.State] <> 'Inactive'"
            current_wiql_query += f" AND [System.State] <> 'Resolved'"
            current_wiql_query += f" ORDER BY [System.ChangedDate] DESC"
            
            data = {"query": current_wiql_query}
            
            try:
                # Use shorter timeout for WIQL queries (they have a 30s server limit)
                response = self._make_request('POST', url, data=data, timeout=25)
                result = response.json()
                # If we get here, the query succeeded
                successful_query = result
                logger.info(f"Successfully used field reference: {attempt_field_ref}")
                break
            except requests.exceptions.HTTPError as e:
                if e.response and e.response.status_code == 400:
                    last_error = e
                    error_body = e.response.text if hasattr(e.response, 'text') else str(e)
                    if "field that does not exist" in error_body or "TF51005" in error_body:
                        # Field reference doesn't exist, try next one
                        logger.debug(f"Field reference {attempt_field_ref} doesn't exist, trying next...")
                        continue
                    else:
                        # Different error, re-raise
                        raise
                else:
                    # Different HTTP error, re-raise
                    raise
        
        if not successful_query:
            # All field references failed
            if last_error:
                raise last_error
            else:
                raise Exception("Could not determine valid field reference")
        
        try:
            result = successful_query
            
            work_items = []
            if 'workItems' in result and result['workItems']:
                # Get work item IDs from WIQL result (limit to max_results)
                work_item_ids = [item['id'] for item in result['workItems'][:max_results]]
                
                if work_item_ids:
                    # Fetch full work item details in batches if needed
                    # Azure DevOps allows up to 200 IDs per request
                    batch_size = 200
                    all_items = []
                    
                    for i in range(0, len(work_item_ids), batch_size):
                        batch_ids = work_item_ids[i:i + batch_size]
                        ids_string = ','.join(map(str, batch_ids))
                        details_url = f"{self.base_url}/_apis/wit/workitems?ids={ids_string}&$expand=all&api-version={self.api_version}"
                        
                        try:
                            details_response = self._make_request('GET', details_url, timeout=30)
                            details_result = details_response.json()
                            
                            if 'value' in details_result:
                                all_items.extend(details_result['value'])
                        except Exception as e:
                            logger.warning(f"Error fetching batch of work items: {e}")
                            continue
                    
                    # Process all fetched items
                    for item in all_items:
                        fields = item.get('fields', {})
                        # Try to find custom fields with common reference name patterns
                        icm_incident_count = None
                        icm_incident_ids = None
                        icm_repair_item_type = None
                        
                        # Look for custom fields - check all fields and match by keywords
                        for field_name, field_value in fields.items():
                            field_name_lower = field_name.lower()
                            # Match IcM Incident Count - look for fields containing "icm", "incident", and "count"
                            if 'icm' in field_name_lower and 'incident' in field_name_lower and 'count' in field_name_lower:
                                icm_incident_count = field_value
                            # Match IcM Incident IDs - look for fields containing "icm", "incident", and "id" or "ids"
                            elif 'icm' in field_name_lower and 'incident' in field_name_lower and ('id' in field_name_lower or 'ids' in field_name_lower):
                                icm_incident_ids = field_value
                            # Match IcM Repair Item Type - look for fields containing "icm", "repair", and "type"
                            elif 'icm' in field_name_lower and 'repair' in field_name_lower and 'type' in field_name_lower:
                                icm_repair_item_type = field_value
                        
                        work_items.append({
                            'id': item.get('id'),
                            'title': fields.get('System.Title', ''),
                            'description': fields.get('System.Description', ''),
                            'work_item_type': fields.get('System.WorkItemType', ''),
                            'state': fields.get('System.State', ''),
                            'assigned_to': fields.get('System.AssignedTo', {}).get('displayName', '') if isinstance(fields.get('System.AssignedTo'), dict) else str(fields.get('System.AssignedTo', '')),
                            'url': item.get('url', '').replace('/_apis/wit/workitems', '/_workitems/edit'),
                            'icm_incident_count': icm_incident_count,
                            'icm_incident_ids': icm_incident_ids,
                            'icm_repair_item_type': icm_repair_item_type
                        })
            
            logger.info(f"Found {len(work_items)} active preventative actions assigned to '{assigned_to}'")
            return work_items
            
        except requests.exceptions.Timeout:
            logger.warning(f"WIQL query timed out for active preventative actions. Query may be too complex or dataset too large.")
            return []
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 400:
                # Bad request - might be due to incorrect field reference name
                error_msg = e.response.text if hasattr(e.response, 'text') else str(e)
                logger.error(f"Error querying active preventative actions - field reference may be incorrect: {error_msg}")
                logger.warning("If 'Custom field 1' reference failed, the field might need a different reference name in WIQL")
            else:
                logger.error(f"Error querying active preventative actions: {e}")
            return []
        except Exception as e:
            logger.error(f"Error querying active preventative actions: {e}")
            return []
    
    def get_work_item(self, work_item_id: int) -> Optional[Dict]:
        """
        Retrieve full work item details.
        
        Args:
            work_item_id: Work item ID
            
        Returns:
            Work item dictionary with all fields, or None if not found
        """
        url = f"{self.base_url}/_apis/wit/workitems/{work_item_id}?$expand=all&api-version={self.api_version}"
        
        try:
            response = self._make_request('GET', url)
            item = response.json()
            
            fields = item.get('fields', {})
            return {
                'id': item.get('id'),
                'title': fields.get('System.Title', ''),
                'description': fields.get('System.Description', ''),
                'work_item_type': fields.get('System.WorkItemType', ''),
                'state': fields.get('System.State', ''),
                'url': item.get('url', '').replace('/_apis/wit/workitems', '/_workitems/edit'),
                'raw': item  # Keep raw data for detailed comparison
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Work item {work_item_id} not found")
                return None
            raise
        except Exception as e:
            logger.error(f"Error retrieving work item {work_item_id}: {e}")
            return None
    
    def compare_preventative_actions(self, existing_work_item: Dict, new_analysis: str, processor) -> Tuple[float, str]:
        """
        Use LLM to compare existing work item with new preventative action analysis.
        
        Args:
            existing_work_item: Dictionary with work item details
            new_analysis: New preventative action analysis text
            processor: IncidentProcessor instance with LLM access
            
        Returns:
            Tuple of (similarity_score, reasoning) where score is 0.0-1.0
        """
        existing_title = existing_work_item.get('title', '')
        existing_description = existing_work_item.get('description', '')
        existing_content = f"Title: {existing_title}\n\nDescription: {existing_description}"
        
        comparison_prompt = f"""Compare the following two preventative action analyses and determine if they address the same or very similar issues.

Existing Work Item (#{existing_work_item.get('id', 'unknown')}):
{existing_content}

New Preventative Action Analysis:
{new_analysis}

Analyze both and provide:
1. A similarity score from 0.0 to 1.0 (where 1.0 means they are essentially the same preventative action)
2. Brief reasoning for your score

Consider:
- Do they recommend the same preventative action category?
- Do they address similar root causes?
- Are the specific recommendations similar or complementary?

Respond in this exact format:
SIMILARITY_SCORE: [0.0-1.0]
REASONING: [brief explanation]"""
        
        try:
            # Use processor's LLM to generate comparison
            messages = [
                {"role": "system", "content": "You are an expert at comparing preventative action analyses to identify duplicates or similar items."},
                {"role": "user", "content": comparison_prompt}
            ]
            
            # Call LLM through processor
            response = processor.client.chat.completions.create(
                model=processor.deployment_name,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
                timeout=60,
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse similarity score
            similarity_score = 0.0
            reasoning = ""
            
            for line in result_text.split('\n'):
                if line.startswith('SIMILARITY_SCORE:'):
                    try:
                        score_text = line.split(':', 1)[1].strip()
                        similarity_score = float(score_text)
                        similarity_score = max(0.0, min(1.0, similarity_score))  # Clamp to 0.0-1.0
                    except (ValueError, IndexError):
                        pass
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip() if ':' in line else ""
            
            logger.info(f"Similarity score for work item #{existing_work_item.get('id')}: {similarity_score}")
            return similarity_score, reasoning
            
        except Exception as e:
            logger.error(f"Error comparing preventative actions: {e}")
            return 0.0, f"Error during comparison: {str(e)}"
    
    def post_preventative_action(self, work_item_id: int, incident_id: str, analysis_text: str) -> bool:
        """
        Post preventative action analysis to a work item.
        
        Args:
            work_item_id: Work item ID to update
            incident_id: Incident ID being mapped
            analysis_text: Preventative action analysis text
            
        Returns:
            True if successful, False otherwise
        """
        # Format the update as a discussion comment or description addition
        comment_text = f"""
## Preventative Action Analysis for Incident {incident_id}

{analysis_text}

---
*Mapped from incident {incident_id}*
"""
        
        # Use JSON Patch format to add to discussion
        patch_operations = [
            {
                "op": "add",
                "path": "/fields/System.History",
                "value": comment_text
            }
        ]
        
        url = f"{self.base_url}/_apis/wit/workitems/{work_item_id}?api-version={self.api_version}"
        headers = {
            'Content-Type': 'application/json-patch+json'
        }
        
        try:
            response = self._make_request('PATCH', url, data=patch_operations, headers=headers)
            logger.info(f"Successfully posted analysis to work item #{work_item_id}")
            return True
        except Exception as e:
            logger.error(f"Error posting to work item #{work_item_id}: {e}")
            return False
    
    def _find_field_reference_from_work_item(self, work_item_fields: Dict, field_keywords: List[str]) -> Optional[str]:
        """
        Find the actual field reference name from a work item's fields.
        
        Args:
            work_item_fields: Dictionary of fields from a work item
            field_keywords: List of keywords that should be in the field name (case-insensitive)
            
        Returns:
            Field reference name if found, None otherwise
        """
        for field_name, field_value in work_item_fields.items():
            field_name_lower = field_name.lower()
            if all(keyword.lower() in field_name_lower for keyword in field_keywords):
                return field_name
        return None
    
    def _strip_markdown(self, text: str) -> str:
        """
        Convert markdown text to plain text by removing markdown syntax.
        
        Args:
            text: Markdown formatted text
            
        Returns:
            Plain text without markdown syntax
        """
        if not text:
            return ""
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove headers
        text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
        
        # Remove bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove horizontal rules
        text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)
        
        # Remove list markers
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text
    
    def create_preventative_action_work_item(self, title: str, icm_repair_item_type: str, incident_id: str, description: str = "", assigned_to: str = "Kirill Kuklin") -> Optional[int]:
        """
        Create a new preventative action work item with required fields.
        
        Args:
            title: Work item title
            icm_repair_item_type: Value for IcM Repair Item Type field
            incident_id: Incident ID to add to IcM Incident IDs
            description: Optional work item description (will be converted to plain text)
            assigned_to: User to assign the work item to (default: "Kirill Kuklin")
            
        Returns:
            Created work item ID, or None if creation failed
        """
        # Use common field reference patterns (Azure DevOps custom fields typically follow these patterns)
        # These will be tried in order, and if one fails, we'll get an error message
        custom_field1_ref = "Custom.CustomField1"  # Common pattern for "Custom field 1"
        icm_repair_type_ref = "Custom.IcMRepairItemType"  # Common pattern
        icm_incident_ids_ref = "Custom.IcMIncidentIDs"  # Common pattern
        icm_incident_count_ref = "Custom.IcMIncidentCount"  # Common pattern
        
        # Convert description to plain text (remove markdown)
        plain_text_description = self._strip_markdown(description) if description else ""
        
        # Use JSON Patch format to create work item
        patch_operations = [
            {
                "op": "add",
                "path": "/fields/System.Title",
                "value": title
            },
            {
                "op": "add",
                "path": "/fields/System.AssignedTo",
                "value": assigned_to
            }
        ]
        
        if plain_text_description:
            patch_operations.append({
                "op": "add",
                "path": "/fields/System.Description",
                "value": plain_text_description
            })
        
        # Add custom fields
        patch_operations.extend([
            {
                "op": "add",
                "path": f"/fields/{custom_field1_ref}",
                "value": "KK"
            },
            {
                "op": "add",
                "path": f"/fields/{icm_repair_type_ref}",
                "value": icm_repair_item_type
            },
            {
                "op": "add",
                "path": f"/fields/{icm_incident_ids_ref}",
                "value": incident_id
            },
            {
                "op": "add",
                "path": f"/fields/{icm_incident_count_ref}",
                "value": 1
            }
        ])
        
        url = f"{self.base_url}/_apis/wit/workitems/$Task?api-version={self.api_version}"
        headers = {
            'Content-Type': 'application/json-patch+json'
        }
        
        try:
            response = self._make_request('POST', url, data=patch_operations, headers=headers)
            result = response.json()
            work_item_id = result.get('id')
            logger.info(f"Successfully created preventative action work item #{work_item_id}")
            return work_item_id
        except Exception as e:
            logger.error(f"Error creating preventative action work item: {e}")
            return None
    
    def update_work_item_with_incident(self, work_item_id: int, incident_id: str) -> bool:
        """
        Update an existing work item to add incident ID to IcM Incident IDs and increment IcM Incident Count.
        
        Args:
            work_item_id: Work item ID to update
            incident_id: Incident ID to add
            
        Returns:
            True if successful, False otherwise
        """
        # Get current work item to find field references and current values
        work_item = self.get_work_item(work_item_id)
        if not work_item or 'raw' not in work_item:
            logger.error(f"Could not retrieve work item {work_item_id}")
            return False
        
        fields = work_item['raw'].get('fields', {})
        
        # Find field reference names - use the same logic as in get_active_preventative_actions
        icm_incident_ids_ref = None
        icm_incident_count_ref = None
        
        for field_name, field_value in fields.items():
            field_name_lower = field_name.lower()
            # Match IcM Incident IDs - look for fields containing "icm", "incident", and "id"/"ids" but NOT "count"
            if 'icm' in field_name_lower and 'incident' in field_name_lower and ('id' in field_name_lower or 'ids' in field_name_lower) and 'count' not in field_name_lower:
                if icm_incident_ids_ref is None:  # Take the first match
                    icm_incident_ids_ref = field_name
                    logger.debug(f"Found IcM Incident IDs field: {field_name} = {field_value}")
            # Match IcM Incident Count - look for fields containing "icm", "incident", and "count"
            elif 'icm' in field_name_lower and 'incident' in field_name_lower and 'count' in field_name_lower:
                if icm_incident_count_ref is None:  # Take the first match
                    icm_incident_count_ref = field_name
                    logger.debug(f"Found IcM Incident Count field: {field_name} = {field_value}")
        
        if not icm_incident_ids_ref or not icm_incident_count_ref:
            # Log all custom fields for debugging
            custom_fields = {k: v for k, v in fields.items() if 'custom' in k.lower() or 'icm' in k.lower()}
            logger.error(f"Could not find required custom fields in work item {work_item_id}")
            logger.debug(f"Available custom/ICM fields: {list(custom_fields.keys())}")
            return False
        
        # Get current values
        current_ids = fields.get(icm_incident_ids_ref, "")
        current_count = fields.get(icm_incident_count_ref, 0)
        
        # Parse current IDs (might be comma-separated string)
        if isinstance(current_ids, str):
            existing_ids = [id.strip() for id in current_ids.split(',') if id.strip()]
        else:
            existing_ids = []
        
        # Add new incident ID if not already present
        if incident_id not in existing_ids:
            existing_ids.append(incident_id)
        
        # Build new IDs string
        new_ids = ','.join(existing_ids)
        
        # Increment count
        try:
            new_count = int(current_count) + 1
        except (ValueError, TypeError):
            new_count = 1
        
        # Build patch operations
        patch_operations = [
            {
                "op": "replace",
                "path": f"/fields/{icm_incident_ids_ref}",
                "value": new_ids
            },
            {
                "op": "replace",
                "path": f"/fields/{icm_incident_count_ref}",
                "value": new_count
            }
        ]
        
        url = f"{self.base_url}/_apis/wit/workitems/{work_item_id}?api-version={self.api_version}"
        headers = {
            'Content-Type': 'application/json-patch+json'
        }
        
        try:
            response = self._make_request('PATCH', url, data=patch_operations, headers=headers)
            logger.info(f"Successfully updated work item #{work_item_id} with incident {incident_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating work item {work_item_id}: {e}")
            return False
    
    def create_work_item(self, title: str, description: str, work_item_type: str = "Task") -> Optional[int]:
        """
        Create a new work item for preventative action.
        
        Args:
            title: Work item title
            description: Work item description
            work_item_type: Type of work item (default: "Task")
            
        Returns:
            Created work item ID, or None if creation failed
        """
        # Use JSON Patch format to create work item
        patch_operations = [
            {
                "op": "add",
                "path": "/fields/System.Title",
                "value": title
            },
            {
                "op": "add",
                "path": "/fields/System.Description",
                "value": description
            }
        ]
        
        url = f"{self.base_url}/_apis/wit/workitems/${work_item_type}?api-version={self.api_version}"
        # Note: $ is literal in the URL path for work item type
        headers = {
            'Content-Type': 'application/json-patch+json'
        }
        
        try:
            response = self._make_request('POST', url, data=patch_operations, headers=headers)
            result = response.json()
            work_item_id = result.get('id')
            logger.info(f"Successfully created work item #{work_item_id}")
            return work_item_id
        except Exception as e:
            logger.error(f"Error creating work item: {e}")
            return None
    
    def close_work_item(self, work_item_id: int, reason: str) -> bool:
        """
        Close a work item by setting its state to Closed and appending a reason to the description.

        Args:
            work_item_id: Work item ID to close
            reason: Explanation of why the item was closed (written into Description)

        Returns:
            True if successful, False otherwise
        """
        patch_operations = [
            {
                "op": "add",
                "path": "/fields/System.State",
                "value": "Closed"
            },
            {
                "op": "add",
                "path": "/fields/System.Description",
                "value": reason
            }
        ]

        url = f"{self.base_url}/_apis/wit/workitems/{work_item_id}?api-version={self.api_version}"
        headers = {
            'Content-Type': 'application/json-patch+json'
        }

        try:
            response = self._make_request('PATCH', url, data=patch_operations, headers=headers)
            logger.info(f"Successfully closed work item #{work_item_id}")
            return True
        except Exception as e:
            logger.error(f"Error closing work item {work_item_id}: {e}")
            return False

    def get_work_item_url(self, work_item_id: int) -> str:
        """Get the web URL for a work item."""
        return f"{self.base_url}/_workitems/edit/{work_item_id}"
