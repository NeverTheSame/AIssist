import sys
import subprocess
import os
import json
from azure.kusto.data.exceptions import KustoNetworkError
from datetime import datetime
import traceback
import argparse
import logging
from timing_utils import start_timing, end_timing, time_operation, time_context, print_timing_summary, save_timing_report, reset_timing_data

# Configure centralized logging
def setup_logging():
    """Setup centralized logging for the entire application"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create logger for this module
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler - detailed logging
    file_handler = logging.FileHandler('logs/summarizer.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler - only warnings and errors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

# Setup logging at module level
logger = setup_logging()

def show_prompt_menu():
    """Display available prompts and get user selection"""
    try:
        with open("prompts.json", "r", encoding="utf-8") as f:
            prompts = json.load(f)
        
        # Filter to show molecular prompt types and create_prompt_for_logs_analyze
        prompt_types = [pt for pt in prompts.keys() if pt.endswith('_molecular') or pt == 'create_prompt_for_logs_analyze']
        
        if not prompt_types:
            print("No molecular prompt types found in prompts.json")
            sys.exit(1)
        
        print("\nAvailable molecular prompt types:")
        print("=" * 40)
        for i, prompt_type in enumerate(prompt_types, 1):
            print(f"{i:2d}. {prompt_type}")
        print("=" * 40)
        
        while True:
            try:
                choice = input("Select a prompt type (enter number): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(prompt_types):
                    selected_prompt = prompt_types[choice_num - 1]
                    print(f"Selected: {selected_prompt}")
                    
                    # Automatically set vector database path for article search mode
                    if selected_prompt == 'article_search_molecular':
                        from config import config
                        default_vector_db_path = config.default_vector_db_path
                        if default_vector_db_path:
                            print(f"🔍 Article search mode detected - automatically using vector database: {default_vector_db_path}")
                            return selected_prompt, default_vector_db_path
                        else:
                            print("⚠️ Article search mode detected but DEFAULT_VECTOR_DB_PATH not set in .env file")
                            return selected_prompt, None
                    
                    return selected_prompt, None
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(prompt_types)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
                
    except Exception as e:
        print(f"Error reading prompts.json: {e}")
        sys.exit(1)

@time_operation("fetch_incident_data", "fetch")
def fetch_incident_data(incident_number):
    """Fetch data for a single incident from database"""
    logger.info(f"Starting database data fetch for incident {incident_number}")
    print(f"Fetching data for incident {incident_number} from database...")
    
    with time_context("kusto_subprocess", "fetch", {"incident_number": incident_number}):
        fetch_proc = subprocess.run([
            sys.executable, "kusto_fetcher.py", str(incident_number), "--output-dir", "icms"
        ], capture_output=True, text=True)
    
    if fetch_proc.returncode != 0:
        # Log all output to a debug file
        debug_log_path = f"logs/fetcher_debug_{incident_number}.log"
        with open(debug_log_path, "w") as log_file:
            log_file.write("STDOUT:\n" + fetch_proc.stdout + "\n\n")
            log_file.write("STDERR:\n" + fetch_proc.stderr + "\n")
        
        logger.error(f"Database fetch failed for incident {incident_number}. Return code: {fetch_proc.returncode}")
        logger.error(f"STDOUT: {fetch_proc.stdout}")
        logger.error(f"STDERR: {fetch_proc.stderr}")
        
        # Check for VPN/database network error in stderr or stdout
        network_error_indicators = [
            "Could not connect to database",
            "KustoNetworkError",
            "Failed to process network request",
            "Network error"
        ]
        
        is_network_error = any(indicator in fetch_proc.stdout or indicator in fetch_proc.stderr 
                              for indicator in network_error_indicators)
        
        if is_network_error:
            logger.error(f"Network error detected for incident {incident_number}. VPN connection may be required.")
            print(f"❌ Network Error: Could not connect to database for incident {incident_number}")
            print(f"🔧 Solution: Please ensure your VPN connection is active and try again.")
            print(f"📋 Full error details are available in {debug_log_path}")
            return False
        
        print(f"Database fetch step failed for incident {incident_number}. See {debug_log_path} for details.")
        return False

    # Check if the CSV file is empty or only contains the placeholder line
    # Try the new incident-specific folder structure first, then fall back to flat structure
    csv_path = os.path.join("icms", str(incident_number), f"{incident_number}.csv")
    if not os.path.exists(csv_path):
        # Fall back to flat structure for backward compatibility
        csv_path = os.path.join("icms", f"{incident_number}.csv")
    logger.info(f"Checking CSV file: {csv_path}")
    
    is_empty = False
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            if len(lines) == 0 or (len(lines) == 1 and lines[0] == "--- Discussions ---"):
                is_empty = True
                logger.warning(f"CSV file is empty or contains only placeholder for incident {incident_number}")
    except Exception as e:
        is_empty = True
        logger.error(f"Error reading CSV file {csv_path}: {e}")
        print(f"[main] Error reading CSV file for incident {incident_number}: {e}")
        return False

    if is_empty:
        logger.warning(f"No data fetched for incident {incident_number}. CSV file is empty.")
        
        # Check if this might be due to a network error by looking at recent logs
        try:
            with open("logs/fetcher.log", "r", encoding="utf-8") as log_file:
                recent_logs = log_file.read()
                if any(indicator in recent_logs for indicator in ["KustoNetworkError", "Failed to process network request", "Network error"]):
                    print(f"❌ Network Error: No data fetched for incident {incident_number} due to network connectivity issues.")
                    print(f"🔧 Solution: Please ensure your VPN connection is active and try again.")
                    print(f"📋 Check logs/fetcher.log for detailed error information.")
                    return False
        except FileNotFoundError:
            pass  # Log file doesn't exist, continue with normal empty file handling
        
        print(f"[main] No data was fetched for incident {incident_number}. The CSV file is empty.")
        
        # No manual fallback available - return failure
        print(f"[main] No data available for incident {incident_number}. Please check the incident ID or try again later.")
        return False
    
    logger.info(f"Successfully fetched data for incident {incident_number}. CSV file created at {csv_path}")
    print(f"✅ Created: {csv_path}")
    return True

@time_operation("process_incident_to_json", "process")
def process_incident_to_json(incident_number):
    """Process CSV to JSON for a single incident"""
    logger.info(f"Starting CSV to JSON conversion for incident {incident_number}")
    print(f"Processing CSV to JSON for incident {incident_number}...")
    
    # Try the new incident-specific folder structure first, then fall back to flat structure
    csv_path = os.path.join("icms", str(incident_number), f"{incident_number}.csv")
    if not os.path.exists(csv_path):
        # Fall back to flat structure for backward compatibility
        csv_path = os.path.join("icms", f"{incident_number}.csv")
    logger.info(f"Processing CSV file: {csv_path}")
    
    try:
        with time_context("transformer_subprocess", "process", {"incident_number": incident_number, "csv_path": csv_path}):
            result = subprocess.run([
                sys.executable, "transformer.py", csv_path
            ], capture_output=True, text=True, check=True)
        
        logger.info(f"Successfully processed CSV to JSON for incident {incident_number}")
        logger.info(f"STDOUT: {result.stdout}")
        print(f"✅ Created: processed_incidents/{incident_number}.json")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"CSV to JSON conversion failed for incident {incident_number}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise

@time_operation("combine_incident_data", "process")
def combine_incident_data(incident_numbers):
    """Combine data from multiple incidents into a single JSON file"""
    logger.info(f"Starting to combine data from {len(incident_numbers)} incidents")
    print("Combining data from multiple incidents...")
    
    combined_data = {
        "incidents": [],
        "total_incidents": len(incident_numbers),
        "combined_timestamp": datetime.now().isoformat()
    }
    
    for incident_number in incident_numbers:
        json_path = os.path.join("processed_incidents", f"{incident_number}.json")
        logger.info(f"Processing incident {incident_number} from {json_path}")
        
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    incident_data = json.load(f)
                    incident_data["incident_number"] = incident_number
                    combined_data["incidents"].append(incident_data)
                logger.info(f"Successfully loaded incident {incident_number}")
            except Exception as e:
                logger.error(f"Error reading JSON for incident {incident_number}: {e}")
                print(f"Error reading JSON for incident {incident_number}: {e}")
                continue
        else:
            logger.warning(f"JSON file not found for incident {incident_number}: {json_path}")
            print(f"Warning: JSON file not found for incident {incident_number}")
    
    # Save combined data
    combined_path = os.path.join("processed_incidents", "combined_incidents.json")
    logger.info(f"Saving combined data to {combined_path}")
    
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Successfully combined {len(combined_data['incidents'])} incidents into {combined_path}")
    print(f"✅ Created: {combined_path}")
    return combined_path

@time_operation("create_troubleshooting_plan_data", "process")
def create_troubleshooting_plan_data(incident_numbers):
    """Create a special combined data structure for troubleshooting plan mode.
    The first incident is treated as the primary incident, others as historical references."""
    print("Creating troubleshooting plan data structure...")
    
    if len(incident_numbers) < 2:
        raise ValueError("Troubleshooting plan mode requires at least 2 incidents")
    
    primary_incident = incident_numbers[0]
    historical_incidents = incident_numbers[1:]
    
    combined_data = {
        "mode": "troubleshooting_plan",
        "primary_incident": primary_incident,
        "historical_incidents": historical_incidents,
        "incidents": [],
        "total_incidents": len(incident_numbers),
        "combined_timestamp": datetime.now().isoformat()
    }
    
    # Load primary incident first
    primary_json_path = os.path.join("processed_incidents", f"{primary_incident}.json")
    if os.path.exists(primary_json_path):
        try:
            with open(primary_json_path, "r", encoding="utf-8") as f:
                primary_data = json.load(f)
                primary_data["incident_number"] = primary_incident
                primary_data["role"] = "primary"
                combined_data["incidents"].append(primary_data)
        except Exception as e:
            print(f"Error reading JSON for primary incident {primary_incident}: {e}")
            raise
    else:
        print(f"Error: Primary incident JSON file not found for {primary_incident}")
        raise FileNotFoundError(f"Primary incident file not found: {primary_json_path}")
    
    # Load historical incidents
    for incident_number in historical_incidents:
        json_path = os.path.join("processed_incidents", f"{incident_number}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    incident_data = json.load(f)
                    incident_data["incident_number"] = incident_number
                    incident_data["role"] = "historical"
                    combined_data["incidents"].append(incident_data)
            except Exception as e:
                print(f"Error reading JSON for historical incident {incident_number}: {e}")
                continue
        else:
            print(f"Warning: JSON file not found for historical incident {incident_number}")
    
    # Save combined data
    combined_path = os.path.join("processed_incidents", f"troubleshooting_plan_{primary_incident}_with_{'_'.join(historical_incidents)}.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created: {combined_path}")
    return combined_path

def _process_single_incident(processor, incident_data, prompts, prompt_type, debug_api, incident_id):
    """Helper function to process a single incident."""
    # Extract conversation data (needed for all modes)
    conversation = incident_data.get('conversation', [])
    
    # Defensive check to ensure conversation is always a list
    if not isinstance(conversation, list):
        logger.warning(f"Conversation data is not a list, got {type(conversation)}. Converting to empty list.")
        conversation = []
    
    # Check if this is article search mode
    if prompt_type == 'article_search_molecular':
        # Use article search processing
        summary_result = processor.process_article_search(
            incident_data=incident_data,
            prompts=prompts,
            prompt_type=prompt_type,
            debug_api=debug_api
        )
        
        # Display incident summary first
        print("\n" + "="*80)
        print("INCIDENT SUMMARY")
        print("="*80)
        summary = incident_data.get('summary', 'No summary available')
        if summary and len(summary) > 0:
            # Truncate summary if it's too long for display
            display_summary = summary[:800] + "..." if len(summary) > 800 else summary
            print(display_summary)
        else:
            print("No incident summary available")
        print("="*80)
        
        # Display article search results
        if summary_result and 'analysis' in summary_result:
            print("\n" + "="*80)
            print("ARTICLE SEARCH RESULTS")
            print("="*80)
            print(summary_result['analysis'])
            print("="*80)
    elif prompt_type == 'logs_analyzer':
        # Use logs analyzer processing
        summary_result = processor.process_logs_analyzer(
            incident_data=incident_data,
            prompts=prompts,
            prompt_type=prompt_type,
            debug_api=debug_api,
            incident_id=incident_id
        )
        
        # Display incident summary first
        print("\n" + "="*80)
        print("INCIDENT SUMMARY")
        print("="*80)
        summary = incident_data.get('summary', 'No summary available')
        if summary and len(summary) > 0:
            # Truncate summary if it's too long for display
            display_summary = summary[:800] + "..." if len(summary) > 800 else summary
            print(display_summary)
        else:
            print("No incident summary available")
        print("="*80)
        
        # Display logs analysis results
        if summary_result and 'analysis' in summary_result:
            print("\n" + "="*80)
            print("LOGS ANALYSIS RESULTS")
            print("="*80)
            # Extract the actual analysis content from the result
            analysis_content = summary_result['analysis']
            if isinstance(analysis_content, dict) and 'summary' in analysis_content:
                print(analysis_content['summary'])
            else:
                print(analysis_content)
            print("="*80)
        
        # Display log analysis details
        if summary_result and 'log_analysis' in summary_result:
            log_analysis = summary_result['log_analysis']
            if 'analyzer_path' in log_analysis:
                print(f"\n📁 Analyzed logs from: {log_analysis['analyzer_path']}")
            if 'log_files' in log_analysis:
                print(f"📄 Analyzed {len(log_analysis['log_files'])} log files")
            if 'mde_logs' in log_analysis:
                print(f"🔍 Analyzed {len(log_analysis['mde_logs'])} security-specific log files")
    else:
        # Generate summary for other modes
        summary = incident_data.get('summary', None)
        formatted_content = processor.format_conversation_with_ai_summary(conversation, summary=summary)
        
        summary_result = processor.generate_summary(
            [{'type': 'text', 'content': formatted_content}],
            prompts['system_prompt'],
            prompts['user_prompt'],
            prompt_type=prompt_type,
            debug_api=debug_api,
            incident_data=incident_data
        )
    
    # Save results
    operation_time = datetime.now().isoformat()
    model_name = processor.deployment_name if hasattr(processor, 'deployment_name') else "unknown"
    
    try:
        processor.save_to_json(
            conversation,
            incident_id,
            ai_summary=summary_result,
            prompt_type=prompt_type,
            operation_time=operation_time,
            model_name=model_name
        )
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        # Continue processing even if save fails
    
    # Store memory about this incident
    if processor.memory_manager:
        try:
            processor.store_incident_memory(incident_id, incident_data, summary_result)
        except Exception as e:
            logger.error(f"Failed to store memory for incident {incident_id}: {e}")

def main():
    # Parse arguments first to check if timing is enabled
    processed_args = []
    for arg in sys.argv[1:]:
        if arg == "-5":
            processed_args.append("--azure-5")
        elif arg == "-4":
            processed_args.append("--azure")
        else:
            processed_args.append(arg)
    
    parser = argparse.ArgumentParser(description="Process multiple support incidents and provide unified summarization")
    parser.add_argument("incident_numbers", nargs="+", help="One or more incident numbers to process")
    parser.add_argument("--prompt-type", help="Type of prompt to use for summarization")
    # Always use AI Service (GPT-5) - no model selection needed
    parser.add_argument("--debug", "-d", action="store_true", help="Enable API debugging")
    parser.add_argument("--troubleshooting-plan", action="store_true", help="Generate troubleshooting plan mode - first incident is primary, others are historical references")
    parser.add_argument("--articles-embeddings", help="Path to article embeddings file (for article search mode)")
    parser.add_argument("--vector-db-path", help="Path to vector database file (for memory management)")
    parser.add_argument("--timing", action="store_true", help="Enable detailed timing analysis and reporting")
    parser.add_argument("--enable-team-analysis", action="store_true", help="Enable team analysis and team matching features")
    
    args = parser.parse_args(processed_args)
    enable_timing = args.timing
    
    # Reset and start timing for the entire workflow only if enabled
    if enable_timing:
        reset_timing_data()
        start_timing()
    
    logger.info("=" * 80)
    logger.info("Starting Summarizer application")
    logger.info("=" * 80)
    
    # Extract arguments
    incident_numbers = args.incident_numbers
    prompt_type = args.prompt_type
    # Always use AI Service (GPT-5)
    use_ai_service_default = True
    debug_api = args.debug
    troubleshooting_plan_mode = args.troubleshooting_plan
    articles_embeddings = args.articles_embeddings
    vector_db_path = args.vector_db_path
    enable_team_analysis = args.enable_team_analysis
    
    # Set default values from config if not provided
    if not articles_embeddings:
        from config import config
        articles_embeddings = config.default_vector_db_path

    logger.info(f"Command line arguments:")
    logger.info(f"  Incident numbers: {incident_numbers}")
    logger.info(f"  Prompt type: {prompt_type}")
    logger.info(f"  Use AI Service (GPT-5): {use_ai_service_default}")
    logger.info(f"  Debug API: {debug_api}")
    logger.info(f"  Troubleshooting plan mode: {troubleshooting_plan_mode}")
    logger.info(f"  Articles embeddings: {articles_embeddings}")
    logger.info(f"  Vector DB path: {vector_db_path}")
    logger.info(f"  Enable team analysis: {enable_team_analysis}")

    print(f"Processing {len(incident_numbers)} incident(s): {', '.join(incident_numbers)}")

    # Handle troubleshooting plan mode
    if troubleshooting_plan_mode:
        logger.info("Troubleshooting plan mode detected")
        
        if len(incident_numbers) < 2:
            error_msg = "Troubleshooting plan mode requires at least 2 incidents (1 primary + 1 historical reference)"
            logger.error(error_msg)
            print(f"Error: {error_msg}")
            sys.exit(1)
        
        primary_incident = incident_numbers[0]
        historical_incidents = incident_numbers[1:]
        
        logger.info(f"Troubleshooting plan configuration:")
        logger.info(f"  Primary incident: {primary_incident}")
        logger.info(f"  Historical reference incidents: {historical_incidents}")
        
        print(f"Troubleshooting plan mode:")
        print(f"  Primary incident: {primary_incident}")
        print(f"  Historical reference incidents: {', '.join(historical_incidents)}")
        
        # Set the prompt type for troubleshooting plan
        prompt_type = "troubleshooting_plan_molecular"
        logger.info(f"Using prompt type: {prompt_type}")
        print(f"Using prompt type: {prompt_type}")
    else:
        # If no prompt type specified, show interactive menu
        if prompt_type is None:
            logger.info("No prompt type specified, showing interactive menu")
            prompt_type, auto_vector_db_path = show_prompt_menu()
            # Use the automatically detected vector database path if available
            if auto_vector_db_path and not vector_db_path:
                vector_db_path = auto_vector_db_path
                logger.info(f"Auto-detected vector database path: {vector_db_path}")
        else:
            logger.info(f"Using specified prompt type: {prompt_type}")

    # Validate prompt_type before proceeding
    logger.info("Validating prompt type...")
    try:
        with open("prompts.json", "r", encoding="utf-8") as f:
            prompts = json.load(f)
        if prompt_type not in prompts:
            available = list(prompts.keys())
            error_msg = f"Prompt type '{prompt_type}' not found. Available types: {available}"
            logger.error(error_msg)
            print(error_msg)
            sys.exit(1)
        logger.info(f"Prompt type '{prompt_type}' validated successfully")
    except Exception as e:
        logger.error(f"Error reading prompts.json: {e}")
        print(f"Error reading prompts.json: {e}")
        sys.exit(1)

    # Auto-detect vector database path for article search mode
    if prompt_type == 'article_search_molecular' and not vector_db_path:
        from config import config
        default_vector_db_path = config.default_vector_db_path
        if default_vector_db_path:
            vector_db_path = default_vector_db_path
            logger.info(f"🔍 Article search mode detected - automatically using vector database: {vector_db_path}")
            print(f"🔍 Article search mode detected - automatically using vector database: {vector_db_path}")
        else:
            logger.warning("⚠️ Article search mode detected but DEFAULT_VECTOR_DB_PATH not set in .env file")
            print("⚠️ Article search mode detected but DEFAULT_VECTOR_DB_PATH not set in .env file")

    # Step 1: Fetch data for all incidents from database
    logger.info("=" * 50)
    logger.info("STEP 1: Fetching data from database")
    logger.info("=" * 50)
    
    successful_incidents = []
    for incident_number in incident_numbers:
        if fetch_incident_data(incident_number):
            successful_incidents.append(incident_number)
        else:
            logger.warning(f"Skipping incident {incident_number} due to fetch failure")
            print(f"Skipping incident {incident_number} due to fetch failure")
    
    if not successful_incidents:
        logger.error("No incidents were successfully fetched. Exiting.")
        print("No incidents were successfully fetched. Exiting.")
        sys.exit(1)
    
    logger.info(f"Successfully fetched data for {len(successful_incidents)} incident(s): {successful_incidents}")
    print(f"Successfully fetched data for {len(successful_incidents)} incident(s): {', '.join(successful_incidents)}")

    # Step 2: Process CSV to JSON for all successful incidents
    logger.info("=" * 50)
    logger.info("STEP 2: Converting CSV to JSON")
    logger.info("=" * 50)
    
    for incident_number in successful_incidents:
        try:
            process_incident_to_json(incident_number)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error processing incident {incident_number} to JSON: {e}")
            print(f"Error processing incident {incident_number} to JSON: {e}")
            successful_incidents.remove(incident_number)

    if not successful_incidents:
        logger.error("No incidents were successfully processed. Exiting.")
        print("No incidents were successfully processed. Exiting.")
        sys.exit(1)

    # Step 3: Combine data from all incidents
    logger.info("=" * 50)
    logger.info("STEP 3: Processing with AI")
    logger.info("=" * 50)
    
    if troubleshooting_plan_mode:
        # For troubleshooting plan mode, we need to create a special combined file
        # that clearly identifies the primary incident vs historical incidents
        logger.info("Creating troubleshooting plan data structure...")
        combined_json_path = create_troubleshooting_plan_data(successful_incidents)
        print("Processing incidents for troubleshooting plan generation...")
        ai_cmd = [
            sys.executable, "processor.py", combined_json_path, "--prompt-type", prompt_type, "--multi-incident"
        ]
        if not enable_team_analysis:
            ai_cmd.append("--no-team-analysis")
    elif len(successful_incidents) > 1:
        logger.info("Combining multiple incidents for unified processing...")
        combined_json_path = combine_incident_data(successful_incidents)
        # Step 4: Process combined JSON with AI
        print("Processing combined incidents with AI...")
        ai_cmd = [
            sys.executable, "processor.py", combined_json_path, "--prompt-type", prompt_type, "--multi-incident"
        ]
        if not enable_team_analysis:
            ai_cmd.append("--no-team-analysis")
    else:
        # Single incident - process directly for better timing granularity
        json_path = os.path.join("processed_incidents", f"{successful_incidents[0]}.json")
        logger.info(f"Processing single incident: {json_path}")
        print("Processing single incident with AI...")
        
        # Import and call processor directly for single incidents to get granular timing
        try:
            from processor import IncidentProcessor, load_prompts
            
            # Load the incident data
            with open(json_path, 'r', encoding='utf-8') as f:
                incident_data = json.load(f)
            
            # Load prompts
            prompts = load_prompts(prompt_type)
            
            # Initialize processor (always uses AI Service GPT-5)
            processor = IncidentProcessor(
                enable_memory=True,
                enable_team_analysis=enable_team_analysis,
                articles_path=articles_embeddings,
                vector_db_path=vector_db_path,
                enable_timing=enable_timing
            )
            
            # Process the incident
            if enable_timing:
                from timing_utils import time_context
                with time_context("ai_processing_detailed", "ai", {
                    "incident_count": 1,
                    "prompt_type": prompt_type,
                    "troubleshooting_plan_mode": False
                }):
                    _process_single_incident(processor, incident_data, prompts, prompt_type, debug_api, successful_incidents[0])
            else:
                _process_single_incident(processor, incident_data, prompts, prompt_type, debug_api, successful_incidents[0])
            
            logger.info("AI processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error in direct AI processing: {e}")
            # Fallback to subprocess approach
            ai_cmd = [
                sys.executable, "processor.py", json_path, "--prompt-type", prompt_type
            ]
            
            # Add debug argument if needed (always uses AI Service GPT-5)
            if debug_api:
                ai_cmd.append("--debug")
            
            # Add article search parameters if provided
            if articles_embeddings:
                ai_cmd.extend(["--articles-embeddings", articles_embeddings])
            if vector_db_path:
                ai_cmd.extend(["--vector-db-path", vector_db_path])
            if not enable_team_analysis:
                ai_cmd.append("--no-team-analysis")
            
            logger.info(f"Falling back to subprocess: {' '.join(ai_cmd)}")
            
            try:
                result = subprocess.run(ai_cmd, check=True)
                logger.info("AI processing completed successfully")
            except subprocess.CalledProcessError as e:
                logger.error("AI processing failed")
                logger.error(f"Return code: {e.returncode}")
                raise
    
    # End timing and print summary only if timing is enabled
    if enable_timing:
        end_timing()
        
        logger.info("=" * 80)
        logger.info("Summarizer application completed successfully")
        logger.info("=" * 80)
        
        # Print timing summary
        print_timing_summary()
        
        # Save timing report
        save_timing_report()
    else:
        logger.info("=" * 80)
        logger.info("Summarizer application completed successfully")
        logger.info("=" * 80)

if __name__ == "__main__":
    main()
