import sys
import subprocess
import os
import json
from azure.kusto.data.exceptions import KustoNetworkError
from datetime import datetime
import traceback
import argparse
import logging

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
        
        # Filter to only show molecular prompt types
        prompt_types = [pt for pt in prompts.keys() if pt.endswith('_molecular')]
        
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

def fetch_incident_data(incident_number):
    """Fetch data for a single incident from Kusto"""
    logger.info(f"Starting Kusto data fetch for incident {incident_number}")
    print(f"Fetching data for incident {incident_number} from Kusto...")
    
    fetch_proc = subprocess.run([
        sys.executable, "kusto_fetcher.py", str(incident_number), "--output-dir", "icms"
    ], capture_output=True, text=True)
    
    if fetch_proc.returncode != 0:
        # Log all output to a debug file
        debug_log_path = f"logs/fetcher_debug_{incident_number}.log"
        with open(debug_log_path, "w") as log_file:
            log_file.write("STDOUT:\n" + fetch_proc.stdout + "\n\n")
            log_file.write("STDERR:\n" + fetch_proc.stderr + "\n")
        
        logger.error(f"Kusto fetch failed for incident {incident_number}. Return code: {fetch_proc.returncode}")
        logger.error(f"STDOUT: {fetch_proc.stdout}")
        logger.error(f"STDERR: {fetch_proc.stderr}")
        
        # Check for VPN/Kusto network error in stderr or stdout
        if "Could not connect to Azure Kusto" in fetch_proc.stdout or "Could not connect to Azure Kusto" in fetch_proc.stderr:
            logger.error(f"Network error detected for incident {incident_number}. VPN connection may be required.")
            print(f"[main] Data fetch failed for incident {incident_number} due to network error. Please ensure your VPN connection (e.g., 'MSFT-AzVPN-Manual') is active and try again.")
            print(f"[main] Full fetcher logs are available in {debug_log_path}")
            return False
        
        print(f"Kusto fetch step failed for incident {incident_number}. See {debug_log_path} for details.")
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
        logger.error(f"No data fetched for incident {incident_number}. CSV file is empty.")
        print(f"[main] No data was fetched for incident {incident_number}. The CSV file is empty.")
        return False
    
    logger.info(f"Successfully fetched data for incident {incident_number}. CSV file created at {csv_path}")
    print(f"✅ Created: {csv_path}")
    return True

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

def main():
    logger.info("=" * 80)
    logger.info("Starting Summarizer application")
    logger.info("=" * 80)
    
    # Preprocess command line arguments to handle -4 and -5 shorthand
    processed_args = []
    for arg in sys.argv[1:]:
        if arg == "-5":
            processed_args.append("--azure-5")
        elif arg == "-4":
            processed_args.append("--azure")
        else:
            processed_args.append(arg)
    
    parser = argparse.ArgumentParser(description="Process multiple ICM incidents and provide unified summarization")
    parser.add_argument("incident_numbers", nargs="+", help="One or more incident numbers to process")
    parser.add_argument("--prompt-type", help="Type of prompt to use for summarization")
    parser.add_argument("--azure", action="store_true", help="Use Azure OpenAI (GPT-4)")
    parser.add_argument("--azure-5", action="store_true", help="Use Azure OpenAI 5 (GPT-5)")
    parser.add_argument("--zai", action="store_true", help="Use ZAI")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable API debugging")
    parser.add_argument("--troubleshooting-plan", action="store_true", help="Generate troubleshooting plan mode - first incident is primary, others are historical references")
    parser.add_argument("--articles-path", help="Path to directory containing troubleshooting articles (for article search mode)")
    parser.add_argument("--vector-db-path", help="Path to vector database file (for article search mode)")
    
    args = parser.parse_args(processed_args)
    
    incident_numbers = args.incident_numbers
    prompt_type = args.prompt_type
    use_azure = args.azure
    use_azure_5 = args.azure_5
    use_zai = args.zai
    debug_api = args.debug
    troubleshooting_plan_mode = args.troubleshooting_plan
    articles_path = args.articles_path
    vector_db_path = args.vector_db_path
    
    # Set default behavior: if no specific model is specified, use Azure OpenAI 5
    if not use_azure and not use_azure_5 and not use_zai:
        use_azure_5 = True

    logger.info(f"Command line arguments:")
    logger.info(f"  Incident numbers: {incident_numbers}")
    logger.info(f"  Prompt type: {prompt_type}")
    logger.info(f"  Use Azure: {use_azure}")
    logger.info(f"  Use Azure 5: {use_azure_5}")
    logger.info(f"  Use ZAI: {use_zai}")
    logger.info(f"  Debug API: {debug_api}")
    logger.info(f"  Troubleshooting plan mode: {troubleshooting_plan_mode}")
    logger.info(f"  Articles path: {articles_path}")
    logger.info(f"  Vector DB path: {vector_db_path}")

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

    # Step 1: Fetch data for all incidents
    logger.info("=" * 50)
    logger.info("STEP 1: Fetching data from Kusto")
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
    elif len(successful_incidents) > 1:
        logger.info("Combining multiple incidents for unified processing...")
        combined_json_path = combine_incident_data(successful_incidents)
        # Step 4: Process combined JSON with AI
        print("Processing combined incidents with AI...")
        ai_cmd = [
            sys.executable, "processor.py", combined_json_path, "--prompt-type", prompt_type, "--multi-incident"
        ]
    else:
        # Single incident - process normally
        json_path = os.path.join("processed_incidents", f"{successful_incidents[0]}.json")
        logger.info(f"Processing single incident: {json_path}")
        print("Processing single incident with AI...")
        ai_cmd = [
            sys.executable, "processor.py", json_path, "--prompt-type", prompt_type
        ]
    
    # Add article search parameters if provided
    if articles_path:
        ai_cmd.extend(["--articles-path", articles_path])
    if vector_db_path:
        ai_cmd.extend(["--vector-db-path", vector_db_path])
    
    # Add model-specific arguments
    if use_azure:
        ai_cmd.append("--azure")
    if use_azure_5:
        ai_cmd.append("--azure-5")
    if use_zai:
        ai_cmd.append("--zai")
    if debug_api:
        ai_cmd.append("--debug")
    
    logger.info(f"Executing AI processing command: {' '.join(ai_cmd)}")
    
    try:
        # Don't capture output so we can see it in real-time
        result = subprocess.run(ai_cmd, check=True)
        logger.info("AI processing completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error("AI processing failed")
        logger.error(f"Return code: {e.returncode}")
        raise
    
    logger.info("=" * 80)
    logger.info("Summarizer application completed successfully")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
