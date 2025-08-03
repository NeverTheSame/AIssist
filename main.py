import sys
import subprocess
import os
import json
from azure.kusto.data.exceptions import KustoNetworkError
from datetime import datetime
import traceback

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
                    return selected_prompt
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <incident_number> [--prompt-type TYPE] [--azure] [--zai]")
        sys.exit(1)

    incident_number = sys.argv[1]
    prompt_type = None  # Changed from "default" to None
    use_azure = False
    use_zai = False
    debug_api = False

    # Parse additional args
    for i, arg in enumerate(sys.argv):
        if arg == "--prompt-type" and i + 1 < len(sys.argv):
            prompt_type = sys.argv[i + 1]
        if arg == "--azure":
            use_azure = True
        if arg == "--zai":
            use_zai = True
        if arg == "--debug_api":
            debug_api = True

    # If no prompt type specified, show interactive menu
    if prompt_type is None:
        prompt_type = show_prompt_menu()

    # Validate prompt_type before proceeding
    try:
        with open("prompts.json", "r", encoding="utf-8") as f:
            prompts = json.load(f)
        if prompt_type not in prompts:
            available = list(prompts.keys())
            error_msg = f"Prompt type '{prompt_type}' not found. Available types: {available}"
            print(error_msg)
            # Ensure logs directory exists
            os.makedirs("logs", exist_ok=True)
            with open("logs/processor.log", "a", encoding="utf-8") as logf:
                logf.write(f"{datetime.now().isoformat()} - {error_msg}\n")
                logf.write(traceback.format_exc())
                logf.write("\n")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading prompts.json: {e}")
        os.makedirs("logs", exist_ok=True)
        with open("logs/processor.log", "a", encoding="utf-8") as logf:
            logf.write(f"{datetime.now().isoformat()} - Error reading prompts.json: {e}\n")
            logf.write(traceback.format_exc())
            logf.write("\n")
        sys.exit(1)

    # Step 1: Fetch from Kusto (creates icms/{incident_number}.csv)
    print("Fetching data from Kusto...")
    fetch_proc = subprocess.run([
        sys.executable, "kusto_fetcher.py", str(incident_number), "--output-dir", "icms"
    ], capture_output=True, text=True)
    if fetch_proc.returncode != 0:
        # Log all output to a debug file
        with open("fetcher_debug.log", "w") as log_file:
            log_file.write("STDOUT:\n" + fetch_proc.stdout + "\n\n")
            log_file.write("STDERR:\n" + fetch_proc.stderr + "\n")
        # Check for VPN/Kusto network error in stderr or stdout
        if "Could not connect to Azure Kusto" in fetch_proc.stdout or "Could not connect to Azure Kusto" in fetch_proc.stderr:
            print("[main] Data fetch failed due to network error. Please ensure your VPN connection (e.g., 'MSFT-AzVPN-Manual') is active and try again.")
            print("[main] Full fetcher logs are available in fetcher_debug.log")
            sys.exit(1)
        print("Kusto fetch step failed. See fetcher_debug.log for details.")
        sys.exit(fetch_proc.returncode)

    # Check if the CSV file is empty or only contains the placeholder line
    csv_path = os.path.join("icms", f"{incident_number}.csv")
    is_empty = False
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            if len(lines) == 0 or (len(lines) == 1 and lines[0] == "--- Discussions ---"):
                is_empty = True
    except Exception as e:
        is_empty = True
        log_msg = f"{datetime.now().isoformat()} - Error reading CSV file {csv_path}: {e}\n"
        os.makedirs("logs", exist_ok=True)
        with open("logs/fetcher.log", "a", encoding="utf-8") as logf:
            logf.write(log_msg)
        print(f"[main] Error reading CSV file: {e}")
        sys.exit(1)

    if is_empty:
        log_msg = f"{datetime.now().isoformat()} - No data fetched for incident {incident_number}. CSV file is empty.\n"
        os.makedirs("logs", exist_ok=True)
        with open("logs/fetcher.log", "a", encoding="utf-8") as logf:
            logf.write(log_msg)
        print(f"[main] No data was fetched for incident {incident_number}. The CSV file is empty. Aborting.")
        sys.exit(1)

    # Step 2: Process CSV to JSON (creates processed_incidents/{incident_number}.json)
    print("Processing CSV to JSON...")
    csv_path = os.path.join("icms", f"{incident_number}.csv")
    subprocess.run([
        sys.executable, "incident_dumper.py", csv_path
    ], check=True)

    # Step 3: Process JSON with AI (outputs AI summary)
    print("Processing JSON with AI...")
    json_path = os.path.join("processed_incidents", f"{incident_number}.json")
    ai_cmd = [
        sys.executable, "incident_processor.py", json_path, "--prompt-type", prompt_type
    ]
    if use_azure:
        ai_cmd.append("--azure")
    if use_zai:
        ai_cmd.append("--zai")
    if debug_api:
        ai_cmd.append("--debug_api")
    subprocess.run(ai_cmd, check=True)

if __name__ == "__main__":
    main()
