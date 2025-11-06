"""
Streamlit Web Interface for Incident Summarizer

This application provides a web-based interface for processing and analyzing
incident data using the Summarizer application.
"""

import streamlit as st
import os
import json
import sys
import re
from pathlib import Path
from datetime import datetime
import subprocess
import traceback
from typing import List, Dict, Any, Optional, Tuple

# Set page config
st.set_page_config(
    page_title="Incident Summarizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = None
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'processed_incidents' not in st.session_state:
    st.session_state.processed_incidents = []


def load_prompt_types() -> Dict[str, str]:
    """Load available prompt types from prompts.json"""
    try:
        if os.path.exists('prompts.json'):
            with open('prompts.json', 'r', encoding='utf-8') as f:
                prompts = json.load(f)
            
            # Filter molecular prompt types and specific analysis prompts
            prompt_types = {
                pt: pt for pt in prompts.keys() 
                if pt.endswith('_molecular') or pt in [
                    'create_prompt_for_logs_analyze', 
                    'stage_duration_analysis', 
                    'resolution_delay_analysis'
                ]
            }
            return prompt_types
        else:
            st.error("""
            **prompts.json not found!**
            
            This file is required for the application to work. Please:
            1. Ensure prompts.json exists in your repository
            2. Or add it to your repository if it's currently in .gitignore
            3. For Streamlit Cloud: Add prompts.json to your repository before deploying
            
            Check STREAMLIT_CLOUD_SETUP.md for deployment instructions.
            """)
            return {}
    except json.JSONDecodeError as e:
        st.error(f"Error parsing prompts.json: {e}")
        return {}
    except Exception as e:
        st.error(f"Error loading prompts: {e}")
        return {}


def get_prompt_descriptions() -> Dict[str, str]:
    """Get human-readable descriptions for prompt types"""
    descriptions = {
        'escalation_molecular': '📈 Create escalation summaries with dynamic examples',
        'mitigation_molecular': '🛡️ Generate mitigation reports with contextual guidance',
        'troubleshooting_molecular': '🔧 Produce detailed troubleshooting guides',
        'article_search_molecular': '🔍 Find relevant troubleshooting articles',
        'troubleshooting_plan_molecular': '📋 Create comprehensive troubleshooting plans',
        'wait_time_molecular': '⏱️ Analyze incident wait times by team',
        'prev_act_molecular': '🔒 Recommend preventative actions',
        'weekly_insights_molecular': '📅 Generate weekly status updates',
        'kb_article_molecular': '📚 Create knowledge base articles and runbooks',
        'improvement_analysis_molecular': '💡 Analyze incidents for process improvement',
        'customer_pending_facilitation_molecular': '👤 Customer pending facilitation summaries',
        'dev_pending_facilitation_molecular': '👨‍💻 Developer pending facilitation summaries',
        'product_improvement_email_molecular': '✉️ Product improvement email generation',
        'runbook_creation_request_molecular': '📖 Runbook creation requests',
        'create_prompt_for_logs_analyze': '📝 Create tailored investigation prompts for log analysis',
        'stage_duration_analysis': '📊 Analyze stage durations',
        'resolution_delay_analysis': '⏳ Analyze resolution delays'
    }
    return descriptions


def fetch_incident_data(incident_number: str) -> Tuple[bool, str]:
    """Fetch incident data from database"""
    try:
        with st.spinner(f"Fetching data for incident {incident_number}..."):
            fetch_proc = subprocess.run(
                [sys.executable, "kusto_fetcher.py", str(incident_number), "--output-dir", "icms"],
                capture_output=True,
                text=True,
                timeout=300
            )
        
        if fetch_proc.returncode != 0:
            error_msg = f"STDOUT:\n{fetch_proc.stdout}\n\nSTDERR:\n{fetch_proc.stderr}"
            return False, error_msg
        
        # Check if CSV file was created
        csv_path = os.path.join("icms", str(incident_number), f"{incident_number}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join("icms", f"{incident_number}.csv")
        
        if os.path.exists(csv_path):
            return True, f"Successfully fetched incident {incident_number}"
        else:
            return False, "CSV file was not created"
            
    except subprocess.TimeoutExpired:
        return False, "Request timed out. Please check your VPN connection."
    except Exception as e:
        return False, f"Error: {str(e)}"


def process_incident_to_json(incident_number: str) -> Tuple[bool, str]:
    """Process CSV to JSON for a single incident"""
    try:
        # Try new folder structure first, then fall back to flat
        csv_path = os.path.join("icms", str(incident_number), f"{incident_number}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join("icms", f"{incident_number}.csv")
        
        if not os.path.exists(csv_path):
            return False, f"CSV file not found: {csv_path}"
        
        with st.spinner(f"Processing CSV to JSON for incident {incident_number}..."):
            result = subprocess.run(
                [sys.executable, "transformer.py", csv_path],
                capture_output=True,
                text=True,
                timeout=300
            )
        
        if result.returncode != 0:
            return False, f"Error: {result.stderr}"
        
        json_path = os.path.join("processed_incidents", f"{incident_number}.json")
        if os.path.exists(json_path):
            return True, f"Successfully processed incident {incident_number}"
        else:
            return False, f"JSON file was not created: {json_path}"
            
    except subprocess.TimeoutExpired:
        return False, "Processing timed out"
    except Exception as e:
        return False, f"Error: {str(e)}"


def process_incident_with_ai(incident_numbers: List[str], prompt_type: str, debug: bool = False) -> Tuple[bool, Dict[str, Any], str]:
    """Process incident(s) with AI"""
    try:
        # Load prompts
        from processor import IncidentProcessor, load_prompts
        
        prompts = load_prompts(prompt_type)
        
        # Initialize processor
        processor = IncidentProcessor(
            enable_memory=True,
            enable_team_analysis=False,
            articles_path=None,
            vector_db_path=None
        )
        
        results = {}
        
        if len(incident_numbers) == 1:
            # Single incident processing
            incident_number = incident_numbers[0]
            json_path = os.path.join("processed_incidents", f"{incident_number}.json")
            
            if not os.path.exists(json_path):
                return False, {}, f"JSON file not found: {json_path}"
            
            with open(json_path, 'r', encoding='utf-8') as f:
                incident_data = json.load(f)
            
            conversation = incident_data.get('conversation', [])
            summary = incident_data.get('summary', None)
            
            # Format conversation
            formatted_content = processor.format_conversation_with_ai_summary(conversation, summary=summary)
            
            # Generate summary
            with st.spinner(f"Generating {prompt_type} summary for incident {incident_number}..."):
                summary_result = processor.generate_summary(
                    [{'type': 'text', 'content': formatted_content}],
                    prompts['system_prompt'],
                    prompts['user_prompt'],
                    prompt_type=prompt_type,
                    debug_api=debug,
                    incident_data=incident_data
                )
            
            # Save results
            operation_time = datetime.now().isoformat()
            model_name = processor.deployment_name if hasattr(processor, 'deployment_name') else "unknown"
            
            processor.save_to_json(
                conversation,
                incident_number,
                ai_summary=summary_result,
                prompt_type=prompt_type,
                operation_time=operation_time,
                model_name=model_name
            )
            
            results[incident_number] = {
                'summary': summary_result.get('summary', ''),
                'prompt_type': prompt_type,
                'timestamp': operation_time
            }
            
        else:
            # Multiple incidents - combine and process
            combined_data = {
                "incidents": [],
                "total_incidents": len(incident_numbers),
                "combined_timestamp": datetime.now().isoformat()
            }
            
            for incident_number in incident_numbers:
                json_path = os.path.join("processed_incidents", f"{incident_number}.json")
                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        incident_data = json.load(f)
                        incident_data["incident_number"] = incident_number
                        combined_data["incidents"].append(incident_data)
            
            # Process combined incidents
            with st.spinner(f"Processing {len(incident_numbers)} incidents with {prompt_type}..."):
                # This is a simplified version - full implementation would use process_multiple_incidents
                summary_result = processor.generate_summary(
                    [{'type': 'text', 'content': json.dumps(combined_data, indent=2)}],
                    prompts['system_prompt'],
                    prompts['user_prompt'],
                    prompt_type=prompt_type,
                    debug_api=debug,
                    incident_data=combined_data
                )
            
            combined_id = '_'.join(incident_numbers)
            results[combined_id] = {
                'summary': summary_result.get('summary', ''),
                'prompt_type': prompt_type,
                'timestamp': datetime.now().isoformat()
            }
        
        return True, results, "Processing completed successfully"
        
    except Exception as e:
        error_msg = f"Error during AI processing: {str(e)}\n\n{traceback.format_exc()}"
        return False, {}, error_msg


def list_processed_incidents() -> List[str]:
    """List all processed incidents"""
    processed_dir = Path("processed_incidents")
    if not processed_dir.exists():
        return []
    
    incidents = []
    for file in processed_dir.glob("*.json"):
        incident_id = file.stem
        # Skip combined incident files
        if not incident_id.startswith("combined_") and not incident_id.startswith("troubleshooting_plan_"):
            incidents.append(incident_id)
    
    return sorted(incidents, reverse=True)


def load_summary_file(incident_id: str) -> Optional[Dict[str, Any]]:
    """Load summary file for an incident"""
    summary_path = Path("summaries") / f"{incident_id}.json"
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def main():
    """Main Streamlit application"""
    
    # Sidebar
    with st.sidebar:
        st.title("📊 Incident Summarizer")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["Process Incident", "View Results", "Browse Files"],
            key="nav"
        )
        
        st.markdown("---")
        st.markdown("### 📝 Quick Info")
        st.markdown("""
        This tool helps you:
        - Process incident data
        - Generate summaries
        - Analyze incidents
        - View results
        """)
    
    # Main content area
    if page == "Process Incident":
        st.header("🔍 Process Incident")
        
        # Get available prompt types
        prompt_types = load_prompt_types()
        prompt_descriptions = get_prompt_descriptions()
        
        if not prompt_types:
            st.error("No prompt types available. Please ensure prompts.json exists.")
            return
        
        with st.form("incident_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                incident_input = st.text_area(
                    "Incident ID(s)",
                    help="Enter one or more incident IDs, separated by commas or spaces",
                    height=100
                )
            
            with col2:
                prompt_type = st.selectbox(
                    "Prompt Type",
                    options=list(prompt_types.keys()),
                    format_func=lambda x: prompt_descriptions.get(x, x),
                    help="Select the type of analysis to perform"
                )
            
            debug_mode = st.checkbox("Enable Debug Mode", value=False)
            
            submitted = st.form_submit_button("🚀 Process Incident", use_container_width=True)
        
        if submitted:
            if not incident_input.strip():
                st.error("Please enter at least one incident ID")
                return
            
            # Parse incident IDs
            incident_ids = [id.strip() for id in re.split(r'[,\s]+', incident_input) if id.strip()]
            
            if not incident_ids:
                st.error("No valid incident IDs found")
                return
            
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            total_steps = len(incident_ids) * 3  # fetch, transform, AI for each
            current_step = 0
            
            try:
                successful_incidents = []
                
                # Step 1: Fetch data
                status_text.text("📥 Step 1/3: Fetching incident data from database...")
                for incident_id in incident_ids:
                    success, message = fetch_incident_data(incident_id)
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    
                    if success:
                        successful_incidents.append(incident_id)
                        st.success(f"✅ Fetched incident {incident_id}")
                    else:
                        st.error(f"❌ Failed to fetch incident {incident_id}: {message}")
                
                if not successful_incidents:
                    st.error("No incidents were successfully fetched. Please check your VPN connection and try again.")
                    return
                
                # Step 2: Process to JSON
                status_text.text("🔄 Step 2/3: Converting CSV to JSON...")
                processed_incidents = []
                for incident_id in successful_incidents:
                    success, message = process_incident_to_json(incident_id)
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    
                    if success:
                        processed_incidents.append(incident_id)
                        st.success(f"✅ Processed incident {incident_id}")
                    else:
                        st.warning(f"⚠️ Failed to process incident {incident_id}: {message}")
                
                if not processed_incidents:
                    st.error("No incidents were successfully processed.")
                    return
                
                # Step 3: AI Processing
                status_text.text("🤖 Step 3/3: Generating AI summary...")
                success, results, message = process_incident_with_ai(
                    processed_incidents,
                    prompt_type,
                    debug_mode
                )
                
                current_step = total_steps
                progress_bar.progress(1.0)
                
                if success:
                    status_text.text("✅ Processing completed!")
                    st.session_state.results = results
                    st.session_state.processed_incidents = processed_incidents
                    
                    # Display results
                    with results_container:
                        st.success("🎉 Processing completed successfully!")
                        st.markdown("---")
                        st.subheader("📋 Results")
                        
                        for incident_id, result in results.items():
                            with st.expander(f"Incident {incident_id} - {prompt_type}", expanded=True):
                                st.markdown("### Summary")
                                st.markdown(result['summary'])
                                st.caption(f"Generated at: {result['timestamp']}")
                else:
                    st.error(f"❌ AI Processing failed: {message}")
                    st.code(message, language='text')
            
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.code(traceback.format_exc(), language='text')
    
    elif page == "View Results":
        st.header("📊 View Results")
        
        # List available summaries
        summary_dir = Path("summaries")
        if not summary_dir.exists():
            st.info("No summaries directory found. Process some incidents first.")
            return
        
        summary_files = list(summary_dir.glob("*.json"))
        
        if not summary_files:
            st.info("No summary files found. Process some incidents first.")
            return
        
        # Select summary to view
        summary_options = {f.stem: f for f in sorted(summary_files, key=lambda x: x.stat().st_mtime, reverse=True)}
        selected_summary = st.selectbox(
            "Select Summary",
            options=list(summary_options.keys()),
            format_func=lambda x: f"{x} ({datetime.fromtimestamp(summary_options[x].stat().st_mtime).strftime('%Y-%m-%d %H:%M')})"
        )
        
        if selected_summary:
            summary_file = summary_options[selected_summary]
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            st.markdown("---")
            st.subheader(f"Summary: {selected_summary}")
            
            # Display summary content
            if 'summary' in summary_data:
                if isinstance(summary_data['summary'], dict) and 'summary' in summary_data['summary']:
                    st.markdown(summary_data['summary']['summary'])
                else:
                    st.markdown(summary_data['summary'])
            elif 'ai_summary' in summary_data:
                if isinstance(summary_data['ai_summary'], dict) and 'summary' in summary_data['ai_summary']:
                    st.markdown(summary_data['ai_summary']['summary'])
                else:
                    st.markdown(str(summary_data['ai_summary']))
            
            # Display metadata
            with st.expander("Metadata"):
                st.json({k: v for k, v in summary_data.items() if k not in ['summary', 'ai_summary', 'conversation']})
    
    elif page == "Browse Files":
        st.header("📁 Browse Files")
        
        tab1, tab2, tab3 = st.tabs(["Processed Incidents", "Raw Data", "Logs"])
        
        with tab1:
            st.subheader("Processed Incidents")
            incidents = list_processed_incidents()
            
            if incidents:
                selected_incident = st.selectbox("Select Incident", incidents)
                
                if selected_incident:
                    json_path = Path("processed_incidents") / f"{selected_incident}.json"
                    if json_path.exists():
                        with open(json_path, 'r', encoding='utf-8') as f:
                            incident_data = json.load(f)
                        
                        st.json(incident_data)
            else:
                st.info("No processed incidents found.")
        
        with tab2:
            st.subheader("Raw Incident Data")
            icms_dir = Path("icms")
            
            if icms_dir.exists():
                incident_folders = [d for d in icms_dir.iterdir() if d.is_dir()]
                
                if incident_folders:
                    folder_names = [f.name for f in sorted(incident_folders, reverse=True)]
                    selected_folder = st.selectbox("Select Incident Folder", folder_names)
                    
                    if selected_folder:
                        folder_path = icms_dir / selected_folder
                        csv_files = list(folder_path.glob("*.csv"))
                        
                        if csv_files:
                            selected_csv = st.selectbox("Select CSV File", [f.name for f in csv_files])
                            csv_path = folder_path / selected_csv
                            
                            if csv_path.exists():
                                with open(csv_path, 'r', encoding='utf-8') as f:
                                    csv_content = f.read()
                                
                                st.text_area("CSV Content", csv_content, height=400)
                else:
                    st.info("No incident folders found.")
            else:
                st.info("No icms directory found.")
        
        with tab3:
            st.subheader("Logs")
            logs_dir = Path("logs")
            
            if logs_dir.exists():
                log_files = list(logs_dir.glob("*.log"))
                
                if log_files:
                    log_options = {f.name: f for f in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)}
                    selected_log = st.selectbox("Select Log File", list(log_options.keys()))
                    
                    if selected_log:
                        log_file = log_options[selected_log]
                        with open(log_file, 'r', encoding='utf-8') as f:
                            log_content = f.read()
                        
                        st.text_area("Log Content", log_content, height=400)
                else:
                    st.info("No log files found.")
            else:
                st.info("No logs directory found.")


if __name__ == "__main__":
    main()
