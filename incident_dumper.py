import csv
import json
from datetime import datetime
import os
from incident_processor import IncidentProcessor
import logging
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_multi_section_csv(file_path):
    sections = {}
    current_section = None
    current_headers = None
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row or all(cell.strip() == '' for cell in row):
                continue  # skip blank lines
            if row[0].startswith('---') and row[0].endswith('---'):
                # New section header
                current_section = row[0].strip('- ').strip()
                sections[current_section] = []
                current_headers = None
            elif current_section and current_headers is None:
                current_headers = row
            elif current_section and current_headers:
                item = {header: value for header, value in zip(current_headers, row)}
                sections[current_section].append(item)
    return sections

def dump_discussion_items_to_json(discussion_items, incident_number, output_dir="processed_incidents", internal_ai_summary=None):
    """Dump incidents to a JSON file in a clean format, including internal AI summary if provided. (No summary field in output)"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Format the data for AI processing
    formatted_discussion_items = []
    for discussion_item in discussion_items:
        raw_content = (discussion_item.get('Text') or discussion_item.get('content') or '').strip()
        # Clean HTML/rich text using BeautifulSoup
        if raw_content:
            soup = BeautifulSoup(raw_content, "html.parser")
            clean_content = soup.get_text(" ", strip=True)
        else:
            clean_content = ''
        formatted_discussion_item = {
            "timestamp": discussion_item.get('Date') or discussion_item.get('timestamp'),
            "author": discussion_item.get('ChangedBy') or discussion_item.get('author'),
            "content": clean_content
        }
        formatted_discussion_items.append(formatted_discussion_item)
    
    # Sort by timestamp
    formatted_discussion_items.sort(key=lambda x: x['timestamp'])
    
    # Remove 'ICM_IncidentId' from internal_ai_summary if present
    if isinstance(internal_ai_summary, dict) and 'ICM_IncidentId' in internal_ai_summary:
        internal_ai_summary = dict(internal_ai_summary)  # make a copy
        internal_ai_summary.pop('ICM_IncidentId', None)
    elif isinstance(internal_ai_summary, list):
        for item in internal_ai_summary:
            if isinstance(item, dict) and 'ICM_IncidentId' in item:
                item.pop('ICM_IncidentId', None)
    
    # Create the output file (without 'incident_' prefix)
    output_file = os.path.join(output_dir, f"{incident_number}.json")
    
    # Write to file (no summary field)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "incident_id": incident_number,
            "total_entries": len(formatted_discussion_items),
            "conversation": formatted_discussion_items,
            "internal_ai_summary": internal_ai_summary
        }, f, indent=2, ensure_ascii=False)
    
    return output_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process and dump incident data to JSON format.')
    parser.add_argument('file_path', help='Path to the CSV file (e.g., icms/12345.csv)')
    parser.add_argument('--output-dir', default='processed_incidents', 
                      help='Directory to store processed incident files')
    args = parser.parse_args()
    
    try:
        # Parse the multi-section CSV
        sections = parse_multi_section_csv(args.file_path)
        
        # Extract incident number from filename
        filename = os.path.basename(args.file_path)
        incident_number = ''.join(filter(str.isdigit, filename))
        
        # Discussions section
        discussion_items = sections.get('Discussions', [])
        # Internal AI Summary section (take the first row if exists)
        internal_ai_summary = None
        ai_summary_section = sections.get('Internal AI Summary', [])
        if ai_summary_section:
            internal_ai_summary = ai_summary_section[0] if len(ai_summary_section) == 1 else ai_summary_section
        
        # Dump to JSON (no summary)
        output_file = dump_discussion_items_to_json(
            discussion_items, 
            incident_number,
            args.output_dir,
            internal_ai_summary
        )
        
        print(f"\nProcessed incident data has been saved to: {output_file}")
        print(f"Total entries processed: {len(discussion_items)}")
        if internal_ai_summary:
            print("Internal AI Summary has been included in the JSON file")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 