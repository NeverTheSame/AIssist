import csv
import json
import sys
from datetime import datetime
import os
import logging
from bs4 import BeautifulSoup

# Configure logging
def setup_logging():
    """Setup logging for incident dumper"""
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
    file_handler = logging.FileHandler('logs/transformer.log', encoding='utf-8')
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

def clean_html_content(html_content):
    """Clean HTML content by removing encrypted images and rich text formatting"""
    if not html_content:
        return ""
    
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove all img tags with data:image/png src (encrypted images)
    for img in soup.find_all('img'):
        src = img.get('src', '')
        if src.startswith('data:image/png'):
            logger.info(f"Removing encrypted image: {src[:50]}...")
            img.decompose()
    
    # Remove other potentially problematic elements
    for element in soup.find_all(['script', 'style', 'link']):
        element.decompose()
    
    # Convert to plain text, preserving some structure
    # Replace common HTML elements with appropriate text formatting
    for tag in soup.find_all(['p', 'div', 'br']):
        if tag.name == 'br':
            tag.replace_with('\n')
        elif tag.name in ['p', 'div']:
            # Add newlines around paragraphs and divs
            tag.insert_before('\n')
            tag.insert_after('\n')
    
    # Get text and clean up whitespace
    text = soup.get_text(separator=' ', strip=True)
    
    # Clean up excessive whitespace and newlines
    import re
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Replace multiple newlines with double newlines
    text = re.sub(r' +', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()
    
    return text

def parse_multi_section_csv(file_path):
    sections = {}
    current_section = None
    current_headers = None
    
    # Increase CSV field size limit to handle large fields
    import csv
    maxInt = sys.maxsize
    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)
    
    try:
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row_num, row in enumerate(reader, 1):
                try:
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
                        # Truncate any fields that are too long
                        processed_row = []
                        for cell in row:
                            if isinstance(cell, str) and len(cell) > 100000:  # Limit to 100KB per field
                                logger.warning(f"Truncating field at row {row_num} that was {len(cell)} characters long")
                                cell = cell[:100000] + "... [TRUNCATED]"
                            processed_row.append(cell)
                        
                        item = {header: value for header, value in zip(current_headers, processed_row)}
                        sections[current_section].append(item)
                except Exception as e:
                    logger.warning(f"Error processing row {row_num}: {e}. Skipping this row.")
                    continue
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        raise
    
    return sections

def dump_discussion_items_to_json(discussion_items, incident_number, output_dir="processed_incidents", summary_content=None):
    """Dump incidents to a JSON file in a clean format, including internal AI summary if provided. (No summary field in output)"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean summary content if provided
    cleaned_summary = None
    if summary_content:
        logger.info("Cleaning summary content...")
        cleaned_summary = clean_html_content(summary_content)
        logger.info(f"Summary content cleaned. Original length: {len(summary_content)}, Cleaned length: {len(cleaned_summary)}")
    
    # Format the data for AI processing
    formatted_discussion_items = []
    filtered_count = 0
    for discussion_item in discussion_items:
        raw_content = (discussion_item.get('Text') or discussion_item.get('content') or '').strip()
        # Clean HTML/rich text using BeautifulSoup
        if raw_content:
            clean_content = clean_html_content(raw_content)
        else:
            clean_content = ''
        
        # Get author information
        author = discussion_item.get('ChangedBy') or discussion_item.get('author')
        
        # Filter out unwanted entries
        # Skip entries with author "gautosvc" and content starting with "Support ICM enrichment CEM MDE"
        if author == "gautosvc" and clean_content.startswith("Support ICM enrichment CEM MDE"):
            logger.info(f"Filtering out gautosvc entry with Support ICM enrichment content")
            filtered_count += 1
            continue
        
        formatted_discussion_item = {
            "timestamp": discussion_item.get('Date') or discussion_item.get('timestamp'),
            "author": author,
            "content": clean_content
        }
        formatted_discussion_items.append(formatted_discussion_item)
    
    # Sort by timestamp
    formatted_discussion_items.sort(key=lambda x: x['timestamp'])
    

    
    # Create the output file (without 'incident_' prefix)
    output_file = os.path.join(output_dir, f"{incident_number}.json")
    
    # Write to file (no summary field)
    with open(output_file, 'w', encoding='utf-8') as f:
        output_data = {
            "incident_id": incident_number,
            "total_entries": len(formatted_discussion_items),
            "conversation": formatted_discussion_items
        }
        
        # Add cleaned summary if available
        if cleaned_summary:
            output_data["summary"] = cleaned_summary
            logger.info(f"Added cleaned summary to output (length: {len(cleaned_summary)})")
        
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Processed {len(formatted_discussion_items)} entries (filtered out {filtered_count} unwanted entries)")
    print(f"✅ Created: {output_file}")
    
    return output_file, filtered_count

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process and dump incident data to JSON format.')
    parser.add_argument('file_path', help='Path to the CSV file (e.g., icms/12345/12345.csv or icms/12345.csv)')
    parser.add_argument('--output-dir', default='processed_incidents', 
                      help='Directory to store processed incident files')
    args = parser.parse_args()
    
    try:
        # Parse the multi-section CSV
        sections = parse_multi_section_csv(args.file_path)
        
        # Debug: Log all available sections
        logger.info(f"Available sections in CSV: {list(sections.keys())}")
        for section_name, section_data in sections.items():
            logger.info(f"Section '{section_name}': {len(section_data)} items")
        
        # Extract incident number from filename
        filename = os.path.basename(args.file_path)
        incident_number = ''.join(filter(str.isdigit, filename))
        
        # Discussions section
        discussion_items = sections.get('Discussions', [])
        
        # Authored summary section (new field from updated query)
        summary_content = None
        summary_section = sections.get('Authored summary', [])
        if summary_section:
            summary_content = summary_section[0].get('Summary', '') if len(summary_section) == 1 else summary_section[0].get('Summary', '')
            logger.info(f"Found authored summary content (length: {len(summary_content)})")
        

        
        # Dump to JSON
        output_file, filtered_count = dump_discussion_items_to_json(
            discussion_items, 
            incident_number,
            args.output_dir,
            summary_content
        )
        
        print(f"\nProcessed incident data has been saved to: {output_file}")
        print(f"Total entries processed: {len(discussion_items)}")
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} gautosvc entries with Support ICM enrichment content")
        if summary_content:
            print("Authored summary content has been cleaned and included in the JSON file")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 