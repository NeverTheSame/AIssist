#!/usr/bin/env python3
"""
Gap Analysis Script - Analyzes gaps between incident data and troubleshooting articles
"""

import json
import os
import sys
from config import config

def load_article_search_results(incident_id: str) -> dict:
    """Load article search results for the given incident."""
    search_file = f"processed_incidents/incident_{incident_id}_article_search.json"
    if not os.path.exists(search_file):
        print(f"❌ Article search file not found: {search_file}")
        return None
    
    try:
        with open(search_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading article search results: {e}")
        return None

def display_articles(articles: list):
    """Display available articles for selection."""
    print("\n" + "="*80)
    print("SELECT AN ARTICLE FOR GAP ANALYSIS")
    print("="*80)
    
    for i, article in enumerate(articles, 1):
        title = article.get('title', 'Unknown Title')
        url = article.get('url', 'No URL')
        relevance = article.get('relevance_score', 'N/A')
        
        print(f"{i}. {title}")
        print(f"   URL: {url}")
        print(f"   Relevance Score: {relevance}")
        print()

def get_article_content(article: dict, article_searcher) -> str:
    """Get the content of the selected article from the actual TSG file."""
    url = article.get('url', '')
    
    if not url:
        return "No URL available for this article."
    
    # Use the ArticleSearcher to get the full content
    try:
        full_content = article_searcher.get_full_article_content(url)
        if full_content and not full_content.startswith("Article file not found") and not full_content.startswith("Error reading article"):
            return full_content
        else:
            return f"Unable to retrieve article content. Please check the file path: {url}"
    except Exception as e:
        return f"Error retrieving article content: {e}"

def run_gap_analysis(incident_id: str, selected_article: dict, article_searcher):
    """Run the gap analysis using Azure OpenAI directly."""
    
    # Load the incident data
    incident_file = f"processed_incidents/{incident_id}.json"
    if not os.path.exists(incident_file):
        print(f"❌ Incident data not found: {incident_file}")
        return
    
    with open(incident_file, 'r') as f:
        incident_data = json.load(f)
    
    # Get the article content
    article_content = get_article_content(selected_article, article_searcher)
    
    # Load the prompts
    with open('prompts.json', 'r') as f:
        prompts = json.load(f)
    
    gap_analysis_prompt = prompts.get('troubleshooting_gap_analysis')
    if not gap_analysis_prompt:
        print("❌ troubleshooting_gap_analysis prompt not found in prompts.json")
        return
    
    # Create the analysis content
    analysis_content = f"""
## Current Incident Data
{json.dumps(incident_data, indent=2)}

## Selected Troubleshooting Article
{article_content}
"""
    
    print("\n" + "="*80)
    print("RUNNING GAP ANALYSIS")
    print("="*80)
    print(f"Incident: {incident_id}")
    print(f"Article: {selected_article.get('title', 'Unknown')}")
    print("="*80)
    
    # Generate the gap analysis using Azure OpenAI directly
    try:
        from openai import AzureOpenAI
        
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            azure_endpoint=config.azure_openai_endpoint
        )
        
        # Create a more focused prompt that explicitly references the incident data
        focused_user_prompt = f"""
IMPORTANT: You must analyze the specific incident data provided below. Do NOT generate generic troubleshooting steps.

INCIDENT DATA TO ANALYZE:
{incident_data.get('summary', 'No summary available')}

TROUBLESHOOTING ARTICLE CONTENT:
{article_content}

Your task is to:
1. Analyze the SPECIFIC incident data above (Device Control policy issue on macOS)
2. Compare it against the troubleshooting article procedures
3. Identify gaps and create an execution plan

Focus ONLY on the Device Control policy issue described in the incident data. Do NOT analyze network connectivity or any other generic issues.
"""
        
        print("🤖 Executing gap analysis with Azure OpenAI...")
        response = client.chat.completions.create(
            model=config.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": gap_analysis_prompt['system_prompt']},
                {"role": "user", "content": focused_user_prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        # Extract the result
        result = response.choices[0].message.content
        
        # Save the gap analysis result
        output_file = f"gap_analysis_{incident_id}_{selected_article.get('title', 'unknown').replace(' ', '_').replace('/', '_')}.txt"
        with open(output_file, 'w') as f:
            f.write(f"GAP ANALYSIS RESULTS\n")
            f.write(f"Incident: {incident_id}\n")
            f.write(f"Article: {selected_article.get('title', 'Unknown')}\n")
            f.write("="*80 + "\n\n")
            f.write(result)
        
        print(f"\n✅ Gap analysis completed and saved to: {output_file}")
        
        # Display the result
        print("\n" + "="*80)
        print("GAP ANALYSIS RESULTS")
        print("="*80)
        print(result)
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error running gap analysis: {e}")
        import traceback
        traceback.print_exc()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 gap_analysis.py <incident_id>")
        print("Example: python3 gap_analysis.py 676925403")
        return
    
    incident_id = sys.argv[1]
    
    # Load article search results
    search_data = load_article_search_results(incident_id)
    if not search_data:
        return
    
    # Debug: Print the structure
    print(f"Debug: Search data keys: {list(search_data.keys())}")
    if 'content' in search_data:
        print(f"Debug: Content keys: {list(search_data['content'].keys())}")
        if 'search_results' in search_data['content']:
            print(f"Debug: Search results keys: {list(search_data['content']['search_results'].keys())}")
    
    # Extract articles from the nested structure
    articles = search_data.get('content', {}).get('search_results', {}).get('search_results', [])
    if not articles:
        print("❌ No articles found in search results")
        print(f"Debug: Articles found: {len(articles) if articles else 0}")
        return
    
    # Display articles for selection
    display_articles(articles)
    
    # Get user selection
    while True:
        try:
            selection = input(f"\nSelect an article (1-{len(articles)}) or 'q' to quit: ").strip()
            
            if selection.lower() == 'q':
                print("Exiting...")
                return
            
            selection_num = int(selection)
            if 1 <= selection_num <= len(articles):
                selected_article = articles[selection_num - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(articles)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Initialize article searcher for content retrieval
    try:
        from article_searcher import ArticleSearcher
        # The article search results file contains the search results, not the vector database
        # We need to use the original vector database path that was used for the search
        vector_db_path = "/Users/kirill/Documents/M/Code/AzureDevops/ArticlesInventorizer/article_embeddings.json"
        article_searcher = ArticleSearcher(
            vector_db_path=vector_db_path,
            use_azure=bool(config.azure_openai_api_key),
            use_azure_5=bool(config.azure_openai_5_api_key),
            use_zai=bool(config.zai_api_key)
        )
    except Exception as e:
        print(f"❌ Error initializing article searcher: {e}")
        article_searcher = None
    
    # Run gap analysis
    run_gap_analysis(incident_id, selected_article, article_searcher)

if __name__ == "__main__":
    main()
