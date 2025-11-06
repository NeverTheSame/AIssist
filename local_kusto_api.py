#!/usr/bin/env python3
"""
Local Kusto API Service

Simple Flask API that runs on your local machine (with VPN) and fetches
incident data from Kusto. Streamlit Cloud can call this API.

Usage:
    python3 local_kusto_api.py [--port 5000]
    
Then expose with ngrok:
    ngrok http 5000
    
Set the ngrok URL in Streamlit Settings as KUSTO_API_URL.

Requirements:
    pip install flask flask-cors ngrok
"""

import sys
import os
import subprocess
import json
import tempfile
import shutil
from pathlib import Path
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import argparse

app = Flask(__name__)
CORS(app)  # Allow Streamlit Cloud to call this API

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "Kusto API is running"})

@app.route('/fetch/<incident_number>', methods=['GET'])
def fetch_incident(incident_number):
    """Fetch incident data from Kusto and return CSV file"""
    try:
        print(f"Fetching incident {incident_number}...")
        
        # Fetch from Kusto using local kusto_fetcher.py
        result = subprocess.run(
            [sys.executable, "kusto_fetcher.py", str(incident_number)],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            return jsonify({
                "error": "Failed to fetch from Kusto",
                "details": result.stderr
            }), 500
        
        # Find the CSV file
        csv_path = Path(f"icms/{incident_number}/{incident_number}.csv")
        if not csv_path.exists():
            csv_path = Path(f"icms/{incident_number}.csv")
        
        if not csv_path.exists():
            return jsonify({
                "error": f"CSV file not found for incident {incident_number}"
            }), 404
        
        # Return the CSV file
        return send_file(
            str(csv_path),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"{incident_number}.csv"
        )
        
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Request timed out"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/fetch_json/<incident_number>', methods=['GET'])
def fetch_incident_json(incident_number):
    """Fetch incident data and return as JSON (CSV converted to JSON)"""
    try:
        # First fetch CSV
        csv_response = fetch_incident(incident_number)
        if isinstance(csv_response, tuple):  # Error response
            return csv_response
        
        # Find CSV path
        csv_path = Path(f"icms/{incident_number}/{incident_number}.csv")
        if not csv_path.exists():
            csv_path = Path(f"icms/{incident_number}.csv")
        
        # Convert CSV to JSON using transformer.py
        result = subprocess.run(
            [sys.executable, "transformer.py", str(csv_path)],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            return jsonify({
                "error": "Failed to convert CSV to JSON",
                "details": result.stderr
            }), 500
        
        # Load JSON file
        json_path = Path(f"processed_incidents/{incident_number}.json")
        if not json_path.exists():
            return jsonify({
                "error": f"JSON file not found for incident {incident_number}"
            }), 404
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return jsonify(data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    parser = argparse.ArgumentParser(description='Local Kusto API Service')
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to run the API on (default: 5000)'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    args = parser.parse_args()
    
    print(f"Starting Kusto API on {args.host}:{args.port}")
    print(f"Endpoints:")
    print(f"  GET /health - Health check")
    print(f"  GET /fetch/<incident_number> - Fetch CSV file")
    print(f"  GET /fetch_json/<incident_number> - Fetch JSON file")
    print(f"\nTo expose publicly, run:")
    print(f"  ngrok http {args.port}")
    print(f"\nThen set KUSTO_API_URL in Streamlit Settings to the ngrok URL")
    
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()

