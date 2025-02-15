import subprocess
import json
from datetime import datetime
import os
import sqlite3
from pathlib import Path
import glob
import re
import base64
import numpy as np
import requests  # Add this import at the top
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI
import threading
import uuid


from flask import Flask, request, jsonify


app = Flask(__name__)
RESULTS_DIR = './task_results'
os.makedirs(RESULTS_DIR, exist_ok=True)


class OperationType(Enum):
    RUN_SCRIPT = "run_script"
    FORMAT_MARKDOWN = "format_markdown"
    COUNT_WEEKDAYS = "count_weekdays"
    SORT_JSON = "sort_json"
    EXTRACT_RECENT_LOGS = "extract_recent_logs"
    CREATE_MARKDOWN_INDEX = "create_markdown_index"
    EXTRACT_EMAIL_SENDER = "extract_email_sender"
    EXTRACT_CREDIT_CARD = "extract_credit_card"
    CALCULATE_GOLD_SALES = "calculate_gold_sales"
    FIND_SIMILAR_COMMENTS = "find_similar_comments"

@dataclass
class TaskParameters:
    weekday: Optional[str] = None
    script_url: Optional[str] = None
    llm_instruction: Optional[str] = None

@dataclass
class ParsedTask:
    input_files: List[str]
    operation: OperationType
    parameters: TaskParameters
    output_file: str
class TaskManager:
    def __init__(self):
        self.tasks = {}

    def create_task(self, task_id, output_path):
        self.tasks[task_id] = {
            'status': 'pending',
            'output_path': output_path,
            'error': None
        }

    def update_task(self, task_id, status, error=None):
        if task_id in self.tasks:
            self.tasks[task_id]['status'] = status
            self.tasks[task_id]['error'] = error

    def get_task(self, task_id):
        return self.tasks.get(task_id, None)

task_manager = TaskManager()

class TaskExecutor:
  
    def __init__(self, api_token: Optional[str] = None):
        
        self.api_token = api_token or os.getenv("API_TOKEN")
        #api_token or os.getenv('AIPROXY_TOKEN')
        self.base_url = "https://aiproxy.sanand.workers.dev/openai/v1/"
        
        if not self.api_token:
            raise ValueError("Missing AIPROXY_TOKEN. Get it from https://aiproxy.sanand.workers.dev/")
 
    def parse_task_description(self, task_description: str) -> ParsedTask:
        """Parse natural language task description using AI Proxy"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": task_description}],
            "functions": [{
                "name": "parse_task",
                "description": "Parse a task description into structured components",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of input file paths"
                        },
                        "operation": {
                            "type": "string",
                            "enum": [op.value for op in OperationType],
                            "description": "The operation to perform"
                        },
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "weekday": {"type": "string"},
                                "script_url": {"type": "string"},
                                "llm_instruction": {"type": "string"}
                            }
                        },
                        "output_file": {"type": "string"}
                    },
                    "required": ["operation", "output_file"]
                }
            }],
            "function_call": {"name": "parse_task"},
            "temperature": 0
        }

        try:
            response = requests.post(
                f"{self.base_url}chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            arguments = json.loads(result['choices'][0]['message']['function_call']['arguments'])
            print(result)
            return ParsedTask(
                input_files=arguments.get("input_files", []),
                operation=OperationType(arguments["operation"]),
                parameters=TaskParameters(**arguments.get("parameters", {})),
                output_file=arguments["output_file"]
            )
            
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"AI Proxy error: {e.response.text}") from e
        except KeyError as e:
            raise RuntimeError(f"Malformed API response: {str(e)}") from e

    def format_markdown(self, input_file: str, output_file: str) -> None:
        """Format markdown file maintaining specific whitespace rules"""
      
        with open(input_file, 'r') as f:
            content = f.read()
            
        # Format headings (ensure single space after #)
        content = re.sub(r'#\s*([^\n]+)', r'# \1', content)
        
        # Format list items (consistent spacing)
        content = re.sub(r'^[-+*]\s+', r'- ', content, flags=re.MULTILINE)
        print(output_file)
        with open(output_file, 'w') as f:
            f.write(content)

    def count_weekdays(self, input_file: str, weekday: str, output_file: str) -> None:
        """Count occurrences of specified weekday in date file"""
        weekday = weekday.capitalize()
        count = 0
        
        with open(input_file, 'r') as f:
            for line in f:
                try:
                    # Parse different date formats
                    for fmt in [
                        "%Y-%m-%d",
                        "%d-%b-%Y",
                        "%b %d, %Y",
                        "%Y/%m/%d %H:%M:%S"
                    ]:
                        try:
                            date = datetime.strptime(line.strip(), fmt)
                            if date.strftime('%A') == weekday:
                                count += 1
                            break
                        except ValueError:
                            continue
                except Exception:
                    continue

        with open(output_file, 'w') as f:
            f.write(str(count))

    def sort_contacts(self, input_file: str, output_file: str) -> None:
        """Sort contacts by last name, then first name"""
        with open(input_file, 'r') as f:
            contacts = json.load(f)
        
        sorted_contacts = sorted(
            contacts,
            key=lambda x: (x['last_name'], x['first_name'])
        )
        
        with open(output_file, 'w') as f:
            json.dump(sorted_contacts, f, indent=2)

    def extract_recent_logs(self, log_dir: str, output_file: str) -> None:
        """Extract contents of most recently modified log files"""
        log_files = glob.glob(os.path.join(log_dir, "*.log"))
        if not log_files:
            raise ValueError(f"No log files found in {log_dir}")
            
        # Get modification times and sort
        files_with_time = [(f, os.path.getmtime(f)) for f in log_files]
        sorted_files = sorted(files_with_time, key=lambda x: x[1], reverse=True)
        
        # Take most recent files (up to 3)
        recent_files = sorted_files[:3]
        
        # Extract and combine contents
        contents = []
        for file_path, _ in recent_files:
            with open(file_path, 'r') as f:
                contents.append(f.read())
                
        with open(output_file, 'w') as f:
            f.write('\n---\n'.join(contents))

    def create_markdown_index(self, docs_dir: str, output_file: str) -> None:
        """Create index of markdown files with their headings"""
        md_files = glob.glob(os.path.join(docs_dir, "**/*.md"), recursive=True)
        print("re")
        index = []
        for file_path in sorted(md_files):
            rel_path = os.path.relpath(file_path, docs_dir)
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Extract first heading
            heading_match = re.search(r'^#\s*(.+)$', content, re.MULTILINE)
            heading = heading_match.group(1) if heading_match else "Untitled"
            
            index.append(f"- [{heading}]({rel_path})")
            print(heading)
        with open(output_file, 'w') as f:
            f.write('\n'.join(index))

    def extract_email_sender(self, email_file: str, output_file: str) -> None:
        """Extract sender information from email file"""
        with open(email_file, 'r') as f:
            content = f.read()
            
        # Extract From field
        from_match = re.search(r'From: "([^"]+)" <([^>]+)>', content)
        if not from_match:
            raise ValueError("Could not find sender information")
            
        name, email = from_match.groups()
        
        with open(output_file, 'w') as f:
            json.dump({"name": name, "email": email}, f, indent=2)

    def extract_credit_card(self, image_file: str, output_file: str) -> None:
        """Extract credit card number from image using OCR patterns"""
        # Import only when needed
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise ImportError("pytesseract and Pillow are required for OCR")
            
        img = Image.open(image_file)
        text = pytesseract.image_to_string(img)
        
        # Find credit card number pattern
        cc_pattern = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        cc_match = re.search(cc_pattern, text)
        
        if not cc_match:
            raise ValueError("Could not find credit card number in image")
            
        with open(output_file, 'w') as f:
            f.write(cc_match.group(0).replace(' ', ''))

    def calculate_gold_sales(self, db_file: str, output_file: str) -> None:
        """Calculate total revenue from gold ticket sales"""
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT SUM(units * price) as revenue
            FROM tickets
            WHERE type = 'Gold'
        """)
        
        revenue = cursor.fetchone()[0]
        conn.close()
        
        with open(output_file, 'w') as f:
            f.write(f"{revenue:.2f}")

    def find_similar_comments(self, comments_file: str, output_file: str) -> None:
        """Find similar comments using TF-IDF and cosine similarity"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        with open(comments_file, 'r') as f:
            comments = f.read().split('\n')
            
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(comments)
        
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find most similar pair
        max_sim = 0
        pair = (0, 0)
        
        for i in range(len(comments)):
            for j in range(i + 1, len(comments)):
                if similarity_matrix[i, j] > max_sim:
                    max_sim = similarity_matrix[i, j]
                    pair = (i, j)
                    
        with open(output_file, 'w') as f:
            json.dump({
                "comment1": comments[pair[0]],
                "comment2": comments[pair[1]],
                "similarity": float(max_sim)
            }, f, indent=2)

    def execute(self, task_description: str) -> None:
        """Execute task based on natural language description"""
        parsed = self.parse_task_description(task_description)
        
        handler_map = {
            OperationType.FORMAT_MARKDOWN: self.format_markdown,
            OperationType.COUNT_WEEKDAYS: self.count_weekdays,
            OperationType.SORT_JSON: self.sort_contacts,
            OperationType.EXTRACT_RECENT_LOGS: self.extract_recent_logs,
            OperationType.CREATE_MARKDOWN_INDEX: self.create_markdown_index,
            OperationType.EXTRACT_EMAIL_SENDER: self.extract_email_sender,
            OperationType.EXTRACT_CREDIT_CARD: self.extract_credit_card,
            OperationType.CALCULATE_GOLD_SALES: self.calculate_gold_sales,
            OperationType.FIND_SIMILAR_COMMENTS: self.find_similar_comments
        }
        
        task_data_dirs = {
            OperationType.FORMAT_MARKDOWN: "./data/docs/agent/director.md",
            OperationType.COUNT_WEEKDAYS: "./data/dates",
            OperationType.SORT_JSON: "./data/contacts.json",
            OperationType.EXTRACT_RECENT_LOGS: "/data/logs",
            OperationType.CREATE_MARKDOWN_INDEX: "/data/docs",
            OperationType.EXTRACT_EMAIL_SENDER: "/data/emails",
            OperationType.EXTRACT_CREDIT_CARD: "/data/images",
            OperationType.CALCULATE_GOLD_SALES: "/data/databases",
            OperationType.FIND_SIMILAR_COMMENTS: "/data/comments",
        }
        
        if parsed.operation not in handler_map:
            raise ValueError(f"Unsupported operation: {parsed.operation}")
        
        handler = handler_map[parsed.operation]
        print(handler)
        # Determine input file or directory
        input_path = parsed.input_files[0] if parsed.input_files else task_data_dirs.get(parsed.operation, "")
        if not input_path:
            raise ValueError(f"No valid input path found for operation: {parsed.operation}")
        
        # Execute the appropriate handler function
        if parsed.operation == OperationType.COUNT_WEEKDAYS:
            handler(input_path, parsed.parameters.weekday or "Wednesday", parsed.output_file)
        elif parsed.operation in (OperationType.EXTRACT_RECENT_LOGS, OperationType.CREATE_MARKDOWN_INDEX):
            handler(os.path.dirname(input_path), parsed.output_file)
        else:
            handler(input_path, parsed.output_file)

    @app.route('/tasks', methods=['POST'])
    def handle_task():
        data = request.get_json()
        task_desc = data.get('task_description')
        

        task_id = str(uuid.uuid4())
        output_path = os.path.join(RESULTS_DIR, f"{task_id}.out")

        executor = TaskExecutor()
        try:
           executor.execute(task_desc)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

        task_manager.create_task(task_id, output_path)

        def background_task():
            try:
                executor.execute_parsed(parsed, user_email)
                task_manager.update_task(task_id, 'completed')
            except Exception as e:
                task_manager.update_task(task_id, 'error', str(e))

        threading.Thread(target=background_task).start()
        return jsonify({"task_id": task_id}), 202


    @app.route('/read', methods=['GET'])
    def read_file():
        file_path = request.args.get('path')  # Get file path from query params

        if not file_path:
            return jsonify({"error": "Missing 'path' parameter"}), 400

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return jsonify({"content": content})
        except FileNotFoundError:
            return jsonify({"error": "File not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

def main():
    app.run(host='0.0.0.0', port=5000)
    
if __name__ == "__main__":
    main()