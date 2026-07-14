import argparse
import json

from app.rag import answer_query

parser = argparse.ArgumentParser(description="Ask MedGuard (baseline RAG)")
parser.add_argument("--query", required=True, help="clinical question based on hypertension management")
args = parser.parse_args()

print(json.dumps(answer_query(args.query), indent=2))