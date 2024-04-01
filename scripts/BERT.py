import json
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to load JSON file
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Load dataset from JSON file
document_file_path = 'pubmed2023.json'
documents_data = load_json_file(document_file_path)

# Extract document titles and abstracts
documents = [entry['article_abstract'] for entry in documents_data]

# Function to load JSONL file
#def load_jsonl_file(file_path):
    #data = []
    #with open(file_path, 'r', encoding='utf-8') as file:
        #for line in file:
            #data.append(json.loads(line))
    #return data

# Load dataset from 11B1 file
query_file_path = '11B1_golden.json'
query_data = load_json_file(query_file_path)

# Extract queries from dataset
queries = [question['body'] for question in query_data['questions']]

# Tokenize queries and documents
query_tokens = tokenizer(queries, padding=True, truncation=True, return_tensors='pt')
document_tokens = tokenizer(documents, padding=True, truncation=True, return_tensors='pt')

# Encode queries and documents using BERT
with torch.no_grad():
    query_outputs = model(**query_tokens)
    document_outputs = model(**document_tokens)

# Compute similarity scores between queries and documents using cosine similarity
query_embeddings = query_outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Use CLS token for representation
document_embeddings = document_outputs.last_hidden_state[:, 0, :].cpu().numpy()
similarity_scores = cosine_similarity(query_embeddings, document_embeddings)

# Rank documents based on similarity scores
retrieval_results = []
for i, query in enumerate(queries):
    query_results = [(documents[j], similarity_scores[i, j]) for j in range(len(documents))]
    ranked_results = sorted(query_results, key=lambda x: x[1], reverse=True)[:5]
    retrieval_results.append(ranked_results)

# Print retrieval results for each query
for i, query in enumerate(queries):
    print(f"Query: {query}")
    for j, (document, score) in enumerate(retrieval_results[i]):
        print(f"{j+1}. Document: {document} - Similarity Score: {score:.4f}")
    print()
