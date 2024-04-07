from langchain.text_splitter import TokenTextSplitter
import random
import json

def load_contents_from_jsonl(files):
    contents = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                content = data.get('content', '')
                contents.append(content)
    return contents

def save_abstract_chunks_to_file(docs, output_file):
    # Split Documents using TokenTextSplitter to chunks
    text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=50)
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in docs:
            chunks = text_splitter.split_text(doc)
            for chunk in chunks:
                f.write(chunk + '\n')

if __name__ == "__main__":
    file_paths = ['pubmed23n0001.jsonl', 'pubmed23n0002.jsonl', 'pubmed23n0003.jsonl']
    contents = load_contents_from_jsonl(file_paths)
    # print(contents)

    # retrieved_json_path = "BioASQ_11B_test_yesno.json"
    output_file = "abstract_chunks_pubmed.txt"

    # Load retrieved JSON documents and extract abstracts
    # docs = load_retrieved_json_docs(retrieved_json_path)

    # Save abstract chunks to file
    save_abstract_chunks_to_file(contents, output_file)

    # Load abstract chunks from file
    with open(output_file, 'r', encoding='utf-8') as f:
        abstract_chunks = f.readlines()

    # Remove newline characters
    abstract_chunks = [chunk.strip() for chunk in abstract_chunks]

    # Ensure there are enough chunks to randomly select 8
    if len(abstract_chunks) >= 2:
        random_chunks = random.sample(abstract_chunks, 2)
        #for chunk in random_chunks:
            #print(chunk, "\n")
        print(random_chunks)
    else:
        print("Not enough abstract chunks available.")
