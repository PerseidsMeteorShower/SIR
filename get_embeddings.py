from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os
import random
from tqdm import tqdm
import argparse
import json
import faiss
import sqlite3
import numpy as np
import pickle
from typing import List, Tuple, Optional

random.seed(2)
torch.manual_seed(2)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2)
np.random.seed(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class EmbeddingManager:
    def __init__(self, db_path: str, embedding_dim: int = 768):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS word_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE,
                embedding BLOB
            )
        ''')
        
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.id_to_word = {}
        self.word_to_id = {}
        
        self._load_existing_data()
    
    def _load_existing_data(self):
        cursor = self.conn.execute('SELECT id, word, embedding FROM word_embeddings')
        embeddings = []
        
        for row_id, word, embedding_blob in cursor:
            embedding = pickle.loads(embedding_blob)
            embeddings.append(embedding)
            self.id_to_word[len(self.id_to_word)] = word
            self.word_to_id[word] = len(self.word_to_id)
        
        if embeddings:
            embeddings_array = np.vstack(embeddings).astype('float32')
            self.index.add(embeddings_array)
    
    def add_word_embedding(self, word: str, embedding: np.ndarray):
        if word in self.word_to_id:
            return
        
        embedding_blob = pickle.dumps(embedding.astype('float32'))
        self.conn.execute(
            'INSERT OR IGNORE INTO word_embeddings (word, embedding) VALUES (?, ?)',
            (word, embedding_blob)
        )
        self.conn.commit()
        
        faiss_id = len(self.id_to_word)
        self.id_to_word[faiss_id] = word
        self.word_to_id[word] = faiss_id
        self.index.add(embedding.reshape(1, -1).astype('float32'))
    
    def search_similar_words(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                word = self.id_to_word[idx]
                results.append((word, float(score)))
        
        return results
    
    def get_word_embedding(self, word: str) -> Optional[np.ndarray]:
        if word not in self.word_to_id:
            return None
        
        cursor = self.conn.execute(
            'SELECT embedding FROM word_embeddings WHERE word = ?', (word,)
        )
        row = cursor.fetchone()
        if row:
            return pickle.loads(row[0])
        return None
    
    def close(self):
        self.conn.close()


def get_sentence_embeddings(sentences, model, tokenizer, device):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state
    sentence_embeddings = hidden_states[:, 0, :]
    return sentence_embeddings


parser = argparse.ArgumentParser(description="Train model with given parameters.")
parser.add_argument('--model_path', type=str, default= 'sentence_transformer', help="Path to the model for semantic embedding")
parser.add_argument('--dataset', type=str, default= 'DBpedia', help="Knowledge graph name")
global args
args = parser.parse_args()

model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
print('Successfully load model for semantic embedding')
model.to(device)

dataset_path = 'datasets/' + args.dataset + '/'
save_dir = 'embeddings/' + args.dataset + '/'
data_name = 'relation_list'
file_path = dataset_path + data_name + '.txt'
print("file_path:", file_path)


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(file_path, 'r', encoding='utf-8') as f:
    sentences = json.load(f)

batch_size = 32
save_batch_size = 1000

first_batch = sentences[:min(batch_size, len(sentences))]
first_embeddings = get_sentence_embeddings(first_batch, model, tokenizer, device)
embedding_dim = first_embeddings.shape[1]

embedding_manager = EmbeddingManager(save_dir + data_name +'_embeddings.db', embedding_dim=embedding_dim)

for start_idx in tqdm(range(0, len(sentences), save_batch_size), desc='Processing batches', unit="batch"):
    end_idx = min(start_idx + save_batch_size, len(sentences))
    batch_sentences = sentences[start_idx:end_idx]
    
    if len(batch_sentences) > batch_size:
        batch_embeddings = []
        for i in range(0, len(batch_sentences), batch_size):
            mini_batch = batch_sentences[i:i+batch_size]
            mini_embeddings = get_sentence_embeddings(mini_batch, model, tokenizer, device)
            batch_embeddings.append(mini_embeddings)
        
        current_embeddings = torch.cat(batch_embeddings, dim=0)
    else:
        current_embeddings = get_sentence_embeddings(batch_sentences, model, tokenizer, device)
    
    for sentence, embedding in zip(batch_sentences, current_embeddings):
        embedding_np = embedding.cpu().numpy()
        embedding_manager.add_word_embedding(sentence, embedding_np)
    
    del current_embeddings
    if 'batch_embeddings' in locals():
        del batch_embeddings
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

embedding_manager.close()
print('Saved embeddings to database with FAISS index:', data_name)






