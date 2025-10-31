import numpy as np
import random
import faiss
import sqlite3
import numpy as np
import pickle
from typing import List, Tuple, Optional
import random
import sqlite3
import pickle

import spacy
import re
from typing import List, Tuple, Dict
import random

random.seed(2)
np.random.seed(2)

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
    
    def search_similar_words_batch(self, query_embeddings: np.ndarray, k: int = 5) -> List[List[Tuple[str, float]]]:
        query_embeddings = query_embeddings.astype('float32')
        scores, indices = self.index.search(query_embeddings, k)
        results = []
        for score_row, idx_row in zip(scores, indices):
            result_row = []
            for score, idx in zip(score_row, idx_row):
                if idx != -1:
                    word = self.id_to_word[idx]
                    result_row.append((word, float(score)))
            results.append(result_row)
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


class WikipediaTextExtractor:
    def __init__(self):
        model = "en_core_web_sm"
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Please install the model first: python -m spacy download {model}")
            raise
    
    def extract_key_phrases(self, text: str) -> Tuple[List[str], List[str]]:
        doc = self.nlp(text)
        
        nouns = []
        verbs = []

        for chunk in doc.noun_chunks:
            if len(chunk.text.strip()) > 2 and not self._is_common_word(chunk.text):
                cleaned_noun = self._clean_noun_phrase(chunk.text)
                if cleaned_noun:
                    nouns.append(cleaned_noun)
        
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN'] and 
                len(token.text) > 2 and 
                not token.is_stop and 
                not token.is_punct and
                not self._is_common_word(token.text)):
                
                noun_text = token.text.strip()
                if noun_text not in [n.split()[-1] for n in nouns]:
                    nouns.append(noun_text)
        
        for token in doc:
            if token.pos_ == 'VERB' and not token.is_stop:
                verb_phrase = self._get_verb_phrase(token)
                if verb_phrase and len(verb_phrase) > 2:
                    verbs.append(verb_phrase)
        
        if not verbs:
            for token in doc:
                if token.pos_ in ['VERB', 'AUX']:
                    verb_phrase = self._get_verb_phrase(token)
                    if verb_phrase:
                        verbs.append(verb_phrase)
        
        if not verbs:
            for token in doc:
                if token.pos_ in ['VERB', 'AUX'] or token.lemma_ in ['be', 'have', 'do']:
                    verb_text = token.text.strip()
                    if verb_text:
                        verbs.append(verb_text)
        
        wiki_terms = self._extract_wikipedia_terms(text)
        nouns.extend(wiki_terms)
        
        nouns = list(set([n for n in nouns if n]))
        verbs = list(set([v for v in verbs if v]))
        
        long_phrases = [n for n in nouns if len(n.split()) > 6]
        nouns = [n for n in nouns if len(n.split()) <= 6]

        return nouns, verbs, long_phrases
    
    def _get_verb_phrase(self, verb_token) -> str:
        phrase_tokens = [verb_token]
        
        for child in verb_token.children:
            if child.dep_ in ['dobj', 'prep', 'advmod', 'aux', 'neg', 'compound']:
                phrase_tokens.append(child)
        
        if len(phrase_tokens) == 1:
            return verb_token.text.strip()
        
        phrase_tokens.sort(key=lambda x: x.i)
        return ' '.join([token.text for token in phrase_tokens])
    
    def _extract_wikipedia_terms(self, text: str) -> List[str]:
        wiki_terms = []

        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            r'\b(?:19|20)\d{2}\b',
            r'\b\w+(?:-\w+)+\b',
            r'\(([^)]+)\)',
            r'"([^"]+)"',
            r'\b\w+(?:ology|ography|ometry|ism|ist|ian|tion|sion)\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if isinstance(matches[0] if matches else None, tuple):
                wiki_terms.extend([match for match in matches if isinstance(match, str)])
            else:
                wiki_terms.extend(matches)
        
        cleaned_terms = []
        for term in wiki_terms:
            if isinstance(term, tuple):
                term = term[0] if term else ""
            cleaned_term = term.strip()
            if (len(cleaned_term) > 2 and 
                not self._is_common_word(cleaned_term) and
                not cleaned_term.isdigit()):
                cleaned_terms.append(cleaned_term)
        
        return cleaned_terms
    
    def _clean_noun_phrase(self, phrase: str) -> str:
        words = phrase.strip().split()
        
        while words and words[0].lower() in ['the', 'a', 'an', 'this', 'that', 'these', 'those']:
            words.pop(0)
        
        while words and words[-1].lower() in ['of', 'in', 'on', 'at', 'to', 'for', 'with', 'by']:
            words.pop()
        
        return ' '.join(words) if words else ""
    
    def _is_common_word(self, word: str) -> bool:
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'following', 'all', 'which', 'what', 
            'how', 'when', 'where', 'why', 'this', 'that', 'these', 'those',
            'also', 'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'such', 'many', 'much', 'some', 'any', 'other', 'another', 'each',
            'every', 'most', 'more', 'less', 'few', 'several', 'various'
        }
        return word.lower().strip() in common_words

    def process_text(self, text: str):
        results = {}

        nouns, verbs, long_phrases = self.extract_key_phrases(text)

        results['original'] = text
        results['nouns'] = nouns
        results['verbs'] = verbs
        results['long_phrases'] = long_phrases

        return results

    def process_article(self, article_text: str, sentence_level: bool = True):
        if sentence_level:
            sentences = self._split_sentences(article_text)
            sentence_results = []
            all_nouns = []
            all_verbs = []
            all_long_phrases = []
            
            for sentence in sentences:
                if len(sentence.strip()) > 10:
                    result = self.process_text(sentence)
                    sentence_results.append(result)
                    all_nouns.extend(result['nouns'])
                    all_verbs.extend(result['verbs'])
                    all_long_phrases.extend(result['long_phrases'])
            
            noun_freq = self._count_frequency(all_nouns)
            verb_freq = self._count_frequency(all_verbs)
            
            return {
                'sentence_results': sentence_results,
                'summary': {
                    'top_nouns': sorted(noun_freq.items(), key=lambda x: x[1], reverse=True)[:20],
                    'top_verbs': sorted(verb_freq.items(), key=lambda x: x[1], reverse=True)[:10],
                    'long_phrases': list(set(all_long_phrases))
                }
            }
        else:
            return self.process_text(article_text)
    
    def _split_sentences(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    
    def _count_frequency(self, items: List[str]) -> Dict[str, int]:
        freq = {}
        for item in items:
            freq[item] = freq.get(item, 0) + 1
        return freq

