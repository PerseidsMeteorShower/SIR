import json
import pandas as pd
import glob
import os
import argparse
import random
import sqlite3
import pickle
from collections import defaultdict
import sys

random.seed(2)

class KGDatabase:
    def __init__(self, db_path='kg_database.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_database()
        self._cache = {}
        
    def _init_database(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                head TEXT NOT NULL,
                relation TEXT NOT NULL,
                tail TEXT NOT NULL,
                UNIQUE(head, relation, tail)
            )
        ''')
        
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_head ON facts (head)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_relation ON facts (relation)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_head_relation ON facts (head, relation)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_tail ON facts (tail)')
        self.conn.commit()
    
    def load_from_dataframe(self, df_all):
        df_all.to_sql('facts', self.conn, if_exists='replace', index=False)
        self._init_database()
        self._clear_cache()
    
    def add_facts(self, facts_list):
        self.conn.executemany(
            'INSERT OR IGNORE INTO facts (head, relation, tail) VALUES (?, ?, ?)',
            facts_list
        )
        self.conn.commit()
        self._clear_cache()

    def add_facts_zero(self, facts_list):
        status_list = []
        for head, relation, tail in facts_list:
            cursor = self.conn.execute(
                'SELECT 1 FROM facts WHERE head=? AND relation=? AND tail=?',
                (head, relation, tail)
            )
            exists = cursor.fetchone()
            if exists:
                status_list.append(1)
            else:
                self.conn.execute(
                    'INSERT INTO facts (head, relation, tail) VALUES (?, ?, ?)',
                    (head, relation, tail)
                )
                status_list.append(0)
        self.conn.commit()
        self._clear_cache()
        return status_list
    
    def check_facts_zero(self, facts_list):
        status_list = []
        for head, relation, tail in facts_list:
            cursor = self.conn.execute(
                'SELECT 1 FROM facts WHERE head=? AND relation=? AND tail=?',
                (head, relation, tail)
            )
            exists = cursor.fetchone()
            if exists:
                status_list.append(1)
            else:
                status_list.append(0)
        return status_list

    
    def batch_query_heads(self, head_list, relation=None):
        placeholders = ','.join(['?' for _ in head_list])
        if relation:
            query = f'SELECT head, relation, tail FROM facts WHERE head IN ({placeholders}) AND relation = ?'
            params = head_list + [relation]
        else:
            query = f'SELECT head, relation, tail FROM facts WHERE head IN ({placeholders})'
            params = head_list
        
        cursor = self.conn.execute(query, params)
        return cursor.fetchall()
    
    def get_tails_by_head_relation(self, head, relation):
        cache_key = (head, relation)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        cursor = self.conn.execute(
            'SELECT tail FROM facts WHERE head = ? AND relation = ?',
            (head, relation)
        )
        result = [row[0] for row in cursor.fetchall()]
        self._cache[cache_key] = result
        return result
    
    def batch_check_multiple_facts(self, head_list):
        placeholders = ','.join(['?' for _ in head_list])
        query = f'''
            SELECT head, COUNT(*) as fact_count 
            FROM facts 
            WHERE head IN ({placeholders}) 
            GROUP BY head
            HAVING COUNT(*) > 1
        '''
        cursor = self.conn.execute(query, head_list)
        return [row[0] for row in cursor.fetchall()]
    
    def has_no_fact(self, head):
        cursor = self.conn.execute(
            'SELECT COUNT(*) FROM facts WHERE head = ?',
            (head,)
        )
        count = cursor.fetchone()[0]
        return count <= 0
    
    def batch_check_single_facts_by_relations(self, relation_list):
        placeholders = ','.join(['?' for _ in relation_list])
        query = f'''
            SELECT relation, COUNT(*) as fact_count 
            FROM facts 
            WHERE relation IN ({placeholders}) 
            GROUP BY relation
        '''
        cursor = self.conn.execute(query, relation_list)
        results = cursor.fetchall()
        
        result_dict = {}
        for relation in relation_list:
            result_dict[relation] = 0
        
        for relation, count in results:
            result_dict[relation] = count
        
        return [relation for relation, count in result_dict.items() if count <= 1]
    
    def has_no_fact_by_relation(self, relation):
        cursor = self.conn.execute(
            'SELECT COUNT(*) FROM facts WHERE relation = ?',
            (relation,)
        )
        count = cursor.fetchone()[0]
        return count <= 0
  

    def _clear_cache(self):
        self._cache.clear()
    
    def close(self):
        self.conn.close()


class CachedKGIndex:
    def __init__(self, kg_db, cache_file='kg_index_cache.pkl'):
        self.kg_db = kg_db
        self.cache_file = cache_file
        self.head_relation_index = self._load_or_build_index()
        
    def _load_or_build_index(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    index = pickle.load(f)
                return index
            except:
                pass
        
        return self._build_index()
    
    def _build_index(self):
        index = defaultdict(list)
        
        cursor = self.kg_db.conn.execute('SELECT head, relation, tail FROM facts')
        for head, relation, tail in cursor:
            index[(head, relation)].append(tail)
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(dict(index), f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return dict(index)
    
    def update_index(self, new_facts):
        for head, relation, tail in new_facts:
            key = (head, relation)
            if key not in self.head_relation_index:
                self.head_relation_index[key] = []
            if tail not in self.head_relation_index[key]:
                self.head_relation_index[key].append(tail)
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.head_relation_index, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def get_tails(self, head, relation):
        return self.head_relation_index.get((head, relation), [])
    
    def batch_get_tails(self, head_relation_pairs):
        result = {}
        for head, relation in head_relation_pairs:
            result[(head, relation)] = self.get_tails(head, relation)
        return result



def load_kg(dataset_path):
    relation_paths, relation_names = [],[]
    file_paths1 = glob.glob(os.path.join(dataset_path, 'relation', '*'))
    file_paths1.sort(key=str.lower)
    for file in file_paths1:
        file_new1 = os.path.basename(file)
        file_new2 = os.path.splitext(file_new1)[0]
        relation_paths.append(file_new1)
        relation_names.append(file_new2)

    return relation_paths, relation_names



def load_large_facts_to_db(facts_file, kg_db, chunksize=100000):
    print(f"Read {facts_file} in chunks and write it to the database...")
    
    kg_db.conn.execute('DELETE FROM facts')
    kg_db.conn.commit()
    
    total_rows = 0
    try:
        for i, chunk in enumerate(pd.read_csv(facts_file, sep='\t', header=None, names=['head', 'relation', 'tail'], chunksize=chunksize)):
            chunk = chunk.dropna().drop_duplicates()
            if not chunk.empty:
                kg_db.add_facts(list(chunk.itertuples(index=False, name=None)))
                total_rows += len(chunk)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1} chunks, total {total_rows} records")

    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    kg_db._init_database()
    kg_db._clear_cache()
    print(f"Large file loading completed! A total of {total_rows} records loaded.")
    return True

def initialize_kg_system(dataset_path, embedding_path = ''):
    kg_db = KGDatabase(embedding_path + 'kg_database.db')
    facts_file = os.path.join(dataset_path, 'facts.txt')
    
    if not os.path.exists(facts_file):
        print(f"Error: File not found {facts_file}")
        return kg_db, CachedKGIndex(kg_db, embedding_path + 'kg_index_cache.pkl')
    
    file_size = os.path.getsize(facts_file)
    need_reload = False

    if need_reload:
        
        if file_size > 50*1024*1024:
            print("Large file detected, use chunk loading mode")
            success = load_large_facts_to_db(facts_file, kg_db)
            if not success:
                print("Chunk loading failed, trying regular loading")
                try:
                    df_all = pd.read_csv(facts_file, sep='\t', header=None, names=['head', 'relation', 'tail'])
                    df_all = df_all.dropna().drop_duplicates()
                    kg_db.load_from_dataframe(df_all)
                    print(f"Regular loading completed, a total of {len(df_all)} records loaded.")
                except Exception as e:
                    print(f"Regular loading also failed: {e}")
        else:
            print("Using regular loading mode")
            try:
                df_all = pd.read_csv(facts_file, sep='\t', header=None, names=['head', 'relation', 'tail'])
                df_all = df_all.dropna().drop_duplicates()
                kg_db.load_from_dataframe(df_all)
                print(f"Regular loading completed, a total of {len(df_all)} records loaded.")
            except Exception as e:
                print(f"Loading failed: {e}")
    else:
        print("Using existing database")

    cursor = kg_db.conn.execute('SELECT COUNT(*) FROM facts')
    final_count = cursor.fetchone()[0]
    print(f"Final database record count: {final_count}")

    cached_index = CachedKGIndex(kg_db, embedding_path + 'kg_index_cache.pkl')
    return kg_db, cached_index


def sample_rule_instances_ultra_optimized(kg_db, cached_index, selected_rules, selected_as, target_samples=50):
    total_sampled_instances = []
    num_rules = len(selected_rules)
    
    if num_rules == 0 or len(selected_as) == 0:
        return total_sampled_instances

    normalized_as = selected_as
    if target_samples > num_rules:
        allocated_samples = -(-target_samples // num_rules)
    else:
        allocated_samples = 1
    
    for rule, r_score in selected_rules:      
        sampled_instances = []
        rel1 = rule[0]
        body_relations = rule[1:]
        
        valid_as = []
        for A, a_score in normalized_as:
            N_candidates = cached_index.get_tails(A, rel1)
            if N_candidates:
                valid_as.append((A, a_score, N_candidates))
        
        for A, a_score, N_candidates in valid_as:
            if len(sampled_instances) >= allocated_samples:
                break
                
            max_n_to_check = min(5, len(N_candidates))
            n_checked = 0
            
            for N in N_candidates:
                if len(sampled_instances) >= allocated_samples or n_checked >= max_n_to_check:
                    break
                n_checked += 1
                
                if verify_path_exists_optimized(A, N, body_relations, cached_index):
                    path = construct_path_optimized(A, N, body_relations, cached_index)
                    if path:
                        instance = {
                            'start_entity': A,
                            'end_entity': N,
                            'direct_relation': rel1,
                            'rule_body': body_relations,
                            'path': path,
                            'rule': rule,
                            'score': a_score + r_score
                        }
                        sampled_instances.append(instance)
                        break
        total_sampled_instances.extend(sampled_instances)
    
    return total_sampled_instances


def sample_rule_body_paths_optimized(kg_db, cached_index, selected_rules, selected_as, target_samples=50, max_paths_per_rule=10):
    total_sampled_instances = []
    num_rules = len(selected_rules)
    
    if num_rules == 0 or len(selected_as) == 0:
        return total_sampled_instances

    normalized_as = selected_as
    
    if target_samples > num_rules:
        allocated_samples = -(-target_samples // num_rules)
    else:
        allocated_samples = 1
    
    for rule_idx, (rule, r_score) in enumerate(selected_rules):
        sampled_instances = []
        body_relations = rule[1:]
        
        if len(body_relations) == 0:
            continue
        
        for A, a_score in normalized_as:
            if len(sampled_instances) >= allocated_samples:
                break
            
            paths_found = find_all_rule_body_paths(A, body_relations, cached_index, max_paths_per_rule)
            
            for path_info in paths_found:
                if len(sampled_instances) >= allocated_samples:
                    break
                
                end_entity = path_info['end_entity']
                path = path_info['path']
                
                instance = {
                    'start_entity': A,
                    'end_entity': end_entity,
                    'rule_body': body_relations,
                    'path': path,
                    'rule': rule,
                    'score': a_score + r_score,
                    'path_length': len(path) - 1,
                    'relations_used': body_relations
                }
                sampled_instances.append(instance)
                
        total_sampled_instances.extend(sampled_instances)
    
    return total_sampled_instances


def find_all_rule_body_paths(start_entity, body_relations, cached_index, max_paths=10):
    if len(body_relations) == 0:
        return [{'end_entity': start_entity, 'path': [start_entity]}]
    
    queue = [(start_entity, 0, [start_entity])]
    all_paths = []
    
    while queue and len(all_paths) < max_paths:
        current_entity, used_relations, current_path = queue.pop(0)
        
        if used_relations == len(body_relations):
            all_paths.append({
                'end_entity': current_entity,
                'path': current_path
            })
            continue
        
        if used_relations < len(body_relations):
            next_relation = body_relations[used_relations]
            next_entities = cached_index.get_tails(current_entity, next_relation)
            
            for next_entity in next_entities:
                new_path = current_path + [next_entity]
                queue.append((next_entity, used_relations + 1, new_path))
    
    return all_paths


def find_all_rule_body_paths_dfs(start_entity, body_relations, cached_index, max_paths=10):
    if len(body_relations) == 0:
        return [{'end_entity': start_entity, 'path': [start_entity]}]
    
    all_paths = []
    
    def dfs(current_entity, relation_index, current_path):
        if len(all_paths) >= max_paths:
            return
        
        if relation_index == len(body_relations):
            all_paths.append({
                'end_entity': current_entity,
                'path': current_path.copy()
            })
            return
        
        relation = body_relations[relation_index]
        next_entities = cached_index.get_tails(current_entity, relation)
        
        for next_entity in next_entities:
            current_path.append(next_entity)
            dfs(next_entity, relation_index + 1, current_path)
            current_path.pop()
    
    dfs(start_entity, 0, [start_entity])
    return all_paths



def sample_rule_body_paths_only(kg_db, cached_index, selected_rules, selected_as, target_samples=50, max_end_entities_per_start=10):
    total_sampled_instances = []
    num_rules = len(selected_rules)
    
    if num_rules == 0 or len(selected_as) == 0:
        return total_sampled_instances

    normalized_as = selected_as
    
    if target_samples > num_rules:
        allocated_samples = -(-target_samples // num_rules)
    else:
        allocated_samples = 1
    
    for rule_idx, (rule, r_score) in enumerate(selected_rules):
        sampled_instances = []
        body_relations = rule[1:]
        
        if len(body_relations) == 0:
            continue
        
        for A, a_score in normalized_as:
            if len(sampled_instances) >= allocated_samples:
                break
            
            end_entities_found = find_rule_body_end_entities(A, body_relations, cached_index, max_end_entities_per_start)
            
            paths_from_current_A = 0
            for N in end_entities_found:
                if len(sampled_instances) >= allocated_samples:
                    break
                    
                path = construct_rule_body_path(A, N, body_relations, cached_index)
                if path:
                    instance = {
                        'start_entity': A,
                        'end_entity': N,
                        'rule_body': body_relations,
                        'path': path,
                        'rule': rule,
                        'score': a_score + r_score,
                        'path_length': len(path) - 1,
                        'relations_used': body_relations
                    }
                    sampled_instances.append(instance)
                    paths_from_current_A += 1
        total_sampled_instances.extend(sampled_instances)
    
    return total_sampled_instances


def find_rule_body_end_entities(start_entity, body_relations, cached_index, max_entities=10):
    if len(body_relations) == 0:
        return [start_entity]
    
    current_entities = {start_entity}
    
    for rel in body_relations:
        next_entities = set()
        for entity in current_entities:
            tails = cached_index.get_tails(entity, rel)
            next_entities.update(tails)
        
        if not next_entities:
            return []
        current_entities = next_entities
    
    result = list(current_entities)
    if len(result) > max_entities:
        result = result[:max_entities]
    
    return result


def construct_rule_body_path(start_entity, end_entity, body_relations, cached_index):
    if len(body_relations) == 0:
        return [start_entity] if start_entity == end_entity else None
    
    def dfs_find_path(current, target, remaining_rels, path):
        if len(remaining_rels) == 0:
            return path if current == target else None
        
        rel = remaining_rels[0]
        next_entities = cached_index.get_tails(current, rel)
        
        for next_entity in next_entities:
            result = dfs_find_path(next_entity, target, remaining_rels[1:], path + [next_entity])
            if result:
                return result
        return None
    
    return dfs_find_path(start_entity, end_entity, body_relations, [start_entity])



def verify_path_exists_optimized(start, end, relations, cached_index):
    if len(relations) == 0:
        return start == end
    current_entities = {start}
    
    for rel in relations:
        next_entities = set()
        for entity in current_entities:
            tails = cached_index.get_tails(entity, rel)
            next_entities.update(tails)
        
        if not next_entities:
            return False
        current_entities = next_entities
        
        if end in current_entities and rel == relations[-1]:
            return True
    
    return end in current_entities


def construct_path_optimized(start, end, relations, cached_index):
    if len(relations) == 0:
        return [start] if start == end else None
    
    def dfs_path(current, target, remaining_rels, path):
        if len(remaining_rels) == 0:
            return path if current == target else None
        
        rel = remaining_rels[0]
        next_entities = cached_index.get_tails(current, rel)
        for next_entity in next_entities:
            result = dfs_path(next_entity, target, remaining_rels[1:], path + [next_entity])
            if result:
                return result
        return None
    
    return dfs_path(start, end, relations, [start])


def add_new_facts(kg_db, cached_index, new_facts):
    kg_db.add_facts(new_facts)
    cached_index.update_index(new_facts)


def add_new_facts_zero(kg_db, cached_index, new_facts):
    status_list = kg_db.add_facts_zero(new_facts)
    cached_index.update_index(new_facts)
    return status_list


def batch_query_selected_as(kg_db, selected_as, relation=None):
    entity_list = []
    for item in selected_as:
        if isinstance(item, tuple):
            entity_list.append(item[0])
        else:
            entity_list.append(item)
    
    return kg_db.batch_query_heads(entity_list, relation)




def explore_kg_data(kg_db, cached_index, selected_as):
    print("=== Data exploration ===")
    
    normalized_as = []
    for item in selected_as:
        if isinstance(item, tuple):
            normalized_as.append(item[0])
        else:
            normalized_as.append(item)
    
    for entity in normalized_as[:3]:
        print(f"\nAll outgoing edges of entity '{entity}':")
        
        cursor = kg_db.conn.execute(
            'SELECT DISTINCT relation FROM facts WHERE head = ?',
            (entity,)
        )
        relations = [row[0] for row in cursor.fetchall()]
        print(f"Available relations: {relations}")

        for rel in relations[:5]:
            tails = cached_index.get_tails(entity, rel)
            print(f"  {entity} --{rel}--> {tails[:3] if tails else '无'}")
    
    target_relations = ['has', 'has_ingredient']
    for rel in target_relations:
        cursor = kg_db.conn.execute(
            'SELECT COUNT(*) FROM facts WHERE relation = ?',
            (rel,)
        )
        count = cursor.fetchone()[0]
        print(f"\nRelation '{rel}' has {count} records in the database")

        cursor = kg_db.conn.execute(
            'SELECT head, tail FROM facts WHERE relation = ? LIMIT 5',
            (rel,)
        )
        examples = cursor.fetchall()
        for head, tail in examples:
            print(f"  {head} --{rel}--> {tail}")


def print_first_five_facts(kg_db):
    cursor = kg_db.conn.execute('SELECT COUNT(*) FROM facts')
    total = cursor.fetchone()[0]
    print(f"\nTotal records in the facts table: {total}")
    cursor = kg_db.conn.execute('SELECT head, relation, tail FROM facts LIMIT 5')
    print("Top five records in the database:")
    found = False
    for i, (head, relation, tail) in enumerate(cursor, 1):
        print(f"{i}. {head} --{relation}--> {tail}")
        found = True
    if not found:
        print("No records found.")


def verify_and_construct_path_optimized(start, end, relations, cached_index):
    if len(relations) == 0:
        if start == end:
            return True, [start]
        else:
            return False, None
    
    def dfs_find_first_path(current, target, remaining_rels, current_path):
        if len(remaining_rels) == 0:
            if current == target:
                return True, current_path
            else:
                return False, None
        
        rel = remaining_rels[0]
        next_entities = cached_index.get_tails(current, rel)
        
        for next_entity in next_entities:
            new_path = current_path + [next_entity]
            found, path = dfs_find_first_path(next_entity, target, remaining_rels[1:], new_path)
            if found:
                return True, path
        
        return False, None
    
    return dfs_find_first_path(start, end, relations, [start])




def sample_rule_instances_optimized_v2(kg_db, cached_index, selected_rules, selected_as, target_samples=50):
    total_sampled_instances = []
    num_rules = len(selected_rules)
    
    if num_rules == 0 or len(selected_as) == 0:
        return total_sampled_instances

    normalized_as = selected_as
    
    if target_samples > num_rules:
        allocated_samples = -(-target_samples // num_rules)
    else:
        allocated_samples = 1
    
    for rule, r_score in selected_rules:      
        sampled_instances = []
        rel1 = rule[0]
        body_relations = rule[1:]
        
        valid_as = []
        for A, a_score in normalized_as:
            N_candidates = cached_index.get_tails(A, rel1)
            if N_candidates:
                valid_as.append((A, a_score, N_candidates))
        
        for A, a_score, N_candidates in valid_as:
            if len(sampled_instances) >= allocated_samples:
                break
            
            for N in N_candidates:
                if len(sampled_instances) >= allocated_samples:
                    break

                path_found, path = verify_and_construct_path_optimized(A, N, body_relations, cached_index)
                
                if path_found and path:
                    instance = {
                        'start_entity': A,
                        'end_entity': N,
                        'direct_relation': rel1,
                        'rule_body': body_relations,
                        'path': path,
                        'rule': rule,
                        'score': a_score + r_score
                    }
                    sampled_instances.append(instance)
                    
                    break
                
        total_sampled_instances.extend(sampled_instances)
    
    return total_sampled_instances



def batch_verify_and_construct_paths(start_entity, candidates_with_relations, cached_index, max_samples=5):
    results = []
    
    for end_entity, body_relations in candidates_with_relations:
        if len(results) >= max_samples:
            break
            
        path_found, path = verify_and_construct_path_optimized(
            start_entity, end_entity, body_relations, cached_index
        )
        
        if path_found and path:
            results.append((end_entity, path))
    
    return results


def sample_rule_instances_batch_optimized(kg_db, cached_index, selected_rules, selected_as, target_samples=50):
    total_sampled_instances = []
    num_rules = len(selected_rules)
    
    if num_rules == 0 or len(selected_as) == 0:
        return total_sampled_instances

    normalized_as = selected_as
    
    if target_samples > num_rules:
        allocated_samples = -(-target_samples // num_rules)
    else:
        allocated_samples = 1
    
    for rule, r_score in selected_rules:      
        sampled_instances = []
        rel1 = rule[0]
        body_relations = rule[1:]
        
        valid_as = []
        for A, a_score in normalized_as:
            N_candidates = cached_index.get_tails(A, rel1)
            if N_candidates:
                valid_as.append((A, a_score, N_candidates))
        
        for A, a_score, N_candidates in valid_as:
            if len(sampled_instances) >= allocated_samples:
                break
            
            max_candidates = min(5, len(N_candidates))
            candidates_with_relations = [
                (N, body_relations) for N in N_candidates[:max_candidates]
            ]
            
            successful_paths = batch_verify_and_construct_paths(
                A, candidates_with_relations, cached_index, 
                max_samples=allocated_samples - len(sampled_instances)
            )
            
            for N, path in successful_paths:
                instance = {
                    'start_entity': A,
                    'end_entity': N,
                    'direct_relation': rel1,
                    'rule_body': body_relations,
                    'path': path,
                    'rule': rule,
                    'score': a_score + r_score
                }
                sampled_instances.append(instance)
                
                if len(sampled_instances) >= allocated_samples:
                    break
            
            if successful_paths:
                break
                
        total_sampled_instances.extend(sampled_instances)
    
    return total_sampled_instances
