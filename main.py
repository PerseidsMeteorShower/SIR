import re
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import json
import sys
from tqdm import tqdm
import torch
import random
import argparse
import asyncio
import io
import gc
import httpx
from collections import Counter
import math
from transformers import AutoTokenizer, AutoModel, set_seed

from util import *
from data import initialize_kg_system, sample_rule_instances_optimized_v2, add_new_facts, sample_rule_body_paths_optimized
from eval import update_answer, final_evaluate

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

random.seed(2)
torch.manual_seed(2)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2)
np.random.seed(2)
set_seed(2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

with open("config.json", "r", encoding="utf-8") as f:
    content = json.load(f)
    API_KEY = content['API_KEY']
    AZURE_ENDPOINT = content['AZURE_ENDPOINT']

def generate_examples(sampled_instances_all, kg_name, target_samples_num):
    sampled_instances = sampled_instances_all[:target_samples_num]
    str1 = 'Please approach the problem by reasoning step by step and clearly showing your thought process. Please output your explanation, the answer, and the symbolic reasoning path. The format of the symbolic reasoning path is:\n'
    str1 += 'Premise: entity1->relation_body1->entity2->relation_body2->entity3.\n'
    str1 += 'Conclusion: entity1->relation_head->entity3.\n'
    str1 += 'Please output the results in JSON format, for example:\n'
    str1 += '{"explanation": "textual form", "Answer": "textual form", "relation_head": "relation_head", "relation_bodies": [relation_body1, relation_body2], "entities": [entity1, entity2, entity3]}\n'
    str1 += 'In the "Answer" field, include only the word or phrase of the answer.'

    examples = []
    for sampled_instance in sampled_instances:
        if kg_name == 'iBKH':
            rule_head = sampled_instance["direct_relation"].replace("_", " ")
            rule_body = [body.replace("_", " ") for body in sampled_instance["rule_body"]]
        elif kg_name == 'DBpedia':
            rule_head = 'has a relationship of "' + sampled_instance["direct_relation"] + '" with'
            rule_body = ['has a relationship of "' + body + '" with' for body in sampled_instance["rule_body"]]
        example_question = f"""Question: What entity is it that {sampled_instance["start_entity"]} {rule_head}?
{str1}"""
        example_explaination = ''
        for j in range(len(rule_body)):
            example_explaination += f"""{sampled_instance["path"][j]} {rule_body[j]} {sampled_instance["path"][j+1]}. """
        example_answer = sampled_instance["end_entity"]
        example_full_answer = {
            "Explanation": example_explaination,
            "Answer": example_answer,
            "relation_head": sampled_instance["direct_relation"],
            "relation_bodies": sampled_instance["rule_body"],
            "entities": sampled_instance["path"]
        }
        examples.append((example_question, example_full_answer))
    return examples, sampled_instances


def general_question(data, used_instances, sampled_knowledge_all, target_samples_num):
    used_entities = []
    used_relations = []
    for instance in used_instances:
        used_entities.extend(instance['path'])
        used_relations.extend(instance['rule'])

    question = data['question']

    options = []
    informations = []
    for text in data['context']:
        options.append(text[0])
        informations.append(text[0] + ': ' + text[1])

    cop = data['answers']
    prefix = data['src']

    str2 = 'Please approach the problem by reasoning step by step and clearly showing your thought process.'
    str1 = 'Please output your explanation, the answer, and the symbolic reasoning path. The format of the symbolic reasoning path is:\n'
    str1 += 'Premise: entity1->relation_body1->entity2->relation_body2->entity3.\n'
    str1 += 'Conclusion: entity1->relation_head->entity3.\n'
    str1 += 'Please output the results in JSON format, for example:\n'
    str1 += '{"explanation": "textual form", "Answer": "textual form", "relation_head": "relation_head", "relation_bodies": [relation_body1, relation_body2], "entities": [entity1, entity2, entity3]}\n'
    str1 += 'In the "Answer" field, include only the word or phrase of the answer.'

    str4 = 'Here is some information about the question:'

    str22 = 'Please use the following entities and relations when reasoning. If none of them are suitable, you may use other terms as needed.'

    str24 = 'Entities: ' + ', '.join(used_entities) + '\n'
    str24 += 'Relations: ' + ', '.join(used_relations) + '\n'

    if len(sampled_knowledge_all) == 0:
        full_question = f"""Question: {question}
{str4}
{'\n'.join(informations)}

{str2}
{str22}
{str24}
{str1}"""
        
    else:
        knowledge_statements = []
        sampled_knowledge_all = sampled_knowledge_all[:target_samples_num]
        for knowledge in sampled_knowledge_all:
            statement = ''
            for i in range(len(knowledge['rule_body'])):
                head = knowledge['path'][i].replace("_", " ")
                tail = knowledge['path'][i+1].replace("_", " ")
                relation = knowledge['rule_body'][i].replace("_", " ")
                statement += f'{head} has a relationship of "{relation}" with {tail}. '
            knowledge_statements.append(statement)
        knowledge_text = "Here are some facts that may be relevant:\n" + "\n".join(knowledge_statements)

        full_question = f"""Question: {question}
{str4}
{'\n'.join(informations)}

{knowledge_text}

{str2}
{str22}
{str24}
{str1}"""

    return full_question, cop, prefix


def get_sentence_embeddings(sentences, model, tokenizer, device):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state
    sentence_embeddings = hidden_states[:, 0, :]
    return sentence_embeddings

embedding_cache = {}
similar_words_cache = {}

def get_embedding(sentences, model, tokenizer, device):
    try:
        if len(sentences) > 32:
            sentence_embeddings = []
            for i in range(0, len(sentences), 32):
                sentence_embeddings.append(get_sentence_embeddings(sentences[i:i+32], model, tokenizer, device))
            sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
        else:
            sentence_embeddings = get_sentence_embeddings(sentences, model, tokenizer, device)
    except Exception as e:
        print(f"Error occurred while getting embeddings: {e}")
        print('sentences:', sentences)

    return sentence_embeddings

def get_embedding_cached(sentences, model, tokenizer, device):
    cache_key = hash(tuple(sentences))
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    embeddings = get_embedding(sentences, model, tokenizer, device)
    embedding_cache[cache_key] = embeddings
    return embeddings

def get_similar_words_cached(nouns, model, tokenizer, device, k, manager):
    cache_key = (tuple(sorted(nouns)), k)
    if cache_key in similar_words_cache:
        return similar_words_cache[cache_key]
    
    result = get_similar_words(nouns, model, tokenizer, device, k, manager)
    similar_words_cache[cache_key] = result
    return result

def deduplicate_nouns(nouns_list):
    word_score_dict = {}
    for word, score in nouns_list:
        if word not in word_score_dict or score > word_score_dict[word]:
            word_score_dict[word] = score
    return [(word, word_score_dict[word]) for word in sorted(word_score_dict.keys())]


def get_similar_words(nouns, model, tokenizer, device, k, manager):
    if len(nouns) == 0:
        return []
    nouns_embeddings = get_embedding_cached(nouns, model, tokenizer, device)
    nouns_embeddings = nouns_embeddings.cpu().numpy()
    
    if len(nouns_embeddings) > 1:
        batch_results = manager.search_similar_words_batch(nouns_embeddings, k)
        all_similar_nouns = []
        for result_list in batch_results:
            all_similar_nouns.extend(result_list)
    else:
        similar_nouns = manager.search_similar_words(nouns_embeddings[0], k)
        all_similar_nouns = similar_nouns
    
    deduped_nouns = deduplicate_nouns(all_similar_nouns)
    deduped_nouns = sorted(deduped_nouns, key=lambda x: x[1], reverse=True)
    
    return deduped_nouns

def get_rules(selected_heads, classed_rules):
    selected_rules = []
    for head, h_score in selected_heads:
        if head in classed_rules:
            for bodys in classed_rules[head]:
                rule_selected = [head] + list(bodys[2:])
                score = h_score + bodys[0]
                selected_rules.append((rule_selected, score))
    return selected_rules

def get_answer(content):
    try:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
            json_data = json.loads(json_str)
            if "Answer" in json_data:
                raw_answer=json_data["Answer"]
            elif "answer" in json_data:
                raw_answer=json_data["answer"]
            elif "Answers" in json_data:
                raw_answer=json_data["Answers"]
            elif "answers" in json_data:
                raw_answer=json_data["answers"]
            else:
                raw_answer=json_data
            answer_type = str(raw_answer)

            if "relation_head" in json_data:
                relation_head=json_data["relation_head"]
            elif "Relation_head" in json_data:
                relation_head=json_data["Relation_head"]
            elif "relation head" in json_data:
                relation_head=json_data["relation head"]
            elif "Relation head" in json_data:
                relation_head=json_data["Relation head"]
            else:
                relation_head = None

            if "relation_bodies" in json_data:
                relation_bodies=json_data["relation_bodies"]
            elif "Relation_bodies" in json_data:
                relation_bodies=json_data["Relation_bodies"]
            elif "relation bodies" in json_data:
                relation_bodies=json_data["relation bodies"]
            elif "Relation bodies" in json_data:
                relation_bodies=json_data["Relation bodies"]
            else:
                relation_bodies = None

            if "entities" in json_data:
                entities=json_data["entities"]
            elif "Entities" in json_data:
                entities=json_data["Entities"]
            else:
                entities = None

            entities = check_list(entities, 'entities')
            relation_bodies = check_list(relation_bodies, 'relation bodies')

            if (entities and len(entities) != 3):
                print(f"Expected 3 entities but not: {entities}")
                entities = None
            if (relation_bodies and len(relation_bodies) != 2):
                print(f"Expected 2 relation bodies but not: {relation_bodies}")
                relation_bodies = None

            if relation_head and relation_bodies and entities:
                return json_data, answer_type, str(relation_head), relation_bodies, entities
            else:
                print("JSON inference path incomplete or incorrect format.")
                print("json_data:", json_data)
                return json_data, answer_type, None, None, None

        else:
            print("JSON not found.")
            print("content:", content)
            return None, None, None, None, None
    except Exception as e:
        print("error:", e)
        print("content:", content)
        return None, None, None, None, None



async def deepseek_async(full_question, examples, prompt_path, response_path):
    max_retries = 100
    retry_delay = 10
    filtered_bool = False

    messages = []
    for example in examples:
        user_message = {"role": "user", "content": example[0]}
        assistant_message = {"role": "assistant", "content": json.dumps(example[1])}
        messages.append(user_message)
        messages.append(assistant_message)
    messages.append({"role": "user", "content": full_question})

    for i in range(max_retries):
        try:
            api_key = API_KEY
            endpoint = AZURE_ENDPOINT
            headers = {
                "Content-Type": "application/json",
                "api-key": api_key
            }
            payload = {
                "messages": messages,
                "temperature": 0.0
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(endpoint, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                contents = result["choices"][0]["message"]["content"]
            pass
        except Exception as e:
            error_message = str(e)
            if "Error code: 400" in error_message and "content_filter" in error_message:
                contents = 'filtered'
                print('filtered')
                filtered_bool = True
                break
            elif "400" in error_message:
                contents = 'filtered 400'
                print('filtered 400')
                filtered_bool = True
                break
            else:
                print(f"error:", error_message)
                if i < max_retries - 1:
                    print(f"Retrying after {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    print(f"Attempting retry #{i+2}...")
                else:
                    print("Maximum retry limit reached, exit program.")
                    contents = error_message
                    filtered_bool = True
                    break
        else:
            break
    return contents, filtered_bool, messages


def add_instance(all_ins, new_ins):
    if len(new_ins) > 0:
        new_ins = sorted(new_ins, key=lambda x: x['score'], reverse=True)
        all_ins.extend(new_ins)
    return all_ins

async def process_question_async(data, counter, extractor, model, tokenizer, device, manager_nouns, manager_verbs, 
                               kg_db, cached_index, classed_rules, top_k_similar, target_samples_num, prompt_path, response_path, kg_name):
    question = data['question']
    context = data['context']
    nouns_q, verbs_q, long_q = extractor.extract_key_phrases(question)

    if len(verbs_q) == 0 or len(nouns_q) == 0:
        print(f"Warning: Fail to extract words from question {counter}: {question}.")
        return None

    nouns_q = sorted(list(set(nouns_q)))
    verbs_q = sorted(list(set(verbs_q)))

    similar_nouns_score_q = get_similar_words(nouns_q, model, tokenizer, device, top_k_similar, manager_nouns)
    similar_verbs_score_q = get_similar_words(verbs_q, model, tokenizer, device, top_k_similar, manager_verbs)

    if len(similar_verbs_score_q) == 0 or len(similar_nouns_score_q) == 0:
        print(f"Warning: Fail to find similar words for question {counter}: {question}.")
        return None

    sampled_instances_all = []
    sampled_knowledge_all = []

    selected_rules_q = get_rules(similar_verbs_score_q, classed_rules)
    sampled_instances_q = sample_rule_instances_optimized_v2(kg_db, cached_index, selected_rules_q, similar_nouns_score_q, target_samples_num)
    sampled_knowledge_q = sample_rule_body_paths_optimized(kg_db, cached_index, selected_rules_q, similar_nouns_score_q, target_samples_num)

    sampled_instances_all = add_instance(sampled_instances_all, sampled_instances_q)
    sampled_knowledge_all = add_instance(sampled_knowledge_all, sampled_knowledge_q)

    if len(sampled_instances_all) < target_samples_num:
        nouns, verbs = [], []
        for text in context:
            nouns_text, verbs_text, long_text = extractor.extract_key_phrases(text[1])
            nouns.extend(nouns_text)
            verbs.extend(verbs_text)
        if len(nouns) != 0:
            nouns = sorted(list(set(nouns)))
            nouns = [noun for noun in nouns if noun not in nouns_q]
            similar_nouns_score = get_similar_words(nouns, model, tokenizer, device, top_k_similar, manager_nouns)
            if len(similar_nouns_score) > 0:
                sampled_instances_noun = sample_rule_instances_optimized_v2(kg_db, cached_index, selected_rules_q, similar_nouns_score, target_samples_num)
                sampled_instances_all = add_instance(sampled_instances_all, sampled_instances_noun)
                if len(sampled_knowledge_all) < target_samples_num:
                    sampled_knowledge_noun = sample_rule_body_paths_optimized(kg_db, cached_index, selected_rules_q, similar_nouns_score, target_samples_num)
                    sampled_knowledge_all = add_instance(sampled_knowledge_all, sampled_knowledge_noun)



    if len(sampled_instances_all) < target_samples_num and len(verbs) != 0:
        verbs = sorted(list(set(verbs)))
        verbs = [verb for verb in verbs if verb not in verbs_q]
        similar_verbs_score = get_similar_words(verbs, model, tokenizer, device, top_k_similar, manager_verbs)
        if len(similar_verbs_score) > 0:
            selected_rules = get_rules(similar_verbs_score, classed_rules)
            sampled_instances_verb = sample_rule_instances_optimized_v2(kg_db, cached_index, selected_rules, similar_nouns_score_q, target_samples_num)
            sampled_instances_all = add_instance(sampled_instances_all, sampled_instances_verb)
            if len(sampled_knowledge_all) < target_samples_num:
                sampled_knowledge_verb = sample_rule_body_paths_optimized(kg_db, cached_index, selected_rules, similar_nouns_score_q, target_samples_num)
                sampled_knowledge_all = add_instance(sampled_knowledge_all, sampled_knowledge_verb)


    if len(sampled_instances_all) != 0:
        paths_in_instances = set()
        for instance in sampled_instances_all:
            paths_in_instances.update(tuple(instance['path']))
        sampled_knowledge_all = [knowledge for knowledge in sampled_knowledge_all if (tuple(knowledge['path'])) not in paths_in_instances]
        examples, used_instances = generate_examples(sampled_instances_all, kg_name, target_samples_num)
        full_question, cop, prefix = general_question(data, used_instances, sampled_knowledge_all, target_samples_num)
        contents, filtered_bool, messages = await deepseek_async(full_question, examples, prompt_path, response_path)
        
        return {
            'contents': contents,
            'filtered_bool': filtered_bool,
            'cop': cop,
            'used_instances': used_instances,
            'counter': counter,
            'messages': messages,
            'prefix': prefix
        }
    
    return None


def secondary_processing(json_contents, used_instances):
    used_entities = []
    used_relations = []
    for instance in used_instances:
        used_entities.extend(instance['path'])
        used_relations.extend(instance['rule'])

    str1 = 'Please extract a syllogistic reasoning path from the following text and summarize it in a "premise-conclusion" format. First, extract the premise and conclusion in textual form, then abstract them into a knowledge graph format as follows:\n'
    str1 += 'Premise: entity1->relation_body1->entity2->relation_body2->entity3.\n'
    str1 += 'Conclusion: entity1->relation_head->entity3.\n'
    str1 += 'Please output the results in JSON format, for example:\n'
    str1 += '{"premise": "textual form", "conclusion": "textual form", "relation_head": "relation_head", "relation_bodies": [relation_body1, relation_body2], "entities": [entity1, entity2, entity3]}\n'
    str1 += 'Make sure that your answer includes exactly one relation_head, two relation_bodies, and three entities.\n'

    str2 = 'Please use the following entities and relations when constructing the knowledge graph. If none of them are suitable, you may use other terms.\n'

    str4 = 'Entities: ' + ', '.join(used_entities) + '\n'
    str4 += 'Relations: ' + ', '.join(used_relations) + '\n'

    total_prompt = f"""{str1}
Text: {str(json_contents)}
{str2}
{str4}
"""
    return total_prompt


def check_list(entities, type_name):
    if entities and not isinstance(entities, list) and ',' in str(entities):
        try:
            entities = json.loads(entities)
            if not isinstance(entities, list):
                entities = f"[{entities}]"
                entities = json.loads(entities)
                if not isinstance(entities, list):
                    print(f"Still cannot converting {type_name} to list: {e}")
                    print(f"{type_name}:", entities)
                    entities = None
        except Exception as e:
            print(f"Error converting {type_name} to list: {e}")
            print(f"{type_name}:", entities)
            entities = None

    if entities and isinstance(entities, list):
        for i in range(len(entities)):
            entity = entities[i]
            if isinstance(entity, str):
                if (entity.startswith("'") and entity.endswith("'")) or (entity.startswith('"') and entity.endswith('"')):
                    entities[i] = entity[1:-1]
            else:
                print(f"Not str entity found in: {entities}")
                entities = None
                break

    return entities

def is_list_of_strs(lst):
    return all(isinstance(i, str) for i in lst)

def process_llm_result(query_counter, result, save_path1, save_path3, classed_rules, 
                      correct_counter, incorrect_counter, none_counter, 
                      pending_counter, save_path5, save_path6, save_path7, 
                      save_path8, kg_db, cached_index, manager_verbs, manager_nouns, model, tokenizer, device,
                      rhead_exist_counter, add_rule_counter, add_rule_embedding_counter, add_entity_embedding_counter, add_instance_counter, update_rule_counter, rule_exist_counter,
                      rhead_exist_counter_path, add_rule_counter_path, add_rule_embedding_counter_path, add_entity_embedding_counter_path, add_instance_counter_path, update_rule_counter_path, rule_exist_counter_path,
                      save_path4, metrics, prefix_counts, verify_rule_counter, verify_rule_counter_path):

    contents = result['contents']
    filtered_bool = result['filtered_bool']
    cop = result['cop']
    used_instances = result['used_instances']
    task_counter = result['counter']
    messages = result['messages']
    prefix = result['prefix']

    if filtered_bool:
        print("Content filtered")
        correctness_record = (task_counter, query_counter, 9.0, 9.0, 9.0, 9.0)
        with open(save_path3, 'a', encoding='utf-8') as f:
            f.write(f"{correctness_record}\n")
    else:
        with open(save_path1, 'a', encoding='utf-8') as f:
            f.write(str(contents) + '\n')
        with open(save_path6, 'a', encoding='utf-8') as f:
            f.write(str(messages) + '\n')
        
        json_contents, answer_type, relation_head, relation_bodies, entities = get_answer(contents)
        if answer_type == None:
            correctness_record = (task_counter, query_counter, 8.0, 8.0, 8.0, 8.0)
        else:
            best_em, best_f1, best_prec, best_recall = update_answer(metrics, answer_type, cop, prefix)
            prefix_counts[prefix] += 1
            correctness_record = (task_counter, query_counter, best_em, best_f1, best_prec, best_recall)

            if best_em == 1.0:
                for used_instance in used_instances:
                    used_body1, used_body2 = used_instance['rule_body']
                    direct_relation = used_instance['direct_relation']
                    for rule in classed_rules[direct_relation]:
                        score, records, body1, body2 = rule
                        if body1 == used_body1 and body2 == used_body2:
                            rule[1][0] = rule[1][0] + 1
                            with open(save_path5, 'a', encoding='utf-8') as f:
                                f.write(f"[{direct_relation}, {rule}]\n")
                            if rule[0] < 50 and rule[0] < 1.0:
                                print(f"Rule {rule} updated, count: {rule[1]}")
                                rule[0] = rule[1][1] + 0.1 * math.log(1 + rule[1][0])
                                update_rule_counter += 1
                                with open(update_rule_counter_path, 'a', encoding='utf-8') as f:
                                    f.write(f"({task_counter}, {query_counter}, {update_rule_counter})\n")
                            elif rule[0] == 50 and rule[1][0] >= 1:
                                print(f"Rule {rule} verified, count: {rule[1]}")
                                rule[0] = 0.8
                                rule[1] = [0.0, 0.8]
                                verify_rule_counter += 1
                                with open(verify_rule_counter_path, 'a', encoding='utf-8') as f:
                                    f.write(f"({task_counter}, {query_counter}, {verify_rule_counter})\n{rule}\n")
                            break

                if relation_head:    
                    new_rule = [50.0, [0.0, 50.0], *[b.replace('_', ' ') for b in relation_bodies]]
                    new_body1, new_body2 = relation_bodies
                    if relation_head in classed_rules:
                        rhead_exist_counter += 1
                        with open(rhead_exist_counter_path, 'a', encoding='utf-8') as f:
                            f.write(f"({task_counter}, {query_counter}, {rhead_exist_counter})\n")
                        new_bool = True
                        for rule in classed_rules[relation_head]:
                            score, records, body1, body2 = rule
                            if body1 == new_body1 and body2 == new_body2 and rule[0] < 1.0:
                                rule[0] = rule[1][1] + 0.1 * math.log(10)
                                print(f"Rule {rule} updated by same reasoning path, score: {rule[0]}")
                                if rule[0] > 50:
                                    print(f"Rule {rule} verified, count: {rule[1]}")
                                    rule[0] = 0.8
                                    rule[1] = [0.0, 0.8]
                                    verify_rule_counter += 1
                                    with open(verify_rule_counter_path, 'a', encoding='utf-8') as f:
                                        f.write(f"({task_counter}, {query_counter}, {verify_rule_counter})\n{rule}\n")
                                with open(save_path5, 'a', encoding='utf-8') as f:
                                    f.write(f"[secondary_exist, {relation_head}, {rule}]\n")
                                rule_exist_counter += 1
                                with open(rule_exist_counter_path, 'a', encoding='utf-8') as f:
                                    f.write(f"({task_counter}, {query_counter}, {rule_exist_counter})\n")
                                new_bool = False
                                break
                        if new_bool:
                            classed_rules[relation_head].append(new_rule)
                            with open(save_path5, 'a', encoding='utf-8') as f:
                                f.write(f"[secondary_new_exhead, {relation_head}, {new_rule}]\n")
                    else:
                        classed_rules[relation_head] = [new_rule]
                        add_rule_counter += 1
                        with open(add_rule_counter_path, 'a', encoding='utf-8') as f:
                            f.write(f"({task_counter}, {query_counter}, {add_rule_counter})\n")
                        with open(save_path5, 'a', encoding='utf-8') as f:
                            f.write(f"[secondary_new, {relation_head}, {new_rule}]\n")
        
                    if kg_db.has_no_fact(entities[0]):
                        new_embeddings = get_sentence_embeddings([entities[0]], model, tokenizer, device)
                        embedding_np = new_embeddings.cpu().numpy()
                        manager_nouns.add_word_embedding(entities[0], embedding_np)
                        add_entity_embedding_counter += 1
                        with open(add_entity_embedding_counter_path, 'a', encoding='utf-8') as f:
                            f.write(f"({task_counter}, {query_counter}, {add_entity_embedding_counter})\n")

                    if kg_db.has_no_fact_by_relation(relation_head):
                        new_embeddings = get_sentence_embeddings([relation_head], model, tokenizer, device)
                        embedding_np = new_embeddings.cpu().numpy()
                        manager_verbs.add_word_embedding(relation_head, embedding_np)
                        add_rule_embedding_counter += 1
                        with open(add_rule_embedding_counter_path, 'a', encoding='utf-8') as f:
                            f.write(f"({task_counter}, {query_counter}, {add_rule_embedding_counter})\n")

                    new_facts = [(entities[0], relation_bodies[0], entities[1]), (entities[1], relation_bodies[1], entities[2]), (entities[0], relation_head, entities[2])]
                    add_new_facts(kg_db, cached_index, new_facts)
                    add_instance_counter += 3
                    with open(add_instance_counter_path, 'a', encoding='utf-8') as f:
                        f.write(f"({task_counter}, {query_counter}, {add_instance_counter})\n")
        
        with open(save_path3, 'a', encoding='utf-8') as f:
            f.write(f"{correctness_record}\n")
    
    return correct_counter, incorrect_counter, none_counter, classed_rules, metrics, prefix_counts



async def main_async():
    parser = argparse.ArgumentParser(description="Train model with given parameters.")
    parser.add_argument('--model_path', type=str, default= 'sentence_transformer', help="Path to the model for semantic embedding")
    parser.add_argument('--limit', type=int, default=100000000, help="Limit number of questions to test")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for questions to be processed")
    parser.add_argument('--result_folder', type=str, default='result-async', help="output folder for results")
    parser.add_argument('--dataset', type=str, default='DBpedia', help="Knowledge graph name")
    parser.add_argument('--test_set', type=str, default='BeerQA', help="test QA set name")
    args = parser.parse_args()

    root_path = 'datasets/'
    kg_name = args.dataset
    kg_dataset_path = root_path + kg_name + '/'
    embedding_path = 'embeddings/'+ kg_name + '/'

    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    print('Successfully load model for semantic embedding')
    model.to(device)

    extractor = WikipediaTextExtractor()

    print("Initialize embedding managers...")
    manager_nouns = EmbeddingManager(embedding_path +'/entity_list_embeddings.db')
    manager_verbs = EmbeddingManager(embedding_path +'/relation_list_embeddings.db')
    print("Embedding managers initialization completed")

    print("Initialize knowledge graph system...")
    kg_db, cached_index = initialize_kg_system(kg_dataset_path, embedding_path)
    print("Knowledge graph system initialization completed")

    print("Loading rules...")
    classed_rules = {}
    with open(kg_dataset_path + 'valid_rules_csrl.txt', 'r', encoding = "utf-8") as f:
        for line in f:
            if len(line.strip().split('\t')) == 2:
                score, rule = line.strip().split('\t')
                head = rule.split(' <-- ')[0].replace('_', ' ')
                body = rule.split(' <-- ')[1].split(', ')
                if head not in classed_rules:
                    classed_rules[head] = []
                classed_rules[head].append([float(score), [0.0, float(score)], *[b.replace('_', ' ') for b in body]])

    for head in classed_rules:
        classed_rules[head].sort(key=lambda x: x[0], reverse=True)
    print(f"Loaded {len(classed_rules)} rules")

    test_set_path = root_path + 'testsets/' + args.test_set + '.json'
    result_folder = args.result_folder + '/'
    save_path1 = result_folder + 'deepseek_answer-async.txt'
    save_path2 = result_folder + 'answer_statistics-async.txt'
    save_path3 = result_folder + 'correctness-async.txt'
    save_path4 = 'classed_rules_new-async.txt'
    save_path5 = result_folder + 'classed_rules_process-async.txt'
    prompt_path = result_folder + 'prompt-async.txt'
    response_path = result_folder + 'response-async.txt'
    save_path6 = result_folder + 'deepseek_messages-async.txt'
    save_path7 = result_folder + 'deepseek_messages_2-async.txt'
    save_path8 = result_folder + 'deepseek_messages_2_textual_form-async.txt'
    rhead_exist_counter_path = result_folder + 'counter_rhead_exist-async.txt'
    add_rule_counter_path = result_folder + 'counter_add_rule-async.txt'
    add_rule_embedding_counter_path = result_folder + 'counter_add_rule_embedding-async.txt'
    add_entity_embedding_counter_path = result_folder + 'counter_add_entity_embedding-async.txt'
    add_instance_counter_path = result_folder + 'counter_add_instance-async.txt'
    update_rule_counter_path = result_folder + 'counter_update_rule-async.txt'
    rule_exist_counter_path = result_folder + 'counter_rule_exist-async.txt'
    verify_rule_counter_path = result_folder + 'counter_verify_rule-async.txt'

    counter = 0
    none_counter = 0
    correct_counter = 0
    incorrect_counter = 0
    query_counter = 0
    answer_statistics = {}
    rhead_exist_counter = 0
    add_rule_counter = 0
    add_rule_embedding_counter = 0
    add_entity_embedding_counter = 0
    add_instance_counter = 0
    update_rule_counter = 0
    rule_exist_counter = 0
    save_counter = 0
    verify_rule_counter = 0

    metrics = Counter()
    prefix_counts = Counter()

    top_k_similar = 10
    target_samples_num = 5
    query_limit = limit = args.limit
    batch_size = args.batch_size

    print("test_set_path:", test_set_path)
    print(f"Processing limit: {limit}, batch_size: {batch_size}")
    print(f"top_k_similar: {top_k_similar}, target_samples_num: {target_samples_num}")

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    with open(test_set_path, 'r', encoding='utf-8') as f_set:
        pending_task = None
        pending_counter = 0
        
        for line in tqdm(f_set, desc="progress", total=limit):
            if counter >= limit or query_counter > query_limit:
                break
            
            data = json.loads(line.rstrip('\n'))
            counter += 1
            current_task = asyncio.create_task(
                process_question_async(data, counter, extractor, model, tokenizer, device, 
                                     manager_nouns, manager_verbs, kg_db, cached_index, 
                                     classed_rules, top_k_similar, target_samples_num, prompt_path, response_path, kg_name)
            )
            
            if pending_task is not None:
                result = await pending_task
                if result is not None:
                    query_counter += 1
                    correct_counter, incorrect_counter, none_counter, classed_rules, metrics, prefix_counts = process_llm_result(
                        query_counter, result, save_path1, save_path3, classed_rules, 
                        correct_counter, incorrect_counter, none_counter, counter-1, 
                        save_path5, save_path6, save_path7, save_path8, kg_db, cached_index, manager_verbs, manager_nouns, model, tokenizer, device,
                        rhead_exist_counter, add_rule_counter, add_rule_embedding_counter, add_entity_embedding_counter, add_instance_counter, update_rule_counter, rule_exist_counter,
                        rhead_exist_counter_path, add_rule_counter_path, add_rule_embedding_counter_path, add_entity_embedding_counter_path, add_instance_counter_path, update_rule_counter_path, rule_exist_counter_path,
                        save_path4, metrics, prefix_counts, verify_rule_counter, verify_rule_counter_path
                    )
                    
                    if counter % 100 == 0:
                        embedding_cache.clear()
                        similar_words_cache.clear()
                        gc.collect()
                        
                    if counter - save_counter >= 500:
                        save_counter = counter
                        with open(result_folder + str(counter) + save_path4, 'w', encoding='utf-8') as f:
                            for classed_rules_head in sorted(classed_rules.keys()):
                                classed_rules_rules = classed_rules[classed_rules_head]
                                for classed_rules_rule in classed_rules_rules:
                                    classed_rules_score, classed_rules_count, classed_rules_body1, classed_rules_body2 = classed_rules_rule
                                    rule_line = f"{classed_rules_score}, {classed_rules_count}\t{classed_rules_head} <-- {classed_rules_body1}, {classed_rules_body2}\n"
                                    f.write(rule_line)

            pending_task = current_task
            pending_counter = counter
        
        if pending_task is not None:
            result = await pending_task
            
            if result is not None:
                query_counter += 1
                correct_counter, incorrect_counter, none_counter, classed_rules, metrics, prefix_counts = process_llm_result(
                    query_counter, result, save_path1, save_path3, classed_rules, 
                    correct_counter, incorrect_counter, none_counter, counter-1, 
                    save_path5, save_path6, save_path7, save_path8, kg_db, cached_index, manager_verbs, manager_nouns, model, tokenizer, device,
                    rhead_exist_counter, add_rule_counter, add_rule_embedding_counter, add_entity_embedding_counter, add_instance_counter, update_rule_counter, rule_exist_counter,
                    rhead_exist_counter_path, add_rule_counter_path, add_rule_embedding_counter_path, add_entity_embedding_counter_path, add_instance_counter_path, update_rule_counter_path, rule_exist_counter_path,
                    save_path4, metrics, prefix_counts, verify_rule_counter, verify_rule_counter_path
                )

    with open(result_folder + str(counter) + save_path4, 'w', encoding='utf-8') as f:
        for classed_rules_head in sorted(classed_rules.keys()):
            classed_rules_rules = classed_rules[classed_rules_head]
            for classed_rules_rule in classed_rules_rules:
                classed_rules_score, classed_rules_count, classed_rules_body1, classed_rules_body2 = classed_rules_rule
                rule_line = f"{classed_rules_score}, {classed_rules_count}\t{classed_rules_head} <-- {classed_rules_body1}, {classed_rules_body2}\n"
                f.write(rule_line)

    total_num = sum(prefix_counts.values())
    metrics_dict = final_evaluate(metrics, prefix_counts, total_num)
    with open(save_path2, 'w', encoding='utf-8') as f:
        for key, value in metrics_dict.items():
            f.write(f"{key}: {value}\n")
    print('save:', save_path2)

    manager_nouns.close()
    manager_verbs.close()
    kg_db.close()
    print("finished.")


if __name__ == "__main__":
    asyncio.run(main_async())
