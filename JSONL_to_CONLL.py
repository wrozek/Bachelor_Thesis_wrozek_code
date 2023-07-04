# -*- coding: utf-8 -*-
import json
from nltk.tokenize import word_tokenize

#Converts jsonl file to a conll under consideration of the labelede ner tags
def jsonl_to_conll(input_file, output_file):
    with open(input_file, 'r', encoding="utf-8") as f:
        with open(output_file, 'w', encoding="utf-8") as out:
            for line in f:
                data = json.loads(line)
                tokens = word_tokenize(data['text'])
                entities = {(e['start_offset'], e['end_offset']): e['label'] for e in data['entities']}
                print(entities)
                print(data['entities'])
                print(tokens)
                for i, token in enumerate(tokens):
                    start = sum(len(t) + 1 for t in tokens[:i]) # add 1 for space
                    end = start + len(token)
                    if (start, end) in entities:
                        out.write('{}\t{}\n'.format(token, entities[(start, end)]))
                    else:
                        out.write('{}\t{}\n'.format(token, 'O'))
                out.write('\n')

jsonl_to_conll("150all_NER_Annotated.jsonl", "150allFinalNEROutput.conll")

#Reads a CoNLL-format file and returns a list of dictionaries with 'id', 'ner_tags', and 'tokens' keys
def read_conll_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    ner_tags = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    result = []
    curr_sent_id = 0
    curr_sent_tokens = []
    curr_sent_ner_tags = []
    for line in lines:

        if line == '\n':
            # End of sentence reached, add current sentence to result
            result.append({
                'id': str(curr_sent_id),
                'ner_tags': curr_sent_ner_tags,
                'tokens': curr_sent_tokens
            })
            curr_sent_id += 1
            curr_sent_tokens = []
            curr_sent_ner_tags = []
        else:
            # Parse line and add token and NER tag to current sentence
            parts = line.strip().split('\t')
            curr_sent_tokens.append(parts[0])
            curr_sent_ner_tags.append(parts[1]) #ner_tags.index(parts[1])

    return result

# Writes a list of sentence dictionaries to a JSONL-format file.
def write_jsonl_file(sentences, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(json.dumps(sent, ensure_ascii=False) + '\n')

# Read the CoNLL-format file
sentences = read_conll_file('150allFinalNEROutput.conll')

# Write the sentence dictionaries to a JSONL-format file
write_jsonl_file(sentences, 'FinalNER150.jsonl')