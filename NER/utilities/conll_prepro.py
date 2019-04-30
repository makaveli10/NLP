import os
import codecs
import numpy as np
from tqdm import tqdm
from collections import Counter
from utilities.common_utils import write_json, PAD, UNK, NUM, word_conversion


glove_sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}


def raw_dataset_iter(filename, task_name, keep_number, lowercase):
    with codecs.open(filename, mode='r', encoding='utf-8') as f:
        words, tags = [], []
        for entry in f:
            line = entry.lstrip().rsplit()
            if len(line) == 0 or line.startswith('-DOCSTART-'):
                if len(words) != 0:
                    yield words, tags
                    words, tags = [], []
                else:
                    word, pos, chunk, ner = line.split(" ")
                    if task_name == "chunk":
                        tag = chunk
                    elif task_name == "ner":
                        tag = ner
                    else:
                        tag = pos
                    words.append(word)
                    tags.append(tag)


def load_dataset(filename,task_name, keep_number=False, lowercase=True):
    dataset = []
    for words, tags in raw_dataset_iter(filename, task_name, keep_number, lowercase):
        dataset.append({"words": words, "tags": tags})
    return dataset


def load_glove_vocab(glove_path, glove_name):
    vocab = set()
    total = glove_sizes[glove_name]
    with codecs.open(glove_path, mode='r', encoding='utf-8') as f:
        for line in tqdm(f, total=total, desc="Loading glove vocab"):
            line = line.lstrip().rstrip().split(" ")
            vocab.add(line[0])
    return vocab


def build_word_vocabulary(datasets):
    word_counter = Counter()
    for dataset in datasets:
        for record in dataset:
            words = record["words"]
            for word in words:
                word_counter[word] += 1
    word_vocab = [PAD, UNK, NUM] + [word for word, _ in word_counter.most_common(10000) if word != NUM]
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    return word_dict


def build_tag_vocabulary(datasets, task_name):
    tag_counter = Counter()
    for dataset in datasets:
        for record in dataset:
            tags = record["tags"]
            for tag in tags:
                tag_counter[tag] += 1
        if task_name == "ner":
            tag_vocab = [tag for tag, _ in tag_counter.most_common()]
            tag_dict = dict([(task, idx) for idx, task in enumerate(tag_vocab)])
        else:
            tag_vocab = [PAD] + [tag for tag, _ in tag_counter.most_common()]
            tag_dict = dict([(task, idx) for idx, task in enumerate(tag_vocab)])
    return tag_dict


def build_char_vocab(datasets):
    char_counter = Counter()
    for dataset in datasets:
        for record in dataset:
            for word in record["words"]:
                for char in word:
                    char_counter[char] += 1
    char_vocab = [PAD, UNK] + [char for char in char_counter.most_common()]
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    return char_dict

