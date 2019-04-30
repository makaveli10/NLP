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
                    word = word_conversion(word, keep_number=keep_number, lowercase=lowercase)
                    words.append(word)
                    tags.append(tag)


def load_dataset(filename, task_name, keep_number=False, lowercase=True):
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


def build_pretrained_word_vocab(datasets, glove_vocab):
    w_counter = Counter()
    for dataset in datasets:
        for record in dataset:
            words = record["word"]
            for word in words:
                w_counter[word] += 1
    word_vocab = [word for word, _ in w_counter.most_common() if word != NUM]
    word_vocab = [PAD, UNK, NUM] + [list(set(word_vocab) & glove_vocab)]
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    return word_dict


def filter_glove_emb(word_dict, glove_path, glove_name, dim):
    vectors = np.zeros([len(word_dict), dim])
    embeddings = np.zeros([len(word_dict), dim])
    scale = np.sqrt(3 / dim)
    embeddings[1:3] = np.random.uniform(-scale, scale, [2, dim])
    with codecs.open(glove_path, mode='r', encoding='utf-8') as f:
        for line in tqdm(f, total=glove_sizes[glove_name], desc="FIlter flove embeddings.."):
            word = line[0]
            vector = [float(x) for x in line[1:]]
            if word in word_dict:
                word_idx = word_dict[word]
                vectors[word_idx] = vector
    return vectors


def build_dataset(data, word_dict, char_dict, tag_dict):
    dataset = []
    for record in data:
        words = []
        chars_list = []
        for word in record["words"]:
            chars = [char_dict[char] if char in char_dict else char_dict[UNK] for char in word]
            chars_list.append(chars)
            word = word_conversion(word, keep_num=False, lowercase=True)
            words.append(word_dict[word] if word in word_dict else word_dict[UNK])
        tags = [tag_dict[tag] for tag in record["tags"]]
        dataset.append({"words": words, "chars": chars_list, "tags": tags})
    return dataset


def preprocess(config):
    train_data = load_dataset(os.path.join(config["raw_data"], "train.txt"), config["task_name"])
    dev_data = load_dataset(os.path.join(config["raw_data"], "valid.txt"), config["task_name"])
    test_data = load_dataset(os.path.join(config["raw_data"], "test.txt"), config["task_name"])
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])
    # build vocab
    if not config["use_pretrained"]:
        word_dict = build_word_vocabulary([train_data, dev_data, test_data])
    else:
        glove_path = config["glove_path"].format(config["glove_name"], config["emb_dim"])
        glove_vocab = load_glove_vocab(glove_path, config["glove_name"])
        word_dict = build_pretrained_word_vocab([train_data, dev_data, test_data], glove_vocab)
        vectors = filter_glove_emb(word_dict,glove_path, config["glove_name"], config["emb_dim"])
        np.savez_compressed(config["pretrained_embeddings"], embeddings=vectors)
    tag_dict = build_tag_vocabulary([train_data, dev_data, test_data], config["task_name"])
    # build char_dict
    train_data = load_dataset(os.path.join(config["raw_path"], "train.txt"), config["task_name"], keep_number=True,
                              lowercase=config["char_lowercase"])
    dev_data = load_dataset(os.path.join(config["raw_path"], "valid.txt"), config["task_name"], keep_number=True,
                            lowercase=config["char_lowercase"])
    test_data = load_dataset(os.path.join(config["raw_path"], "test.txt"), config["task_name"], keep_number=True,
                             lowercase=config["char_lowercase"])
    char_dict = build_char_vocab([train_data, dev_data, test_data])
    # tokenize dataset
    train_set = build_dataset(train_data, word_dict, char_dict, tag_dict)
    dev_set = build_dataset(dev_data, word_dict, char_dict, tag_dict)
    test_set = build_dataset(test_data, word_dict, char_dict, tag_dict)
    vocab = {"word_dict": word_dict, "char_dict": char_dict, "tag_dict": tag_dict}
    # write to file in json format
    write_json(os.path.join(config["save_path"], "train.json"), train_set)
    write_json(os.path.join(config["save_path"], "vocab.json"), vocab)
    write_json(os.path.join(config["save_path"], "dev.json"), dev_set)
    write_json(os.path.join(config["save_path"], "test.json"), test_set)
