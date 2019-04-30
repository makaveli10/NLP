import ujson
import codecs
import random


def load_dataset(filename):
    with codecs.open(filename, mode='r', encoding='utf-8') as f:
        dataset = ujson.load(f)
    return dataset


def pad_seq(sequences, pad_token=None, max_len=None):
    if pad_token is None:
        pad_token = 0
    if max_len is None:
        max_len = max([len(seq) for seq in sequences])
    seq_padded, seq_length = [], []
    for seq in sequences:
        _seq = seq[:max] + [pad_token] * max(max_len - len(seq), 0)
        seq_padded.append(_seq)
        seq_length.append(min(len(_seq), max_len))
    return seq_padded, seq_length


def pad_char_sequences(sequences, max_length=None, max_length_2=None):
    seq_padded, seq_length =[], []
    if max_length is None:
        max_length = max(map(lambda x: len(x), sequences))

    if max_length_2 is None:
        max_length_2 = max([max(map(lambda x: len(x), seq)) for seq in sequences])

    for seq in sequences:
        sp, sl = pad_seq(seq, max_len=max_length_2)
        seq_padded.append(sp)
        seq_length.append(sl)
    seq_padded, _ = pad_seq(seq_padded, pad_token=[0] * max_length_2, max_len=max_length)
    seq_length, _ = pad_seq(seq_length, max_len=max_length)
    return seq_padded, seq_length


def pre_process_batch(batch_words, batch_chars, batch_tags=None):
    b_words, b_words_len = pad_seq(batch_words)
    b_chars, b_chars_len = pad_char_sequences(batch_chars)
    if batch_tags is None:
        return {"words": b_words, "chars": b_chars, "seq_len": b_words_len, "char_seq_len": b_chars_len,
                "batch_size": len(b_words)}
    else:
        b_tags, _ = pad_seq(batch_tags)
        return {"words": b_words, "chars": b_chars, "tags": b_tags, "seq_len": b_words_len,
                "char_seq_len": b_chars_len, "batch_size": len(b_words)}


def dataset_batch_iter(dataset, batch_size):
    batch_words, batch_chars, batch_tags = [], [], []
    for rec in dataset:
        batch_words.append(rec["words"])
        batch_chars.append(rec["chars"])
        batch_tags.append(rec["tags"])
        if batch_size == len(batch_words):
            yield pre_process_batch(batch_words, batch_chars, batch_tags)
            batch_words, batch_chars, batch_tags = [], [], []

    if len(batch_words) > 0:
        yield pre_process_batch(batch_words, batch_chars, batch_tags)


def batchnize_dataset(data, batch_size=None, shuffle=True):
    if type(data) == str:
        dataset = load_dataset(data)
    else:
        dataset = data
    if shuffle:
        random.shuffle(dataset)
    batches = []
    if batch_size is None:
        for batch in dataset_batch_iter(dataset, len(dataset)):
            batches.append(batch)
        return batches[0]
    else:
        for batch in dataset_batch_iter(dataset, batch_size):
            batches.append(batch)
        return batches


def align_data(data):
    """Given dict with lists, creates aligned strings
    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]
    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                             data_align["y"] = "O O    O  "
    """
    spacings = [max([len(seq[i]) for seq in data.values()]) for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()
    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ''
        for token, spacing in zip(seq, spacings):
            str_aligned += token + ' ' * (spacing - len(token) + 1)
        data_aligned[key] = str_aligned
    return data_aligned