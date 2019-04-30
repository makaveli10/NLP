import codecs
import ujson
import re
import unicodedata


PAD = "<PAD>"
UNK = "<UNK>"
NUM = "<NUM>"
END = "</S>"
SPACE = "_SPACE"


# Serialise obj to a JSON string, writing it to the given stream (ujson.dump(obj, stream))
def write_json(file, dataset):
    with codecs.open(file, mode="w", encoding="utf-8") as f:
        ujson.dump(dataset, f)


def word_conversion(word, keep_num=True, lowercase=True):
    if not keep_num:
        if is_num_digit(word):
            word = NUM
        if lowercase:
            word = word.lower()
        return word


def is_num_digit(word):
    try:
        float(word)
        return True
    except ValueError:
        pass
    try:
        # Returns the numeric value assigned to the Unicode character unichr as float.
        # If no such value is defined, default is returned, or, if not given, ValueError is raised.
        unicodedata.numeric(word)
        return True
    except (TypeError, ValueError):
        pass
    result =  re.compile(r'^[-+]?[0-9]+,[0-9]+$').match(word)
    if result:
        return True
    return False