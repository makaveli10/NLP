import unicodedata
import re
import os
import requests
from zipfile import ZipFile


def read_file(url, filename):
    if not os.path.exists(filename):
        session = requests.Session()
        response = session.get(url, stream=True)

        CHUNK_SIZE = 32768
        with open(filename, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    zip_f = ZipFile(filename)
    filename = zip_f.namelist()

    with zip_f.open('fra.txt') as f:
        lines = f.read()

    return lines


def unicode_to_ascii(string):
    return ''.join(
        c for c in unicodedata.normalize('NFD', string)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(string):
    string = unicode_to_ascii(string)
    string = re.sub(r'([!.?])', r' \1', string)
    string = re.sub(r'[^a-zA-Z.!?]+', r' ', string)
    string = re.sub(r'\s+', r' ', string)
    return string
