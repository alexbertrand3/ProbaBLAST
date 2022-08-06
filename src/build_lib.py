import numpy as np
import json


'''
Outputs a JSON library of all the words from the library sequence with length "w" and a probability above "p"

Some potentially useful stats:
    Sequence Length: 604466 base-pairs
    Average confidence: 0.927
    Conf standard deviation: 0.1346
    Minimum confidence: 0.39
'''
def build_library(sequence, confidence, w, p=0.8):
    lib = {}
    for i in range(0, len(sequence) - w + 1):
        word = sequence[i : i+w]
        conf = np.prod(
            confidence[i : i+w]
        )
        if conf > p:
            add_to_word_list(lib, word, i)
    return lib


def save_library(lib, file_path, indent=0):
    with open(file_path, 'w') as f:
        json.dump(lib, f, indent=indent)


def load_library(file_path):
    with open(file_path) as f:
        return json.load(f)


def add_to_word_list(lib, word, entry):
    if word in lib:
        lib[word].append(entry)
    else:
        lib[word] = [entry]

