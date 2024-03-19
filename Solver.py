import os, random, sys, matplotlib.pyplot as plt
import torch, torch.nn as nn, numpy
from nltk.tokenize import word_tokenize


#Read/load pre-trained embeddings
def read_file(file):
    embeddings_dict = {}

    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embed = numpy.asarray(line[1:], "float")
            embeddings_dict[word] = embed