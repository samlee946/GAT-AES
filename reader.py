# From https://github.com/CarryCKW/AES-NPCR/blob/7a89a331be1f81871764b3eb4257f6732cabe9fb/aes-neural-pairwise-contrastive-regression/reader.py
# -*- coding: utf-8 -*-

import sys
import nltk
# import logging
import re
import numpy as np
import pickle as pk
import torch

url_replacer = '<url>'
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'

MAX_SENTLEN = 50
MAX_SENTNUM = 100

asap_ranges = {
    0: (-60, 60),
    1: (-10, 10),
    2: (-5, 5),
    3: (-3, 3),
    4: (-3, 3),
    5: (-4, 4),
    6: (-4, 4),
    7: (-30, 30),
    8: (-60, 60)
}


def get_ref_dtype():
    return ref_scores_dtype


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens


def get_score_range(prompt_id):
    return asap_ranges[prompt_id]


def get_model_friendly_scores(scores_array, prompt_id_array):
    arg_type = type(prompt_id_array)
    assert arg_type in {int, np.ndarray}
    if arg_type is int:
        low, high = asap_ranges[prompt_id_array]
        scores_array = (scores_array - low) / (high - low)
    # else:
    #     assert scores_array.shape[0] == prompt_id_array.shape[0]
    #     dim = scores_array.shape[0]
    #     low = np.zeros(dim)
    #     high = np.zeros(dim)
    #     for ii in range(dim):
    #         low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
    #     scores_array = (scores_array - low) / (high - low)
    # assert np.all(scores_array >= 0) and np.all(scores_array <= 1)
    return scores_array


def convert_to_dataset_friendly_scores(scores_array, prompt_id_array):
    arg_type = type(prompt_id_array)
    assert arg_type in {int, np.ndarray}
    if arg_type is int:
        low, high = asap_ranges[prompt_id_array]
        scores_array = scores_array * (high - low) + low
        assert np.all(scores_array >= low) and np.all(scores_array <= high)
    else:
        assert scores_array.shape[0] == prompt_id_array.shape[0]
        dim = scores_array.shape[0]
        low = np.zeros(dim)
        high = np.zeros(dim)
        for ii in range(dim):
            low[ii], high[ii] = asap_ranges[prompt_id_array[ii]]
        scores_array = scores_array * (high - low) + low
    return scores_array


def is_number(token):
    return bool(num_regex.match(token))


def read_essays(file_path, prompt_id):
    logger.info('Reading tsv from: ' + file_path)
    essays_list = []
    essays_ids = []
    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        next(input_file)
        for line in input_file:
            tokens = line.strip().split('\t')
            if int(tokens[1]) == prompt_id or prompt_id <= 0:
                essays_list.append(tokens[2].strip())
                essays_ids.append(int(tokens[0]))
    return essays_list, essays_ids


def replace_url(text):
    replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', url_replacer, text)
    return replaced_text


def text_tokenizer(text, replace_url_flag=True, tokenize_sent_flag=True, create_vocab_flag=False):
    text = replace_url(text)
    text = text.replace(u'"', u'')
    if "..." in text:
        text = re.sub(r'\.{3,}(\s+\.{3,})*', '...', text)
        # print text
    if "??" in text:
        text = re.sub(r'\?{2,}(\s+\?{2,})*', '?', text)
        # print text
    if "!!" in text:
        text = re.sub(r'\!{2,}(\s+\!{2,})*', '!', text)
        # print text

    tokens = tokenize(text)
    if tokenize_sent_flag:
        punctuation = '.!,;:?"\'、，；'
        text = " ".join(tokens)

        # text = text.replace('.', ' [SEP] ')
        # text = text.replace('!', ' [SEP] ')
        # text = text.replace('?', ' [SEP] ')

        text_nopun = re.sub(r'[{}]+'.format(punctuation), '', text)
        sent_tokens = text_nopun
        # sent_tokens = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)
        # sent_tokens = tokenize_to_sentences(text, MAX_SENTLEN, create_vocab_flag)
        # print sent_tokens
        # sys.exit(0)
        # if not create_vocab_flag:
        #     print "After processed and tokenized, sentence num = %s " % len(sent_tokens)
        return sent_tokens
    else:
        raise NotImplementedError
