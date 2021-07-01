#!/usr/bin/env python3

import os
import logging

import jiwer
import editdistance
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.asr.pytorch_backend.asr_init import load_trained_model
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
import lattice
import torch

def text_processing(acronyms_path=None):
    if acronyms_path:
        acronyms = load_acronyms(acronyms_path)
    else:
        acronyms = {}
    transform = jiwer.Compose([
        jiwer.RemoveKaldiNonWords(),
        jiwer.SubstituteWords(acronyms),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    return transform

def compute_word_error(hyp, ref, transform=None):
    """Compute the number of word error of a hypothesis for a reference.

    :param hyp: Hypothesis word sequence.
    :type hyp: str
    :param ref: Reference word sequence.
    :type ref: str
    :param transform: Transform hyp by removing non words and contractions.
    :type transform: jiwer.Composed object
    :return: Word errors, reference lengths, stripped hypothesis.
    :rtype: (int, int, str)
    """
    if not hyp:
        word_error = len(ref.split())
    else:
        if transform:
            hyp_transformed = transform(hyp)
        else:
            hyp_transformed = hyp
        word_error = editdistance.eval(
            hyp_transformed.upper().split(), ref.upper().split())
    return word_error, len(ref.split()), hyp

def tokenize(hyps, dictionary, sp=None, level='word', acronyms={}):
    """Tokenize hypotheses from strings into ints based on dictionary.
    
    :param hyps: Hypotheses list to tokenize.
    :type hyps: list (of strings)
    :param dictionary: Mapping from word/subword string to index.
    :type dictionary: dict
    :param sp: Sentencepiece model.
    :type sp: loaded sentencepiece model
    :param level: Tokenization level, including 'word', 'bpe', 'char'.
    :type level: string
    :param acronyms: Mapping for acronyms.
    :type acronyms: dict
    :return: List of token indices, sos and eos are added.
    :rtype: list
    """
    token_ids = []
    for hyp in hyps:
        # map acronyms before tokenizing
        if acronyms:
            words = hyp.split()
            for idx, word in enumerate(words):
                if word in acronyms:
                    words[idx] = acronyms[word]
            hyp = ' '.join(words)

        tokens = [lattice.SOS]
        if level == 'word':
            tokens += hyp.split()
        elif level == 'bpe':
            assert sp is not None, 'sentencepiece model must be provided'
            tokens += sp.encode_as_pieces(hyp)
        elif level == 'char':
            tokens += list(hyp)
        else:
            raise ValueError('level has to be one of [word, bpe, char]')
        tokens += [lattice.EOS]
        token_id = [sym2idx(dictionary, token) for token in tokens]
        token_ids.append(token_id)
    return token_ids

def sym2idx(dictionary, sym):
    """Convert symbol string to index in dictionary.

    :param dictionary: Mapping from string to int.
    :type dictionary: dict
    :param sym: Word string.
    :type sym: str
    :return: Index in dictionary.
    :rtype: int
    """
    mapping = {lattice.SOS: '<eos>', lattice.EOS: '<eos>', ' ': '<space>'}
    if sym in mapping:
        # map start & end of sentence
        sym_id = dictionary[mapping[sym]]
    else:
        # convert to index, if oov, convert and print
        try:
            sym_id = dictionary[sym]
        except KeyError:
            sym_id = dictionary['<unk>']
    return sym_id

def load_acronyms(acronyms_path):
    """Load acronyms mapping (swbd)

    :param acronyms_path: Path to the acronyms file.
    :type acornyms_path: str
    :return: The mapping from a word to another form.
    :rtype: dict
    """
    acronyms = {}
    with open(acronyms_path, 'r') as fh:
        for line in fh:
            line = line.split('\t')
            acronyms[line[1]] = line[2]
    return acronyms

def load_espnet_rnnlm(rnnlm_path):
    """Load RNNLM.

    :param rnnlm_path: Path to RNNLM model (trained using ESPnet).
    :type rnnlm_path: str
    :return: The model itself and the mapping from word to index.
    :rtype: (torch.nn.Module, dict)
    """
    rnnlm_args = get_model_conf(rnnlm_path, None)
    dictionary = rnnlm_args.char_list_dict
    rnnlm = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(
        len(dictionary), rnnlm_args.layer, rnnlm_args.unit, rnnlm_args.embed_unit))
    torch_load(rnnlm_path, rnnlm)
    rnnlm.eval()
    return rnnlm, dictionary

def load_espnet_model(model_path):
    """Load an end-to-end model from ESPnet.

    :param model_path: Path to the model.
    :type model_path: str
    :return: The model itself, mapping from subword to index,
             and training arguments used.
    :rtype: (torch.nn.Module, dict, dict)
    """
    model, train_args = load_trained_model(model_path)
    char_dict = {v: k for k, v in enumerate(train_args.char_list)}
    model.eval()
    return model, char_dict, train_args

def load_ref(file_path):
    """Load the reference text file.

    :param file_path: Path to reference file, each line in the file
                      has the format "uttid ref".
    :type file_path: str
    :return: A mapping from unique id to utt name in ref,
             and a mapping from unique id to ref string.
    :rtype: (dict, dict)
    """
    utt2name = {}
    utt2ref = {}
    with open(file_path, 'r') as fh:
        for line in fh:
            try:
                uttid, ref = line.strip().split(' ', 1)
            except ValueError:
                uttid = line.strip()
                ref = ''
            name = uttid
            utt2name[name] = uttid
            utt2ref[name] = ref
    return utt2name, utt2ref

def file_iterator(dir_in, suffix_in, dir_out=None, suffix_out=None,
                  resource=None):
    """Iterate through two-layer directory for file with certain suffix_in.

    :param dir_in: Input directory.
    :type dir_in: str
    :param suffix_in: Suffix of the file to look for.
    :type suffix_in: str
    :param dir_out: Output director, which mirrors the input directory.
    :type dir_out: str
    :param suffix_out: Suffix of the output filename.
    :type suffix_out: str
    :param resource: A dictionary containing mapping from utterance id
                     to reference or feature.
    :type resource: dict
    :yield: utterance Id, input file, output file, ref or feature.
    :rtype: tuple
    """
    dir_in = os.path.abspath(dir_in)
    dir_out = os.path.abspath(dir_out) if dir_out else None
    for root, _, file_names in os.walk(dir_in):
        for file_name in file_names:
            if file_name.endswith(suffix_in):
                file_in = os.path.join(root, file_name)
                if dir_out:
                    sub_dir_out = root.replace(dir_in, dir_out)
                    os.makedirs(sub_dir_out, exist_ok=True)
                    file_out = os.path.join(
                        sub_dir_out, file_name.replace(suffix_in, suffix_out))
                else:
                    file_out = None
                if resource:
                    uttid = file_name.replace(suffix_in, '')
                    if uttid in resource:
                        yield uttid, file_in, file_out, resource[uttid]
                    else:
                        continue
                else:
                    yield uttid, file_in, file_out, None
