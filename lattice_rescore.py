#!/usr/bin/env python3

import math
from itertools import chain
from collections.abc import MutableMapping
from copy import deepcopy

import numpy as np
import torch

from espnet.nets.pytorch_backend.transformer.mask import target_mask
import lattice
from utils import sym2idx

class Cache(dict):
    """A customised dictionary as the cache for lattice rescoring.
    Input list of string is first concatenated and then loop up.
    The key value pair is ngram: (state, pred, post).
    """
    __slots__ = 'model', 'dict', 'type', 'sp', 'h', 'cache_locality', 'acronyms'

    @staticmethod
    def _process_args(mapping=(), **kwargs):
        if hasattr(mapping, 'items'):
            mapping = getattr(mapping, 'items')()
        return ((' '.join(k), v)
                for k, v in chain(mapping, getattr(kwargs, 'items')()))

    def __init__(self, model, dictionary, feat=None, model_type='rnnlm',
                 sp=None, cache_locality=9, acronyms={}, mapping=(), **kwargs):
        super(Cache, self).__init__(self._process_args(mapping, **kwargs))
        self.model = model
        self.dict = dictionary
        self.type = model_type
        self.sp = sp
        self.cache_locality = cache_locality
        self.acronyms = acronyms
        self.h = None
        if self.type == 'rnnlm':
            self.init_rnnlm()
        elif self.type == 'las':
            assert self.sp is not None, 'sentencepiece model required'
            self.init_las(feat)
        elif self.type == 'tfm':
            assert self.sp is not None, 'sentencepiece model required'
            self.init_tfm(feat)

    def init_rnnlm(self):
        assert hasattr(self.model, 'predict'), 'model must have predict method'
        # RNNLM state, output distribution, posterior, previou ngram
        self.__setitem__(([], 0),
                         (None, torch.zeros(len(self.dict)), [], ([], 0)))

    def init_las(self, feat):
        assert feat is not None, 'for LAS model, acoustic feature must be given'
        with torch.no_grad():
            self.h = self.model.encode(feat).unsqueeze(0)
        c_list = [self.model.dec.zero_state(self.h)
                  for _ in range(self.model.dec.dlayers)]
        z_list = [self.model.dec.zero_state(self.h)
                  for _ in range(self.model.dec.dlayers)]
        att_w = None
        self.model.dec.att[0].reset()
        self.__setitem__(([], 0),
                         ((att_w, c_list, z_list), torch.zeros(len(self.dict)),
                          [], ([], 0), {}))

    def init_tfm(self, feat):
        assert feat is not None, 'for TFM model, acoustic feature must be given'
        with torch.no_grad():
            self.h = self.model.encode(feat).unsqueeze(0)
        self.__setitem__(([], 0), (('', 0.0), None, [], ([], 0), {}))

    def get_value_by_locality(self, timed_cache, key):
        cached_keys = list(timed_cache.keys())
        distance = [abs(key - i) for i in cached_keys]
        min_idx = np.argmin(distance)
        if distance[min_idx] <= self.cache_locality:
            return timed_cache[cached_keys[min_idx]]
        else:
            raise KeyError

    def __getitem__(self, k):
        ngram, timestamp = k
        ngram_dict = super(Cache, self).__getitem__(' '.join(ngram))
        return self.get_value_by_locality(ngram_dict, timestamp)

    def __setitem__(self, k, v):
        ngram, timestamp = k
        try:
            ngram_dict = super(Cache, self).__getitem__(' '.join(ngram))
            return ngram_dict.__setitem__(timestamp, v)
        except KeyError:
            return super(Cache, self).__setitem__(
                ' '.join(ngram), {timestamp: v})

    def __delitem__(self, k):
        ngram, timestamp = k
        return super(Cache, self).__delitem__(' '.join(ngram))

    def get(self, k, default=None):
        try:
            return self.__getitem__(k)
        except KeyError:
            return default

    def setdefault(self, k, default=None):
        try:
            return self.__getitem__(k)
        except KeyError:
            return self.__setitem__(k, default)

    def pop(self, k, v=object()):
        ngram, timestamp = k
        if v is object():
            return super(Cache, self).pop(' '.join(ngram))
        return super(Cache, self).pop(' '.join(ngram), v)

    def update(self, mapping=(), **kwargs):
        super(Cache, self).update(self._process_args(mapping, **kwargs))

    def __contains__(self, k):
        ngram, timestamp = k
        contain_ngram = super(Cache, self).__contains__(' '.join(ngram))
        if not contain_ngram:
            return False
        try:
            _ = self.__getitem__(k)
            return True
        except KeyError:
            return False

    def copy(self):
        return type(self)(self, self.model, self.dict)

    @classmethod
    def fromkeys(cls, keys, v=None):
        return super(Cache, cls).fromkeys((' '.join(k) for k in keys), v)

    def __repr__(self):
        return '{0}({1})'.format(
            type(self).__name__, super(Cache, self).__repr__())

    def get_state(self, k):
        return deepcopy(self.__getitem__(k)[0])

    def get_pred(self, k, word):
        if self.type == 'rnnlm':
            return self.__getitem__(k)[1][sym2idx(self.dict, word)].item()
        elif self.type == 'las':
            with torch.no_grad():
                att_w, c_list, z_list = self.get_state(k)
                if word == lattice.SOS:
                    y = torch.LongTensor([self.model.sos])
                elif word == lattice.EOS:
                    y = torch.LongTensor([self.model.eos])
                else:
                    # This also works for the first word
                    mapped_word = self.acronyms.get(word, word)
                    hyp = self.sp.encode_as_pieces(' ' + mapped_word)
                    y = torch.LongTensor(
                        [sym2idx(self.dict, char) for char in hyp])
                if word in self.get_word_dict(k):
                    return self.get_word_dict(k)[word]
                score = self.__getitem__(k)[1][y[0]].item()
                for char_idx in range(len(y) - 1):
                    att_c, att_w = self.model.dec.att[0](
                        self.h,
                        [self.h.size(1)],
                        self.model.dec.dropout_dec[0](z_list[0]),
                        att_w,
                    )
                    ey = self.model.dec.dropout_emb(
                        self.model.dec.embed(y[char_idx].unsqueeze(0)))
                    ey = torch.cat((ey, att_c), dim=1)
                    z_list, c_list = self.model.dec.rnn_forward(
                        ey, z_list, c_list, z_list, c_list)
                    input_to_layerout = self.model.dec.dropout_dec[-1](
                        z_list[-1])
                    if self.model.dec.context_residual:
                        input_to_layerout = torch.cat(
                            (input_to_layerout, att_c), dim=-1)
                    out = self.model.dec.output(input_to_layerout)
                    pred = torch.nn.functional.log_softmax(out, dim=1)
                    score += pred[0][y[char_idx + 1]].item()
            self.get_word_dict(k)[word] = score
            return score
        elif self.type == 'tfm':
            with torch.no_grad():
                hist, hist_score = self.get_state(k)
                if word == lattice.SOS:
                    y = torch.LongTensor([self.model.sos])
                    score = 0.0
                else:
                    if word in self.get_word_dict(k):
                        return self.get_word_dict(k)[word]
                    if word == lattice.EOS:
                        hyp = self.sp.encode_as_pieces(hist)
                        y = torch.LongTensor(
                            [self.model.sos]
                            + [sym2idx(self.dict, char) for char in hyp]
                            + [self.model.eos])
                    else:
                        mapped_word = self.acronyms.get(word, word)
                        hyp = self.sp.encode_as_pieces(hist + ' ' + mapped_word)
                        y = torch.LongTensor(
                            [self.model.sos]
                            + [sym2idx(self.dict, char) for char in hyp])
                    y_in = y[:-1].unsqueeze(0)
                    y_mask = target_mask(y_in, self.model.ignore_id)
                    pred = self.model.decoder(y_in, y_mask, self.h, None)[0]
                    pred = torch.nn.functional.log_softmax(pred[0], dim=1)
                    score = torch.sum(
                        pred[torch.arange(pred.size(0)), y[1:]]).item()
            self.get_word_dict(k)[word] = score - hist_score
            return score - hist_score

    def get_post(self, k):
        return self.__getitem__(k)[2]

    def get_prev_ngram(self, k):
        return self.__getitem__(k)[3]

    def get_word_dict(self, k):
        # this is only for ISCA subword rescoring
        return self.__getitem__(k)[4]

    def get_timestamp(self, k):
        ngram, timestamp = k
        ngram_dict = super(Cache, self).__getitem__(' '.join(ngram))
        cached_keys = list(ngram_dict.keys())
        distance = [abs(timestamp - i) for i in cached_keys]
        min_idx = np.argmin(distance)
        return cached_keys[min_idx]

    def renew(self, prev_ngram, new_ngram, post):
        word = new_ngram[0][-1]
        if word in [lattice.OOV, lattice.UNK]:
            # skip oov and unk
            value = self.__getitem__(prev_ngram)
        else:
            with torch.no_grad():
                if self.type == 'rnnlm':
                    state, pred = self.model.predict(
                        self.get_state(prev_ngram),
                        torch.LongTensor([sym2idx(self.dict, word)])
                    )
                    value = (state, pred[0], post, prev_ngram)
                elif self.type == 'las':
                    att_w, c_list, z_list = self.get_state(prev_ngram)
                    if word == lattice.SOS:
                        y = torch.LongTensor([self.model.sos])
                    elif word == lattice.EOS:
                        y = torch.LongTensor([self.model.eos])
                    else:
                        mapped_word = self.acronyms.get(word, word)
                        hyp = self.sp.encode_as_pieces(mapped_word)
                        y = torch.LongTensor(
                            [sym2idx(self.dict, char) for char in hyp])
                    score = self.__getitem__(prev_ngram)[1][y[0]].item()
                    for char_idx in range(len(y)):
                        att_c, att_w = self.model.dec.att[0](
                            self.h,
                            [self.h.size(1)],
                            self.model.dec.dropout_dec[0](z_list[0]),
                            att_w,
                        )
                        ey = self.model.dec.dropout_emb(
                            self.model.dec.embed(y[char_idx].unsqueeze(0)))
                        ey = torch.cat((ey, att_c), dim=1)
                        z_list, c_list = self.model.dec.rnn_forward(
                            ey, z_list, c_list, z_list, c_list)
                        input_to_layerout = self.model.dec.dropout_dec[-1](
                            z_list[-1])
                        if self.model.dec.context_residual:
                            input_to_layerout = torch.cat(
                                (input_to_layerout, att_c), dim=-1)
                        out = self.model.dec.output(input_to_layerout)
                        pred = torch.nn.functional.log_softmax(out, dim=1)
                        if char_idx < len(y) - 1:
                            score += pred[0][y[char_idx + 1]].item()
                    self.get_word_dict(prev_ngram)[word] = score
                    state = (att_w, c_list, z_list)
                    value = (state, pred[0], post, prev_ngram, {})
                elif self.type == 'tfm':
                    hist, hist_score = self.get_state(prev_ngram)
                    if word == lattice.SOS:
                        y = torch.LongTensor([self.model.sos])
                        score = 0.0
                        state = ('', 0.0)
                    else:
                        if word == lattice.EOS:
                            hyp = self.sp.encode_as_pieces(hist)
                            y = torch.LongTensor(
                                [self.model.sos]
                                + [sym2idx(self.dict, char) for char in hyp]
                                + [self.model.eos])
                        else:
                            mapped_word = self.acronyms.get(word, word)
                            hyp = self.sp.encode_as_pieces(
                                hist + ' ' + mapped_word)
                            y = torch.LongTensor(
                                [self.model.sos]
                                + [sym2idx(self.dict, char) for char in hyp])
                        y_in = y[:-1].unsqueeze(0)
                        y_mask = target_mask(y_in, self.model.ignore_id)
                        pred = self.model.decoder(y_in, y_mask, self.h, None)[0]
                        pred = torch.nn.functional.log_softmax(pred[0], dim=1)
                        score = torch.sum(
                            pred[torch.arange(pred.size(0)), y[1:]]).item()
                        state = (hist + ' ' + mapped_word, score)
                    self.get_word_dict(prev_ngram)[word] = score - hist_score
                    value = (state, None, post, prev_ngram, {})
        self.__setitem__(new_ngram, value)

def rnnlm_rescore(lat, lm, dictionary, ngram, replace=False, cache_locality=9):
    """Lattice rescoring with RNNLM with on-the-fly lattice expansion
    using n-gram based history clustering.
    Optionally, run forward-backward before calling this function,
    so the cache can be updated based on lattice node posterior.

    :param lat: Word lattice object.
    :type lat: lattice.Lattice
    :param lm: RNNLM.
    :type lm: torch.nn.Module
    :param dictionary: Mapping from word to index.
    :type dictionary: dict
    :param ngram: Number of n-gram for history clustering.
    :type ngram: int
    :param replace: Replace existing scores if True, otherwise append.
    :type replace: bool
    :param cache_locality: Only use cache if around given number of frames.
    :type cache_locality: int
    :return: An expanded lattice with RNNLM score on each arc.
    :rtype: lattice.Lattice
    """
    # setup ngram cache
    cache = Cache(
        lm, dictionary, model_type='rnnlm', cache_locality=cache_locality)
    cache_hit = 0
    cache_miss_new = 0
    cache_miss_update = 0
    # initialise expanded node & outbound arc list
    for node in lat.nodes:
        node.expnodes = []
        node.exparcs = []
    lat.start.expnodes.append(lat.start.subnode())
    # lattice traversal
    for n_i in lat.nodes:
        for n_j in n_i.expnodes:
            for a_k in n_i.exits:
                # find the destination node n_k of arc a_k
                n_k = a_k.dest
                # find the LM state phi(h_{n_0}^{n_j}) of expanded node n_j
                phi_nj = n_j.lmstate
                # find a new LM state phi(h_{n_0}^{n_k}) for node n_k
                phi_nk = ((phi_nj[0] + [n_k.sym])[-ngram:], n_k.entry)
                try:
                    # check if the destination node needs to be expanded
                    idx = [' '.join(i.lmstate[0]) for i in n_k.expnodes].index(
                        ' '.join(phi_nk[0]))
                    n_l = n_k.expnodes[idx]
                except ValueError:
                    # create a new node for expansion
                    n_l = n_k.subnode()
                    n_l.lmstate = deepcopy(phi_nk)
                    n_k.expnodes.append(n_l)
                new_arc = a_k.subarc(n_j, n_l)

                # update cache except for the final node
                if n_k.sym != lattice.EOS:
                    phi_nk_post = (cache.get_post(phi_nj) + [a_k.post])[-ngram:]
                    # compute LM probability P(n_k|phi(h_{n_0}^{n_j}))
                    if phi_nk not in cache:
                        # create new entry in cache for unseen ngram
                        cache.renew(phi_nj, phi_nk, phi_nk_post)
                        cache_miss_new += 1
                    else:
                        timestamp_condition = (
                            abs(cache.get_timestamp(phi_nk) - n_k.entry)
                            <= cache.cache_locality)
                        # if the new ngram phi_nk in already cache
                        if (cache.get_prev_ngram(phi_nk)[0] == phi_nj
                            and timestamp_condition):
                            # if the previous ngram phi_nj is the same
                            # then cache hit, do not forward again
                            # note that same ngram can have different posterior
                            # because of different timestamps
                            cache_hit += 1
                        else:
                            # if the previous ngram phi_nj is different
                            posterior_condition = (
                                sum(phi_nk_post) > sum(cache.get_post(phi_nk)))
                            if posterior_condition or not timestamp_condition:
                                # renew the cache with higher posterior
                                cache.renew(phi_nj, phi_nk, phi_nk_post)
                                cache_miss_update += 1
                            else:
                                cache_hit += 1

                if n_k.sym in [lattice.OOV, lattice.UNK]:
                    if replace:
                        new_arc.nscr = [0.0]
                    else:
                        new_arc.nscr.append(0.0)
                else:
                    if replace:
                        new_arc.nscr = [cache.get_pred(phi_nj, n_k.sym)]
                    else:
                        new_arc.nscr.append(cache.get_pred(phi_nj, n_k.sym))
                n_j.exits.append(new_arc)
                n_l.entries.append(new_arc)
    # Rebuild lattice from expanded nodes & arcs
    new_lat = lattice.Lattice(header=lat.header, nframes=lat.nframes)
    for node in lat.nodes:
        if node != lat.end:
            for expnode in node.expnodes:
                new_lat.nodes.append(expnode)
                new_lat.arcs.extend(expnode.entries)
        else:
            # merge the final node
            end_node = node.expnodes[0]
            for expnode in node.expnodes[1:]:
                for exparc in expnode.entries:
                    exparc.dest = end_node
                    end_node.entries.append(exparc)
            new_lat.nodes.append(end_node)
            new_lat.arcs.extend(end_node.entries)
    new_lat.start = new_lat.nodes[0]
    new_lat.end = new_lat.nodes[-1]
    # print('Cache hit: %d' % cache_hit)
    # print('Cache miss new: %d' % cache_miss_new)
    # print('Cache miss update: %d' % cache_miss_update)
    # print('Cache hit rate: %.4f' % (
    #     cache_hit / (cache_hit + cache_miss_new + cache_miss_update)))
    return new_lat

def isca_rescore(lat, feat, model, char_dict, ngram, sp, model_type='las',
                 replace=False, cache_locality=9, acronyms={}):
    """Lattice rescoring with LAS model with on-the-fly lattice expansion
    using n-gram based history clustering.
    Optionally, run forward-backward before calling this function,
    so the cache can be updated based on lattice node posterior.

    :param lat: Word lattice object.
    :type lat: lattice.Lattice
    :param feat: Acoustic feature for the utterance.
    :type feat: np.ndarray
    :param model: ESPnet RNN-based LAS model.
    :type model: torch.nn.Module
    :param char_dict: Mapping from character to index.
    :type char_dict: dict
    :param ngram: Number of n-gram for history clustering.
    :type ngram: int
    :param sp: Sentencepiece model.
    :type sp: sentencepiece model object
    :param model_type: Type of the end-to-end model, one of ['las', 'tfm'].
    :type model_type: str
    :param replace: Replace existing scores if True, otherwise append.
    :type replace: bool
    :param cache_locality: Only use cache if around given number of frames.
    :type cache_locality: int
    :param acronyms: Mapping to match vocabulary.
    :type acronyms: dict
    :return: An expanded lattice with ISCA score on each arcs.
    :rtype: lattice.Lattice
    """
    # setup ngram cache
    cache = Cache(
        model, char_dict, feat=feat, model_type=model_type, sp=sp,
        cache_locality=cache_locality, acronyms=acronyms
    )
    cache_hit = 0
    cache_miss_new = 0
    cache_miss_update = 0
    # initialise expanded node & outbound arc list
    for node in lat.nodes:
        node.expnodes = []
        node.exparcs = []
    lat.start.expnodes.append(lat.start.subnode())
    # lattice traversal
    for n_i in lat.nodes:
        for n_j in n_i.expnodes:
            for a_k in n_i.exits:
                # find the destination node n_k of arc a_k
                n_k = a_k.dest
                # find the LM state phi(h_{n_0}^{n_j}) of expanded node n_j
                phi_nj = n_j.lmstate
                # find a new LM state phi(h_{n_0}^{n_k}) for node n_k
                phi_nk = ((phi_nj[0] + [n_k.sym])[-ngram:], n_k.entry)
                # check if the destination node needs to be expanded
                try:
                    idx = [' '.join(i.lmstate[0]) for i in n_k.expnodes].index(
                        ' '.join(phi_nk[0]))
                    n_l = n_k.expnodes[idx]
                except ValueError:
                    # create a new node for expansion
                    n_l = n_k.subnode()
                    n_l.lmstate = deepcopy(phi_nk)
                    n_k.expnodes.append(n_l)
                new_arc = a_k.subarc(n_j, n_l)

                # update cache except for the final node
                if n_k.sym != lattice.EOS:
                    phi_nk_post = (cache.get_post(phi_nj) + [a_k.post])[-ngram:]
                    # compute LM probability P(n_k|phi(h_{n_0}^{n_j}))
                    if phi_nk not in cache:
                        # create new entry in cache for unseen ngram
                        cache.renew(phi_nj, phi_nk, phi_nk_post)
                        cache_miss_new += 1
                    else:
                        timestamp_condition = (
                            abs(cache.get_timestamp(phi_nk) - n_k.entry)
                            <= cache.cache_locality)
                        if (cache.get_prev_ngram(phi_nk)[0] == phi_nj
                            and timestamp_condition):
                            # if the previous ngram phi_nj is the same
                            # then cache hit, do not forward again
                            # note that same ngram can have different posterior
                            # because of different timestamps
                            cache_hit += 1
                        else:
                            posterior_condition = (
                                sum(phi_nk_post) > sum(cache.get_post(phi_nk)))
                            # if the previous ngram phi_nj is different
                            if posterior_condition or not timestamp_condition:
                                # renew the cache with higher posterior
                                cache.renew(phi_nj, phi_nk, phi_nk_post)
                                cache_miss_update += 1
                            else:
                                cache_hit += 1

                if n_k.sym in [lattice.OOV, lattice.UNK]:
                    if replace:
                        new_arc.iscr = [0.0]
                    else:
                        new_arc.iscr.append(0.0)
                else:
                    if replace:
                        new_arc.iscr = cache.get_pred(phi_nj, n_k.sym)
                    else:
                        new_arc.iscr.append(cache.get_pred(phi_nj, n_k.sym))
                n_j.exits.append(new_arc)
                n_l.entries.append(new_arc)
    # Rebuild lattice from expanded nodes & arcs
    new_lat = lattice.Lattice(header=lat.header, nframes=lat.nframes)
    for node in lat.nodes:
        if node != lat.end:
            for expnode in node.expnodes:
                new_lat.nodes.append(expnode)
                new_lat.arcs.extend(expnode.entries)
        else:
            # merge the final node
            end_node = node.expnodes[0]
            for expnode in node.expnodes[1:]:
                for exparc in expnode.entries:
                    exparc.dest = end_node
                    end_node.entries.append(exparc)
            new_lat.nodes.append(end_node)
            new_lat.arcs.extend(end_node.entries)
    new_lat.start = new_lat.nodes[0]
    new_lat.end = new_lat.nodes[-1]
    # print('Cache hit: %d' % cache_hit)
    # print('Cache miss new: %d' % cache_miss_new)
    # print('Cache miss update: %d' % cache_miss_update)
    # print('Cache hit rate: %.4f' % (
    #     cache_hit / (cache_hit + cache_miss_new + cache_miss_update)))
    return new_lat
