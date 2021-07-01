#!/usr/bin/env python3

import math
import numpy as np
import torch

from espnet.nets.pytorch_backend.transformer.mask import target_mask
import lattice
import utils

def get_nbest(lat, n, aw=1.0, lw=1.0, ip=0.0):
    """Obtain nbest hypotheses with scores from the lattice."""
    hyps, scores = [], []
    nbest_paths = lat.nbest(n, aw=aw, lw=lw, ip=0.0)
    for path in nbest_paths:
        hyp = [arc.dest.sym for arc in path]
        # sos, eos, and other tokens are stripped
        hyp = [word for word in hyp if word not in lattice.SPECIAL_KEYS]
        hyps.append(' '.join(hyp))
        score = [sum([arc.ascr for arc in path]),
                 sum([arc.lscr for arc in path])]
        scores.append(score)
    return hyps, np.array(scores)

def nbest_oracle(hyps, ref, transform=None):
    """Compute oracle error in nbest hyps."""
    err = []
    for hyp in hyps:
        err.append(utils.compute_word_error(hyp, ref, transform)[0])
    min_idx = np.argmin(err)
    return err[min_idx], hyps[min_idx]

def read_nbest(file_path):
    """Read nbest hyps and scores from a text file."""
    hyps, scores = [], []
    with open(file_path, 'r') as fh:
        for line in fh:
            line = line.strip().split('|')
            score = [float(i) for i in line[0].split()[1:]]
            scores.append(score)
            hyps.append(line[1].strip())
    return hyps, np.array(scores)

def write_nbest(hyps, scores, file_path, uttid):
    """Write nbest hyps and scores to a text file."""
    with open(file_path, 'w') as fh:
        for idx, (hyp, score) in enumerate(zip(hyps, scores)):
            string = "{}-{:<6}".format(uttid, idx)
            string += ''.join(['{:10.3f}'.format(x) for x in score])
            string += " |    {}\n".format(hyp)
            fh.write(string)

def rnnlm_rescore(hyps, model, dictionary):
    """Using a word-level RNNLM to rescore nbest hypotheses."""
    scores = []
    token_ids = utils.tokenize(hyps, dictionary, level='word')
    for idx, token_id in enumerate(token_ids):
        state, pred = None, None
        total_score = 0
        for i in token_id:
            if pred is not None:
                total_score += pred[0][i].item()
            state, pred = model.predict(state, torch.LongTensor([i]))
        scores.append(total_score)
    return np.array(scores)

def isca_rescore(hyps, feat, model, dictionary, sp,
                 model_type='las', acronyms={}):
    """Using an encoder-decoder model to rescore nbest hypotheses.
    Assuming the model is based on word piece model.
    """
    scores = []
    token_ids = utils.tokenize(
        hyps, dictionary, sp=sp, level='bpe', acronyms=acronyms)
    with torch.no_grad():
        h = model.encode(feat).unsqueeze(0)
        for hyp in token_ids:
            y = torch.LongTensor(hyp)
            if model_type == 'las':
                c_list = [
                    model.dec.zero_state(h) for _ in range(model.dec.dlayers)]
                z_list = [
                    model.dec.zero_state(h) for _ in range(model.dec.dlayers)]
                c_prev = c_list
                z_prev = z_list
                att_w_prev = None
                model.dec.att[0].reset()
                total_score = 0
                for char_idx in range(len(y) - 1):
                    att_c, att_w = model.dec.att[0](
                        h,
                        [h.size(1)],
                        model.dec.dropout_dec[0](z_prev[0]),
                        att_w_prev,
                    )
                    ey = model.dec.dropout_emb(
                        model.dec.embed(y[char_idx].unsqueeze(0)))
                    ey = torch.cat((ey, att_c), dim=1)
                    z_list, c_list = model.dec.rnn_forward(
                        ey, z_list, c_list, z_prev, c_prev)
                    input_to_layerout = model.dec.dropout_dec[-1](z_list[-1])
                    if model.dec.context_residual:
                        input_to_layerout = torch.cat(
                            (input_to_layerout, att_c), dim=-1)
                    out = model.dec.output(input_to_layerout)
                    pred = torch.nn.functional.log_softmax(out, dim=1)
                    total_score += pred[0][y[char_idx+1]].item()
                    att_w_prev = att_w[:]
                    c_prev = c_list[:]
                    z_prev = z_list[:]
                scores.append(total_score)
            elif model_type == 'tfm':
                y_in = y[:-1].unsqueeze(0)
                y_mask = target_mask(y_in, model.ignore_id)
                pred = model.decoder(y_in, y_mask, h, None)[0]
                pred = torch.nn.functional.log_softmax(pred[0], dim=1)
                scores.append(torch.sum(
                    pred[torch.arange(pred.size(0)), y[1:]]).item())
    return np.array(scores)
