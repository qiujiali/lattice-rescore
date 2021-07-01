#!/usr/bin/env python3

import sys
import argparse
import logging

import numpy as np
import cma

import lattice
import utils
from nbest_rescore import read_nbest

# CMA related
STD = 0.5
SEED = 123

def load_all_nbest(lat_dir, ref_dict, max_n, prob=1.0):
    nbest_dict = {}
    for each_utt in utils.file_iterator(lat_dir, '.nbest', resource=ref_dict):
        if np.random.uniform() < prob:
            name, path, _, _ = each_utt
            hyps, scores = read_nbest(path)
            nbest_dict[name] = (hyps[:max_n], scores[:max_n])
    return nbest_dict

def load_all_lattice(lat_dir, ref_dict, prob=1.0):
    lat_dict = {}
    for each_utt in utils.file_iterator(lat_dir, '.lat.gz', resource=ref_dict):
        if np.random.uniform() < prob:
            name, path, _, _ = each_utt
            lat_dict[name] = lattice.Lattice(path)
    return lat_dict

def compute_nbest_wer(weights, nbest_dict, ref_dict, transform=None):
    err_count = 0
    word_count = 0
    for utt in nbest_dict.keys():
        hyps, scores = nbest_dict[utt]
        lens = np.array([len(hyp.split()) for hyp in hyps])
        scores = np.concatenate((scores, -np.expand_dims(lens.T, 1)), axis=1)
        assert len(weights) == scores.shape[1], 'number of weights must match'
        new_scores = np.dot(scores, weights)
        best_idx = np.argmax(new_scores)
        err, nword, _ = utils.compute_word_error(
            hyps[best_idx], ref_dict[utt], transform)
        err_count += err
        word_count += nword
    return err_count / word_count

def compute_lattice_wer(weights, lattice_dict, ref_dict, transform=None):
    err_count = 0
    word_count = 0
    for utt in lattice_dict.keys():
        lat = lattice_dict[utt]
        n_nscr = len(lat.arcs[2].nscr)
        n_iscr = len(lat.arcs[2].iscr)
        assert len(weights) == 2 + n_nscr + n_iscr + 1, (
            'number of weights must match')
        best_path = lat.onebest(
            aw=weights[0],
            lw=weights[1],
            nw=weights[2: (2+n_nscr)],
            iw=weights[(2+n_nscr): -1],
            ip=weights[-1]
        )
        best_hyp = ' '.join([arc.dest.sym for arc in best_path
                             if arc.dest.sym not in lattice.SPECIAL_KEYS])
        err, nword, _ = utils.compute_word_error(
            best_hyp, ref_dict[utt], transform)
        err_count += err
        word_count += nword
    return err_count / word_count

def cma_optimization(objective_fn, init_weights, hyp_dict, ref_dict, bounds,
                     freeze_dim=None, transform=None):
    # freeze acoustic weight to 1
    freeze = {0: 1}
    if freeze_dim:
        for dim in freeze_dim:
            freeze[dim] = 0
    logging.info('Freeze dimension: %s' % freeze)
    es = cma.CMAEvolutionStrategy(
        init_weights,
        STD,
        {'seed': SEED, 'bounds': bounds, 'fixed_variables': freeze}
    )
    es.optimize(objective_fn, args=(hyp_dict, ref_dict, transform), verb_disp=1)
    logging.info('The optimal WER is %.2f' % (es.result.fbest * 100))
    logging.info('The optimal weight combination is %s'
                 % np.array2string(es.result.xbest, precision=3, separator=','))

def main():
    parser = argparse.ArgumentParser(
        description='Find best interpolation weights for rescoring')
    parser.add_argument('struct', type=str, choices=['lattice', 'nbest'],
                        help='Input data structure for rescoring.')
    parser.add_argument('latdir', type=str,
                        help='Input lattice directory.')
    parser.add_argument('ref', type=str,
                        help='Ground truth reference.')
    parser.add_argument('--nbest', type=int, default=20,
                        help='Maximinum number of nbest to use.')
    parser.add_argument('--subset', type=float, default=1.0,
                        help='Proportion of the data to use.')
    parser.add_argument('--freeze', type=int, default=None, nargs='*',
                        help='Leave certain scores out during rescoring, '
                             '1 is LM, etc.')
    parser.add_argument('--clean_hyp', default=False, action='store_true',
                        help='remove non words and expand contractions.')
    parser.add_argument('--acronyms', type=str, default=None,
                        help='Path to acronoym mapping (swbd)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )
    logging.info(' '.join([sys.executable] + sys.argv))

    if args.subset < 1.0:
        logging.info('%d%% of the data is used for tuning.'
                     % (args.subset * 100))
        np.random.seed(SEED)

    _, ref_dict = utils.load_ref(args.ref)
    if args.struct == 'lattice':
        hyp_dict = load_all_lattice(args.latdir, ref_dict, args.subset)
        score_dim = (2 + len(list(hyp_dict.values())[0].arcs[2].nscr)
                     + len(list(hyp_dict.values())[0].arcs[2].iscr))
        objective_fn = compute_lattice_wer
    elif args.struct == 'nbest':
        hyp_dict = load_all_nbest(
            args.latdir, ref_dict, args.nbest, args.subset)
        score_dim = list(hyp_dict.values())[0][1].shape[1]
        objective_fn = compute_nbest_wer
    else:
        raise NotImplementedError
    init = np.ones(score_dim + 1)
    if args.freeze:
        assert 0 not in args.freeze, 'acoustic score must be used'
        assert max(args.freeze) < score_dim, 'freeze dim does not exist'
    # The bound for insertion penalty is [-2, 2]
    bounds = [[0 for _ in range(score_dim)] + [-2],
              [2 for _ in range(score_dim)] + [2]]

    transform = utils.text_processing(args.acronyms) if args.clean_hyp else None

    cma_optimization(
        objective_fn, init, hyp_dict, ref_dict, bounds, args.freeze, transform)

if __name__ == '__main__':
    main()
