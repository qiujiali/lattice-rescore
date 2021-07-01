#!/usr/bin/env python3

import os
import sys
import argparse
import logging
from subprocess import call

import numpy as np

import lattice
from nbest_rescore import read_nbest
import utils

def get_onebest_nbest(lat_dir, weights, ref_dict, limit):
    onebest_dict = {}
    for each_utt in utils.file_iterator(lat_dir, '.nbest', resource=ref_dict):
        name, path, _, _ = each_utt
        hyps, scores = read_nbest(path)
        lens = np.array([len(hyp.split()) for hyp in hyps])
        scores = np.concatenate((scores, -np.expand_dims(lens.T, 1)), axis=1)
        assert len(weights) == scores.shape[1], 'number of weights must match'
        best_idx = np.argmax(np.dot(scores[:limit], weights))
        onebest_dict[name] = hyps[best_idx]
    return onebest_dict

def get_onebest_lattice(lat_dir, weights, ref_dict):
    onebest_dict = {}
    for each_utt in utils.file_iterator(lat_dir, '.lat.gz', resource=ref_dict):
        name, path, _, _ = each_utt
        lat = lattice.Lattice(path)
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
        best_hyp = [arc.dest.sym if arc.dest.sym not in lattice.SPECIAL_KEYS
                    else '' for arc in best_path]
        onebest_dict[name] = ' '.join(best_hyp)
    return onebest_dict

def write_trn(file_name, hyp_dict, utt_dict):
    with open(file_name, 'w') as fh:
        for key in sorted(hyp_dict.keys()):
            uttid = utt_dict[key]
            # AMI
            spk = '_'.join(uttid.split('_')[:4])
            fh.write('%s (%s_00-%s)\n' %(hyp_dict[key], spk, uttid))
            # SWBD
            # spk = uttid.rsplit('_', 1)[0].replace('-', '_')
            # fh.write('%s (%s-%s)\n' % (hyp_dict[key], spk, uttid))

def main():
    parser = argparse.ArgumentParser(
        description='Rescore nbest or lattice with given weights.')
    parser.add_argument('struct', type=str, choices=['lattice', 'nbest'],
                        help='Input data structure for rescoring.')
    parser.add_argument('latdir', type=str,
                        help='Input lattice/nbest directory.')
    parser.add_argument('ref', type=str,
                        help='Ground truth reference.')
    parser.add_argument('--weights', type=float, nargs='*',
                        help='Rescoring weights for AM, LM, RNNLM, ISCA, IP.')
    parser.add_argument('--limit', type=int, default=20,
                        help='Nbest length for rescoring.')
    parser.add_argument('--tag', type=str, default=None,
                        help='Tag for outupt file name.')
    parser.add_argument('--scoring-script', type=str,
                        default='local/score_sclite_ami.sh',
                        help='Scoring script to run, may depend on dataset.')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )
    logging.info(' '.join([sys.executable] + sys.argv))

    name_dict, ref_dict = utils.load_ref(args.ref)
    write_trn(os.path.join(args.latdir, 'ref.wrd.trn'), ref_dict, name_dict)
    if args.struct == 'lattice':
        onebest_dict = get_onebest_lattice(args.latdir, args.weights, ref_dict)
    elif args.struct == 'nbest':
        onebest_dict = get_onebest_nbest(
            args.latdir, args.weights, ref_dict, args.limit)
    else:
        raise NotImplementedError
    file_name = 'hyp_%s.wrd.trn' % args.tag if args.tag else 'hyp.wrd.trn'
    write_trn(os.path.join(args.latdir, file_name), onebest_dict, name_dict)
    sclite_cmd = '%s %s %s' % (
        args.scoring_script, os.path.dirname(args.ref), args.latdir)
    if args.tag:
        sclite_cmd += ' %s' % args.tag
    call(sclite_cmd, shell=True)

if __name__ == '__main__':
    main()
