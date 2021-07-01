#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import re
from itertools import repeat
from multiprocessing import Pool, Manager

import lattice
import utils
import nbest_rescore

def nbest_operation(from_iterator, onebest_dict, nbest_dict, oracle_dict,
                    density_dict, limits, transform=None):
    """Get onebest, nbest, oracle from `limits` number of nbest hyps."""
    uttid, file_path, _, ref = from_iterator
    hyps, scores = nbest_rescore.read_nbest(file_path)
    onebest_err, _, onebest_hyp = utils.compute_word_error(
        hyps[0], ref, transform)
    onebest_dict[uttid] = (onebest_err, onebest_hyp, scores[0])
    nbest_dict[uttid] = hyps, scores

    oracles = []
    for each_limit in limits:
        oracle_err, oracle_hyp = nbest_rescore.nbest_oracle(
            hyps[:each_limit], ref, transform)
        oracles.append((oracle_err, oracle_hyp))
    oracle_dict[uttid] = oracles

    densities = []
    start, end = [int(i) for i in re.split('_|-', uttid)[-2:]]
    duration = (end - start) / lattice.FRATE
    nwords = [len(hyp.split()) for hyp in hyps]
    for each_limit in limits:
        densities.append(
            (sum(nwords[:each_limit]) - min(len(hyps), each_limit) + 1)
            / duration)
    density_dict[uttid] = densities

def run_parallel(args):
    """Looping through all nbests in parallel and get information."""
    # get reference and iterator
    name_dict, ref_dict = utils.load_ref(args.ref)
    all_utts = utils.file_iterator(args.latdir, '.nbest', resource=ref_dict)
    transform = utils.text_processing(args.acronyms) if args.clean_hyp else None

    # run all
    manager = Manager()
    onebest_dict = manager.dict()
    nbest_dict = manager.dict()
    oracle_dict = manager.dict()
    density_dict = manager.dict()
    logging.info('Processing nbests using %d processes' % args.nproc)
    with Pool(processes=args.nproc) as pool:
        pool.starmap(
            nbest_operation,
            zip(all_utts, repeat(onebest_dict), repeat(nbest_dict),
                repeat(oracle_dict), repeat(density_dict), repeat(args.limits),
                repeat(transform))
        )

    # gather results
    total_onebest_err = sum([i[0] for i in onebest_dict.values()])
    total_oracle_err = []
    for limit in range(len(args.limits)):
        total_oracle_err.append(
            sum([i[limit][0] for i in oracle_dict.values()]))
    total_ref_words = sum(
        [len(v.split()) for k, v in ref_dict.items() if k in onebest_dict])
    if len(ref_dict) != len(onebest_dict):
        logging.warning('Some lattices are missing:')
        logging.warning((set(ref_dict.keys()) - set(onebest_dict.keys())))

    # write out to info file
    file_path = os.path.join(args.latdir, 'nbest_%s.info' % args.tag)
    logging.info('Writing nbest info to file %s' % file_path)
    with open(file_path, 'w') as fh:
        fh.write('=================Summary=================\n')
        fh.write('Number of lattices:            %d\n' % len(onebest_dict))
        fh.write('Word error rate:               %.2f\n'
                 % (total_onebest_err / total_ref_words * 100))
        for idx, limit in enumerate(args.limits):
            fh.write('Oracle error rate[{:>4}-best]:  {:.2f}\n'.format(
                limit, total_oracle_err[idx] / total_ref_words * 100))
        for idx, limit in enumerate(args.limits):
            fh.write('Avg lattice density[{:>4}-best]:{:.2f}\n'.format(
                limit,
                sum([i[idx] for i in density_dict.values()]) / len(density_dict)
            ))
        fh.write('\n================Breakdown================\n')
        for uttid in sorted(onebest_dict.keys()):
            fh.write('UTTID:                %s\n' % name_dict[uttid])
            fh.write('REF:                  %s\n' % ref_dict[uttid])
            fh.write('HYP:                  %s\n' % onebest_dict[uttid][1])
            fh.write('SCORE:                %s\n' % ''.join(
                ['{:10.3f}'.format(x) for x in onebest_dict[uttid][2]]))
            for idx, limit in enumerate(args.limits):
                fh.write('ORACLE [{:>4}-best]:   {}\n'.format(
                    limit, oracle_dict[uttid][idx][1]))
            for idx, limit in enumerate(args.limits):
                fh.write('DENSITY [{:>4}-best]:  {}\n'.format(
                    limit, density_dict[uttid][idx]))
            fh.write('-----------------------------------------\n')

def main():
    parser = argparse.ArgumentParser(
        description='Gather all nbest and compute stats.')
    parser.add_argument('latdir', type=str,
                        help='Input lattice directory.')
    parser.add_argument('ref', type=str,
                        help='Ground truth reference.')
    parser.add_argument('nproc', type=int,
                        help='Number of process to run in parallel.')
    parser.add_argument('tag', type=str,
                        help='Tag for output summary file.')
    parser.add_argument('--limits', type=int, default=[], nargs='*',
                        help='Various number of limits for nbest oracle WER.')
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

    run_parallel(args)

if __name__ == '__main__':
    main()
