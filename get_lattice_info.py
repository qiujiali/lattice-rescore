#!/usr/bin/env python3

import os
import sys
import logging
import argparse
from itertools import repeat
from multiprocessing import Pool, Manager
import numpy as np

import lattice
import utils

def lattice_operation(from_iterator, onebest_dict, oracle_dict, density_dict,
                      gsf=None, transform=None):
    """Get onebest, oracle, lattice density from a lattice."""
    uttid, file_path, _, ref = from_iterator
    lat = lattice.Lattice(file_path, file_type='htk')
    if gsf is None:
        gsf = float(lat.header['lmscale'])
    onebest_path = lat.onebest(aw=(1 / gsf), lw=1, nw=0, iw=0)
    onebest_hyp = ' '.join([arc.dest.sym for arc in onebest_path
                            if arc.dest.sym not in lattice.SPECIAL_KEYS])
    onebest_score = [sum([arc.ascr for arc in onebest_path]),
                     sum([arc.lscr for arc in onebest_path])]
    onebest_nscr = np.sum([np.array(arc.nscr) for arc in onebest_path], axis=0)
    onebest_iscr = np.sum([np.array(arc.iscr) for arc in onebest_path], axis=0)
    onebest_score = np.concatenate([onebest_score, onebest_nscr, onebest_iscr])
    onebest_err, _, onebest_hyp = utils.compute_word_error(
        onebest_hyp, ref, transform)
    oracle_err, oracle_align = lat.oracle_wer(ref)
    oracle_hyp = []
    for _, j in oracle_align:
        if (not j.startswith('*')) and (not j.endswith('*')):
            if j not in lattice.SPECIAL_KEYS:
                oracle_hyp.append(j)
        elif j != '**DEL**':
            oracle_hyp.append(j.lstrip('*').rstrip('*'))
    oracle_hyp = ' '.join(oracle_hyp)
    _, _, oracle_hyp = utils.compute_word_error(oracle_hyp, ref, transform)
    onebest_dict[uttid] = (onebest_err, onebest_hyp, onebest_score)
    oracle_dict[uttid] = (oracle_err, oracle_hyp)
    density_dict[uttid] = lat.density()

def run_parallel(args):
    """Looping through all lattices in parallel and get information."""
    # get reference and iterator
    name_dict, ref_dict = utils.load_ref(args.ref)
    all_lat = utils.file_iterator(args.latdir, '.lat.gz', resource=ref_dict)
    transform = utils.text_processing(args.acronyms) if args.clean_hyp else None

    # run all
    manager = Manager()
    onebest_dict = manager.dict()
    oracle_dict = manager.dict()
    density_dict = manager.dict()
    logging.info('Processing lattices using %d processes' % args.nproc)
    with Pool(processes=args.nproc) as pool:
        pool.starmap(
            lattice_operation,
            zip(all_lat, repeat(onebest_dict), repeat(oracle_dict),
                repeat(density_dict), repeat(args.gsf), repeat(transform))
        )

    # gather results
    total_onebest_err = sum([i[0] for i in onebest_dict.values()])
    total_oracle_err = sum([i[0] for i in oracle_dict.values()])
    total_ref_words = sum(
        [len(v.split()) for k, v in ref_dict.items() if k in onebest_dict])
    if len(ref_dict) != len(onebest_dict):
        logging.warning('Some lattices are missing:')
        logging.warning(set(ref_dict.keys() - set(onebest_dict.keys())))

    # write out to info file
    file_path = os.path.join(args.latdir, 'lattice_%s.info' % args.tag)
    logging.info('Writing lattice info to file %s' % file_path)
    with open(file_path, 'w') as fh:
        fh.write('=================Summary=================\n')
        fh.write('Number of lattices:    %d\n' % len(onebest_dict))
        fh.write('Word error rate:       %.2f\n'
                 % (total_onebest_err / total_ref_words * 100))
        fh.write('Oracle error rate:     %.2f\n'
                 % (total_oracle_err / total_ref_words * 100))
        fh.write('Avg lattice density:   %.2f\n'
                 % (sum(density_dict.values()) / len(density_dict)))
        fh.write('\n================Breakdown================\n')
        for uttid in sorted(onebest_dict.keys()):
            fh.write('UTTID:   %s\n' % name_dict[uttid])
            fh.write('REF:     %s\n' % ref_dict[uttid])
            fh.write('HYP:     %s\n' % onebest_dict[uttid][1])
            fh.write('SCORE:   %s\n' % ''.join(
                ['{:10.3f}'.format(x) for x in onebest_dict[uttid][2]]))
            fh.write('ORACLE:  %s\n' % oracle_dict[uttid][1])
            fh.write('DENSITY: %.2f\n' % density_dict[uttid])
            fh.write('-----------------------------------------\n')

def main():
    parser = argparse.ArgumentParser(
        description='Compute oracle & one-best WERs for lattices.')
    parser.add_argument('latdir', type=str,
                        help='input lattice directory')
    parser.add_argument('ref', type=str,
                        help='ground truth reference')
    parser.add_argument('nproc', type=int,
                        help='number of process to run in parallel')
    parser.add_argument('tag', type=str,
                        help='tag for output summary file')
    parser.add_argument('--gsf', type=float, default=5.0,
                        help='grammar scaling factor')
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
