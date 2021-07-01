#!/usr/bin/env python3

import os
import sys
import json
import argparse
import logging
import time
import numpy as np

from lattice import Lattice
import nbest_rescore
import utils

def lat2nbest(from_iterator, nbest, gsf=1.0, ip=0.0, rnnlms=None, iscas=None,
              sp=None, loaders=None, overwrite=False, acronyms={}):
    """N-best generation and compute RNNLM and/or ISCA scores."""
    uttid, lat_in, nbest_out, feat_path = from_iterator
    if not os.path.isfile(nbest_out) or overwrite:
        timing = []
        logging.info('Processing lattice %s' % lat_in)
        lat = Lattice(lat_in, file_type='htk')
        if gsf is None:
            gsf = float(lat.header['lmscale'])
        start = time.time()
        hyps, scores = nbest_rescore.get_nbest(lat, nbest, aw=(1 / gsf), ip=ip)
        timing.append(time.time() - start)
        if rnnlms is not None:
            rnnlm_scores = []
            for lm, word_dict in rnnlms:
                start = time.time()
                rnnlm_scores.append(nbest_rescore.rnnlm_rescore(
                    hyps, lm, word_dict))
                timing.append(time.time() - start)
            scores = np.concatenate((scores, np.array(rnnlm_scores).T), axis=1)
        if iscas is not None:
            isca_scores = []
            assert loaders is not None, 'loader is needed for ISCA rescoring'
            assert sp is not None, 'sp model is needed for ISCA rescoring'
            for isca, loader in zip(iscas, loaders):
                model, char_dict, model_type = isca
                start = time.time()
                feat = loader([(uttid, feat_path)])[0][0]
                isca_scores.append(
                    nbest_rescore.isca_rescore(
                        hyps, feat, model, char_dict, sp,
                        model_type=model_type, acronyms=acronyms
                    )
                )
                timing.append(time.time() - start)
            scores = np.concatenate((scores, np.array(isca_scores).T), axis=1)
        logging.info('Write nbest %s' % nbest_out)
        nbest_rescore.write_nbest(hyps, scores, nbest_out, uttid)
        logging.info('Time taken for %s: %s' % (
            uttid, ' '.join(['{:.3f}'.format(x) for x in timing])))
        return np.array(timing)
    else:
        logging.info('Keep the existing nbest %s' % nbest_out)

def main():
    parser = argparse.ArgumentParser(
        description='N-best generation and rescoring. '
                    'This should be run on the queue.')
    parser.add_argument('indir', type=str,
                        help='Input lattice directory.')
    parser.add_argument('outdir', type=str,
                        help='Output nbest directory, assuming the same '
                             'structure as input.')
    parser.add_argument('nbest', type=int,
                        help='Number of nbest hypotheses')
    parser.add_argument('--rnnlm_path', type=str, default=None, action='append',
                        help='Path to rnnlm model, may be multiple.')
    parser.add_argument('--isca_path', type=str, default=None, action='append',
                        help='Path to isca model, may be multiple.')
    parser.add_argument('--spm_path', type=str, default=None,
                        help='Path to sentencepiece model.')
    parser.add_argument('--js_path', type=str, default=None,
                         help='Path to json feature file for ISCA.')
    parser.add_argument('--gsf', type=float, default=1.0,
                        help='Grammar scaling factor.')
    parser.add_argument('--ip', type=float, default=0.0,
                        help='Insertion penalty per word.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing output file if exits.')
    parser.add_argument('--acronyms', type=str, default=None,
                        help='Path to acronoym mapping (swbd)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
    )
    logging.info(' '.join([sys.executable] + sys.argv))

    # read acronym mapping
    if args.acronyms:
        acronyms = utils.load_acronyms(args.acronyms)
    else:
        acronyms = {}

    # set up RNNLM
    if args.rnnlm_path:
        rnnlms = []
        for rnnlm_path in args.rnnlm_path:
            rnnlm = utils.load_espnet_rnnlm(rnnlm_path)
            rnnlms.append(rnnlm)
    else:
        rnnlms = None

    # set up ISCA
    if args.isca_path:
        from espnet.utils.io_utils import LoadInputsAndTargets
        iscas, loaders = [], []
        for isca_path in args.isca_path:
            model, char_dict, train_args = utils.load_espnet_model(isca_path)
            module_name = train_args.model_module
            if 'transformer' in module_name or 'conformer' in module_name:
                model_type = 'tfm'
            else:
                model_type = 'las'
            loader = LoadInputsAndTargets(
                mode='asr',
                load_output=False,
                sort_in_input_length=False,
                preprocess_conf=train_args.preprocess_conf,
                preprocess_args={'train': False},
            )
            iscas.append((model, char_dict, model_type))
            loaders.append(loader)
        with open(args.js_path, 'rb') as fh:
            js = json.load(fh)['utts']
    else:
        iscas, loaders, js = None, None, None

    # get sentencepice model if used
    if args.spm_path:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(args.spm_path)
    else:
        sp = None

    # set up iterator and run all
    all_lat = utils.file_iterator(
        args.indir, '.lat.gz', args.outdir, '.nbest', resource=js)
    all_time = None
    counter = 0
    for each_iteration in all_lat:
        timing = lat2nbest(each_iteration, args.nbest, args.gsf, args.ip,
                           rnnlms, iscas, sp, loaders, args.overwrite, acronyms)
        all_time = timing if all_time is None else all_time + timing
        counter += 1
    logging.info('Job finished on %s' % os.uname()[1])
    logging.info('Overall, for %d lattices, %.3f seconds used'
                 % (counter, sum(all_time)))
    logging.info('On average, the time taken for each key part is %s'
                 % (' '.join(['{:.3f}'.format(x) for x in all_time / counter])))

if __name__ == '__main__':
    main()
