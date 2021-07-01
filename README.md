# Combining Frame-Synchronous and Label-Synchronous Systems for Speech Recognition

This repository is the code used in our paper:

>**[Combining Frame-Synchronous and Label-Synchronous Systems for Speech Recognition](https://arxiv.org/abs/2107.?????)**
>
>*Qiujia Li, Chao Zhang, Phil Woodland*
>
>Submitted to TASLP

## Overview

Frame-synchronous (e.g. HMM-based systems) and label-synchronous systems (e.g. attention-based end-to-end systems) have their own advantages for ASR. To obtain the best of both worlds and to exploit the complementarity of both systems, we propose to first generate n-best lists/lattices from the frame-synchronous system and then rescore the hypothesis space using the label-synchronous system during the second pass.

In this work, we use the standard Kaldi systems for the first pass and the ESPnet systems (attention-based models only) for the second pass. The main results on AMI-IHM and SwitchBoard 300hare as follows.

|                     	| AMI-dev 	| AMI-eval 	| SWB-Hub5'00 	| SWB-RT03 	|
|---------------------	|--------:	|---------:	|------------:	|---------:	|
| Kaldi               	|    19.9 	|     19.2 	|        12.8 	|     15.3 	|
| ESPnet(LSTM)        	|    19.6 	|     18.2 	|        11.4 	|     13.5 	|
| ESPnet(Transformer) 	|    19.4 	|     19.1 	|        11.1 	|     14.1 	|
| Combine(500-best)   	|    15.8 	|     14.1 	|         9.1 	|     11.3 	|
| **Combine(lattice)** 	|    **15.7** 	|     **13.8** 	|         **8.9** 	|     **10.5** 	|

For more details, please refer to the [paper](https://arxiv.org/pdf/2107.?????.pdf).

## Dependencies
Apart from dependencies for [ESPnet](https://github.com/espnet/espnet) and [Kaldi](https://github.com/kaldi-asr/kaldi) for ASR, the following tools are used:
* [HTK](https://htk.eng.cam.ac.uk/) - only `HLRescore` is used for converting Kaldi lattices into HTK format
* [CMA-ES](https://github.com/CMA-ES/pycma) - for tuning interpolation coefficients
* [edidistance](https://github.com/roy-ht/editdistance) and [jiwer](https://github.com/jitsi/jiwer/) - for WER computation

## Pipeline

### Building frame and label-synchronous systems

Please follow ESPnet and Kaldi recipes for training and evaluating ASR systems.

### Kaldi to HTK Lattice Conversion

Kaldi lattices are stored in the form of FSTs, which is not straightforward to see and manipulate. Therefore, we prune and convert Kaldi lattices into the HTK format first. A detailed discussion about these two formats can be found [here](https://senarvi.github.io/kaldi-lattices/).

To prune Kaldi lattices,

```sh
kaldi_dir=  # e.g. kaldi/egs/ami/s5b
ln -s $(pwd)/utils/kaldi_lattice_prune.sh ${kaldi_dir}/utils/
cd ${kaldi_dir} 
utils/kaldi_lattice_prune.sh --beam ${beam} --max_arcs ${max_arcs} ${data} ${lang} ${dir}
```
In our paper, `beam=8.0` and `max_arcs=75` for AMI and `beam=9.5` and `max_arcs=75` for SwitchBoard.

Then convert the pruned lattices into HTK format,
```sh
utils/kaldi2htk_lattices.sh ${kaldi_lat_dir} ${htk_lat_dir}
```
Before executing the script, please make sure `${dict}` `${kaldilm}` `${kaldilmformat}` and `${htklm}` are the ones that you are intend to use. Examples of these files can be found at the default location for AMI dataset.

### N-best or Lattice Rescoring
`get_nbest.py` and `get_lattice.py` read HTK lattices, compute corresponding scores from RNNLM and/or attention-based models trained using ESPnet, and write out the N-best lists or lattices with additional scores from label-synchronous models. The help option `-h` will explain all input arguments.

`get_nbest_info.py` and `get_lattice_info.py` compute useful information about n-best lists or lattices, such as one-best WER, oracle WER and lattice density. Reference transcription is required.

`qsub_get_lattice.sh` and `qsub_get_nbest.sh` are scripts for submitting array jobs to the queueing systems such as SGE.

An n-best list and a lattice after rescoring with an RNNLM and two attention-based models on AMI dev set is provided in `examples/`. 

### Find Optimal Interpolation Coefficients
For optimal performance, interpolation coefficients between acoustic model, n-gram language model, RNNLM, and attention-based models need to be searched. `interpolation.py` loads an N-best or a lattice directory and the corresponding reference transcription, and runs a search algorithm called revolution strategy to find the coefficients that minimises the WER on the dataset. This should be run on the dev set.

### Final Evaluation
After obtaining the optimal interpolation coefficients on the dev set, run `get_rescore_results.py` with these coefficients to find the WER on the eval set. Note that different datasets may have different scoring scripts. AMI and SwitchBoard scoring scripts are available in `local/`.

## References
```
@article{Li2021CombiningFA,
  title={Combining frame-synchronous and label-synchronous systems for speech recognition},
  author={Li, Qiujia and Zhang, Chao and Woodland, Philip C.},
  journal={arXiv},
  year={2021},
}

@inproceedings{Li2019IntegratingSA,
  title={Integrating source-channel and attention-based sequence-to-sequence models for speech recognition},
  author={Li, Qiujia and Zhang, Chao and Woodland, Philip C.},
  booktitle={ASRU},
  year={2019},
  address={Singapore}
}
```