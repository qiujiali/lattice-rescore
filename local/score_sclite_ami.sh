#!/bin/bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.

# begin configuration section.
cmd=run.pl
stage=0
tag=
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 2 ]; then
  echo "Usage: local/score_sclite.sh [--cmd (run.pl|queue.pl...)] <data-dir> <decode-dir> (<dict>)"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  exit 1;
fi

data=$1
dir=$2
tag=${3:-}
dict=${4:-}

hubscr=${KALDI_ROOT}/tools/sctk/bin/hubscr.pl
[ ! -f ${hubscr} ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=$(dirname ${hubscr})

for f in ${data}/stm ${data}/glm; do
  [ ! -f ${f} ] && echo "$0: expecting file $f to exist" && exit 1;
done
name=$(basename ${data}) # e.g. eval2000

if [ -z ${tag} ]; then
  score_dir=${dir}/scoring
else
  score_dir=${dir}/scoring_${tag}
fi
ctm=${score_dir}/hyp.ctm
stm=${score_dir}/ref.stm
mkdir -p ${score_dir}
if [ ${stage} -le 0 ]; then
    if [ -z ${dict} ]; then
        # Assuming trn files exist
        ref=${dir}/ref.wrd.trn
        if [ -z ${tag} ]; then
          hyp=${dir}/hyp.wrd.trn
        else
          hyp=${dir}/hyp_${tag}.wrd.trn
        fi
        python trn2stm.py --orig-stm ${data}/stm ${ref} ${stm}
        python trn2ctm_ami.py ${hyp} ${ctm}
    else
        ref=${dir}/ref.trn
        hyp=${dir}/hyp.trn
        json2sctm.py ${dir}/data.json ${dict} --orig-stm ${data}/stm --stm ${stm} --refs ${ref} --ctm ${ctm} --hyps ${hyp}
    fi
fi

if [ $stage -le 1 ]; then
# Remove some stuff we don't want to score, from the ctm.
# - we remove hesitations here, otherwise the CTM would have a bug!
#   (confidences in place of the removed hesitations),
    cp ${ctm} ${score_dir}/tmpf;
    cat ${score_dir}/tmpf | grep -i -v -E '\[noise|laughter|vocalized-noise\]' | \
      grep -i -v -E ' (ACH|AH|EEE|EH|ER|EW|HA|HEE|HM|HMM|HUH|MM|OOF|UH|UM) ' | \
      grep -i -v -E '<unk>' > ${ctm};
fi

# Score the set...
if [ ${stage} -le 2 ]; then
    ${cmd} ${score_dir}/score.log ${hubscr} -p ${hubdir} -V -v -l english -h rt-stt -g ${data}/glm -r ${stm} ${ctm} || exit 1;
fi

grep -e Ref -e Error ${score_dir}/hyp.ctm.filt.dtl
#grep 'Percent Total Error' ${score_dir}/hyp.ctm.filt.dtl
grep Sum ${score_dir}/hyp.ctm.filt.sys

exit 0
