#!/bin/bash

echo $0 $@

stage=1
cmd=run.pl
tscale=10.0 # the transition probability scale
original_scores=false
lm_fst_format=fst
frame_rate=0.01
remove_lm=false
remove_fst=true
model=

. ./path.sh
. parse_options.sh

set -euo pipefail

if [ $# -ne 3 ]; then
  echo "Convert Kaldi lattices to HTK format. Removes LM scores and pushes transition probabilities to acoustic scores. Need to do LM rescoring in HTK"
  echo "Usage: $0 [options] <lang> <src-lattice-dir> <htk-lattice-dir>"
  echo "e.g.:  "
  echo ""
  echo "Options:"
  echo "--cmd              (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "--original-scores  If true, do not modify LM and AC scores. If false, remove LM scores and push transition probabilities to AC score (default: $original_scores)"
  echo "--tscale           Transition probability scale. Only applicable if --original-scores is false (default: $tscale)"
  echo "--lm-fst-format    Format of language model FST in <lang>/G.* . Either fst or carpa (default: $lm_fst_format)"
  exit 1;
fi

lang=$1
src_dir=$2
dir=$3

if [ -z "$model" ]; then
  model=$src_dir/../final.mdl # assume model one level up from decoding dir.
fi

for x in $src_dir/lat.1.gz $model; do
  if [ ! -f $x ]; then
    echo "Error: missing file $x"
    exit 1
  fi
done

if [ "$original_scores" = "false" ]; then
  if [ "$lm_fst_format" = "fst" ]; then
    [ ! -f "$lang/G.fst" ] && echo "Error: missing file $lang/G.fst" && exit 1
  elif [ "$lm_fst_format" = "carpa" ]; then
    [ ! -f "$lang/G.carpa" ] && echo "Error: missing file $lang/G.carpa" && exit 1
  else
    echo "Error: unrecognised --lm-fst-format $lm_fst_format" && exit 1
  fi
fi

nj=`cat $src_dir/num_jobs`
mkdir -p $dir/log
echo $0 $@ >> $dir/CMD
dirname=fsts-in-htk-cued

if [ $stage -le 1 ]; then
  # remove LM score, fix timings, determinize, convert to FST
  # need to remove LM score, so that transition probability can be isolated and pushed to AC score, following the convention in HTK
  # need to align words to fix timing information
  if [ ! -e "$dir/$dirname/fst.scp" ]; then
    if [ "$original_scores" = "false" ]; then
      if [ "$remove_lm" = "true" ]; then
        # remove LM score, merge transition probabilities into AC score
        if [ "$lm_fst_format" = "fst" ]; then
          $cmd JOB=1:$nj $dir/log/lat2fst.JOB.log \
            lattice-lmrescore --lm-scale=-1.0 "ark:gunzip -c $src_dir/lat.JOB.gz |" "fstproject --project_output=true $lang/G.fst |" ark:- \| \
            lattice-align-words --output-error-lats=true $lang/phones/word_boundary.int $model ark:- ark,t:- \| \
            utils/int2sym.pl -f 3 $lang/words.txt \| \
            utils/convert_slf.pl --frame-rate $frame_rate --tscale $tscale - $dir/$dirname/JOB || exit 1
        elif [ "$lm_fst_format" = "carpa" ]; then
          $cmd JOB=1:$nj $dir/log/lat2fst.JOB.log \
            lattice-lmrescore-const-arpa --lm-scale=-1.0 "ark:gunzip -c $src_dir/lat.JOB.gz |" "$lang/G.carpa" ark:- \| \
            lattice-align-words --output-error-lats=true $lang/phones/word_boundary.int $model ark:- ark,t:- \| \
            utils/int2sym.pl -f 3 $lang/words.txt \| \
            utils/convert_slf.pl --frame-rate $frame_rate --tscale $tscale - $dir/$dirname/JOB || exit 1
        else
          echo "Error: unrecognised --lm-fst-format $lm_fst_format"
          exit 1
        fi
      else
        # retain LM score, merge transition probabilities into AC score
        if [ "$lm_fst_format" = "fst" ]; then
          $cmd JOB=1:$nj $dir/log/lat2fst.JOB.log \
            lattice-lmrescore --lm-scale=-1.0 "ark:gunzip -c $src_dir/lat.JOB.gz |" "fstproject --project_output=true $lang/G.fst |" ark:- \| \
            lattice-scale --lm2acoustic-scale=$tscale --lm-scale=0.0 ark:- ark:- \| \
            lattice-lmrescore --lm-scale=1.0 ark:- "fstproject --project_output=true $lang/G.fst |" ark:- \| \
            lattice-align-words --output-error-lats=true $lang/phones/word_boundary.int $model ark:- ark,t:- \| \
            utils/int2sym.pl -f 3 $lang/words.txt \| \
            utils/convert_slf.pl --frame-rate $frame_rate --original-scores - $dir/$dirname/JOB || exit 1
        elif [ "$lm_fst_format" = "carpa" ]; then
          $cmd JOB=1:$nj $dir/log/lat2fst.JOB.log \
            lattice-lmrescore-const-arpa --lm-scale=-1.0 "ark:gunzip -c $src_dir/lat.JOB.gz |" "$lang/G.carpa" ark:- \| \
            lattice-scale --lm2acoustic-scale=$tscale --lm-scale=0.0 ark:- ark:- \| \
            lattice-lmrescore-const-arpa --lm-scale=1.0 ark:- "$lang/G.carpa" ark:- \| \
            lattice-align-words --output-error-lats=true $lang/phones/word_boundary.int $model ark:- ark,t:- \| \
            utils/int2sym.pl -f 3 $lang/words.txt \| \
            utils/convert_slf.pl --frame-rate $frame_rate --original-scores - $dir/$dirname/JOB || exit 1
        else
          echo "Error: unrecognised --lm-fst-format $lm_fst_format"
          exit 1
        fi
      fi
    else
      # retain (LM+transition prob) and AC scores as they are in Kaldi lattice
      $cmd JOB=1:$nj $dir/log/lat2fst.JOB.log \
        lattice-align-words --output-error-lats=true $lang/phones/word_boundary.int $model "ark:gunzip -c $src_dir/lat.JOB.gz |" ark,t:- \| \
        utils/int2sym.pl -f 3 $lang/words.txt \| \
        utils/convert_slf.pl --frame-rate $frame_rate --original-scores - $dir/$dirname/JOB || exit 1
    fi

    # make list of lattices
    find -L $PWD/$dir/$dirname/ -name '*.lat.gz' > $dir/$dirname/fst.scp || exit 1
    for i in `seq 1 $nj`; do
      find -L $PWD/$dir/$dirname/${i}/ -name '*.lat.gz' > $dir/$dirname/${i}/fst.scp || exit 1
    done
  fi
fi

if [ $stage -le 2 ]; then
  # convert FST to HTK lattice, mapping the words from arcs to nodes
  $cmd JOB=1:$nj $dir/log/fst2htklat.JOB.log \
    utils/fst2htklat.sh $dir/$dirname/JOB/fst.scp $dir/JOB || exit 1
fi

if [ $stage -le 3 ]; then
  find -L $PWD/$dir/ -name '*.lat.gz' | sed 's/.gz//g' | LC_ALL=C sort > $dir/lat_htk.scp || exit 1
  for i in `seq 1 $nj`; do
    find -L $PWD/$dir/${i}/ -name '*.lat.gz' | sed 's/.gz//g' | LC_ALL=C sort > $dir/${i}/lat_htk.scp || exit 1
  done

  if [ "$remove_fst" = "true" ]; then
    rm -rf $dir/$dirname
  fi
fi

echo "Done!"

