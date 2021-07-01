echo $0 $@

stage=1
cmd='run.pl'
tscale=1.0
hlrescore="${HLRescore_BIN} -A -D -V -T 1"
cfg='utils/hlrescore.cfg'
dict='data/local/dict/lexicon.4htk.txt'
kaldilm='data/lang_ami_fsh.o3g.kn'
kaldilmformat='fst' # for swbd, the default kaldi lm is in arpa format
htklm='data/local/lm/ami_fsh.o3g.kn.4htk'

. ./path.sh
. parse_options.sh

set -euo pipefail

kaldi_lat_dir=$1
htk_lat_dir=$2
htk_lat_dir_nolm=${htk_lat_dir}_nolm
mkdir -p $htk_lat_dir

echo "=========================" >> $htk_lat_dir/../CMD
echo "#$PWD" >> $htk_lat_dir/../CMD
echo $0 $@ >> $htk_lat_dir/../CMD


if [ $stage -le 1 ]; then
    mkdir -p $htk_lat_dir_nolm
    echo "Converting Kaldi lattices from ${kaldi_lat_dir} to HTK lattices in ${htk_lat_dir_nolm}"
    utils/convert_slf_parallel.sh \
        --cmd "$cmd" \
        --tscale $tscale \
        --original-scores false \
        --frame-rate 0.03 \
        --remove-lm true \
        --lm-fst-format ${kaldilmformat} \
        $kaldilm \
        $kaldi_lat_dir \
        $htk_lat_dir_nolm
fi

if [ $stage -le 2 ]; then
    nj=`cat $kaldi_lat_dir/num_jobs`
    rm -f $htk_lat_dir/.error
    echo "Writing to $htk_lat_dir"
    for i in $(seq 1 $nj); do
        (
        mkdir -p $htk_lat_dir/$i
        for file in $htk_lat_dir_nolm/$i/*.lat.gz; do
            $hlrescore -C $cfg -n $htklm -w -l $htk_lat_dir/$i \
                $dict ${file%.gz} >> $htk_lat_dir/$i/log 2>&1
        done
        ) || touch $htk_lat_dir/.error &
    done
    wait
    if [ -f $htk_lat_dir/.error ]; then
        echo "$0: something went wrong for hlrescore"
        exit 1
    fi
fi

echo "Done!"
