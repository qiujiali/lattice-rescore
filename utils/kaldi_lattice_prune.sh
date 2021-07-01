cmd=run.pl
lmwt=10.0
beam=8.0
max_arcs=75

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1

if [ $# -ne 3 ]; then
    echo "Usage: utils/kaldi_lattice_prune.sh [--cmd (run.pl|queue.pl)] <data> <lang> <lattice-dir>"
    echo " Options:"
    echo "    --cmd (run.pl|queue.pl...)        # specify how to run the sub-processes."
    echo "    --lmwt <float>                    # language model weight for lattice pruning."
    echo "    --beam <float>                    # beam size for lattice pruning."
    echo "    --max_arcs <int>                  # maximum number of arcs per frame to keep."
    exit 1;
fi

data=$1
lang=$2
dir=$3
odir=${dir}_b${beam}_d${max_arcs}
mkdir -p ${odir}
cp ${dir}/num_jobs ${odir}/num_jobs

nj=$(cat ${dir}/num_jobs)
$cmd JOB=1:$nj ${odir}/log/lattice_prune.JOB.log \
    lattice-scale --inv-acoustic-scale=${lmwt} "ark:gunzip -c ${dir}/lat.JOB.gz|" ark:- \| \
    lattice-prune --beam=${beam} ark:- ark:- \| \
    lattice-limit-depth --max-arcs-per-frame=${max_arcs} ark:- "ark:|gzip -c > ${odir}/lat.JOB.gz"

echo "Pruning Done"

$cmd JOB=1:$nj ${odir}/log/lattice_depth.JOB.log \
    lattice-depth "ark:gunzip -c ${odir}/lat.JOB.gz|" ark:/dev/null || exit 1;

grep -w Overall ${odir}/log/lattice_depth.*.log | \
    awk -v nj=$nj '{num+=$6*$8; den+=$8; nl++} END{
      if (nl != nj) { print "Error: expected " nj " lines, got " nl | "cat 1>&2"; }
      printf("%.2f ( %d / %d )\n", num/den, num, den); }' > ${odir}/depth || exit 1;
echo -n "Depth is: "
cat ${odir}/depth


mkdir -p ${odir}/oracle
steps/cleanup/lattice_oracle_align.sh ${data} ${lang} ${odir} ${odir}/oracle
tail -n 2 ${odir}/oracle/analysis/per_spk_details.txt

steps/diagnostic/analyze_lats.sh ${lang} ${odir}

