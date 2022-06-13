#!/bin/bash

echo $0 $@
command="$0 $@"

lmwt=
acwt=
wip=

. ./path.sh
. parse_options.sh

if [ $# -ne 2 ]; then
  echo "Convert FST to HTK lattice"
  echo "Usage: $0 [options] <fst-scp> <htk-lat-dir>"
  echo "e.g.:  "
  echo ""
  echo "Options:"
  echo "--lmwt     LM scale in HTK lattice header"
  echo "--acwt     AC scale in HTK lattice header"
  echo "--wip      word insertion penalty in HTK lattice header"
  exit 1;
fi

data=$1;
outdir=$2;

mkdir -p $outdir/CMDs;
echo "$command" >> $outdir/CMDs/fst2htklat.log

if [ -z "$lmwt" ]; then
  lmwt=1.00
fi
if [ -z "$acwt" ]; then
  acwt=1.00
fi
if [ -z "$wip" ]; then
  wip=0.00
fi

for i in `cat ${data} `; do 
	utt=`echo $i | awk -F'/' '{printf("%s\n", $NF);}' `;
	zcat ${i} | \
	egrep -v ^[A-Z] |  \
	sed 's:[al]=::g;' | \
        awk '(NF>0){print}' | \
        utils/fst2htklat.py - | \
        utils/htklat_rm_eps.py - | \
        sed "s/^'/\\\'/g" | sed "s/\([^\\]\)'/\1\\\'/g" | \
        sed "1iVERSION=1.0\nlmscale=${lmwt}\nwdpenalty=${wip}\nprscale=1.00\nacscale=${acwt}" | \
        gzip -fc > ${outdir}/${utt};
done


