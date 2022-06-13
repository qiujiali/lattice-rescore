#!/usr/bin/env python

import sys
import re

class fstarc:
  start_id=-1
  end_id=-1
  word=""
  time=0.0
  ac_score=0.0
  lm_score=0.0

  def __init__(self, start_id, end_id, word, time, ac_score, lm_score):
    self.start_id = start_id
    self.end_id = end_id
    self.word = word
    self.time = time
    self.ac_score = ac_score
    self.lm_score = lm_score

class fstnode:
  in_arcs=[]
  out_arcs=[]

class htkarc:
  start_node=-1
  end_node=-1
  ac_score=0.0
  lm_score=0.0

  def __init__(self, start_node, end_node, ac_score, lm_score):
    self.start_node = start_node
    self.end_node = end_node
    self.ac_score = ac_score
    self.lm_score = lm_score

class htknode:
  time=0.0
  word=""

  def __init__(self, time, word):
    self.time = time
    self.word = word

if (len(sys.argv) != 2):
  print("Read FST format lattice, push words from arcs to nodes, and print HTK format lattice.")
  print("Usage: fst2htklat.py fst > htk.lat")
  print("       cat fst | fst2htklat.py - > htk.lat")
  sys.exit(1)

if (sys.argv[1] == "-"):
  f = sys.stdin
else:
  f = open(sys.argv[1])

# read in FST arcs
max_end_id = 0
fst_arcs = []
fst_nodes = {}
fst_arc_id = 0
with f as readfile:
  for line in readfile:
    line = line.rstrip()
    match = re.compile(r'^\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$').search(line)
    if match:
      start_id = int(match.group(1))
      end_id = int(match.group(2))
      in_word = match.group(3)
      out_word = match.group(4)
      time = float(match.group(5))
      ac_score = float(match.group(6))
      lm_score = float(match.group(7))

      if in_word != out_word:
        sys.stderr.write("Error: non-matching input and output symbols in FST %s %s\n" % (in_word, out_word))
        sys.exit(1)

      arc = fstarc(start_id, end_id, out_word, time, ac_score, lm_score)
      fst_arcs.append(arc)

      if end_id in fst_nodes:
        fst_nodes[end_id].in_arcs.append(fst_arc_id)
      else:
        fst_nodes[end_id] = fstnode()
        fst_nodes[end_id].in_arcs.append(fst_arc_id)

      if start_id in fst_nodes:
        fst_nodes[start_id].out_arcs.append(fst_arc_id)
      else:
        fst_nodes[start_id] = fstnode()
        fst_nodes[start_id].out_arcs.append(fst_arc_id)

      if end_id > max_end_id:
        max_end_id = end_id

      fst_arc_id = fst_arc_id + 1

f.close()

# create HTK lattice, by pushing words from arcs to nodes
fst_node_to_htk_node = [[] for i in range(max_end_id+1)] # stores a mapping from fst arc end nodes to htk nodes.
fst_arc_to_htk_node = [-1 for i in range(len(fst_arcs))] # keeps track of which htk node each word is pushed to
max_end_id = max_end_id + 1
htk_nodes = {}
htk_nodes[0] = htknode(0.0, "!NULL")
fst_node_to_htk_node[0].append(0)
for fst_arc_id in range(len(fst_arcs)):

  # this script assumes that the very first arc in the FST file corresponds to the starting arc
  if not htk_nodes[fst_arcs[fst_arc_id].start_id]:
    sys.stderr.write("Error: start node not found\n")
    sys.exit(1)

  # we have already seen this fst end node
  if fst_arcs[fst_arc_id].end_id in htk_nodes:
    # the new arc does not agree with the previously seen arc that leads to the same fst end node, so we make a new HTK node and push the fst arc word to it
    # also, need to create separate nodes for !NULL, otherwise there may be problems with LM rescoring
    if ((not (htk_nodes[fst_arcs[fst_arc_id].end_id].word==fst_arcs[fst_arc_id].word and htk_nodes[fst_arcs[fst_arc_id].end_id].time==fst_arcs[fst_arc_id].time)) or (fst_arcs[fst_arc_id].word=="!NULL")):
      htk_nodes[max_end_id] = htknode(fst_arcs[fst_arc_id].time, fst_arcs[fst_arc_id].word)
      fst_node_to_htk_node[fst_arcs[fst_arc_id].end_id].append(max_end_id)
      fst_arc_to_htk_node[fst_arc_id] = max_end_id
      max_end_id = max_end_id + 1
    else:
      fst_arc_to_htk_node[fst_arc_id] = fst_arcs[fst_arc_id].end_id

  # this is the first time that we see this fst end node, so we make a new HTK node and push the fst arc word to it
  else:
    htk_nodes[fst_arcs[fst_arc_id].end_id] = htknode(fst_arcs[fst_arc_id].time, fst_arcs[fst_arc_id].word)
    fst_node_to_htk_node[fst_arcs[fst_arc_id].end_id].append(fst_arcs[fst_arc_id].end_id)
    fst_arc_to_htk_node[fst_arc_id] = fst_arcs[fst_arc_id].end_id

# join the HTK nodes with arcs
htk_arcs = []
for fst_arc_id in range(len(fst_arcs)):
  for start_node in fst_node_to_htk_node[fst_arcs[fst_arc_id].start_id]:
    htk_arcs.append(htkarc(start_node, fst_arc_to_htk_node[fst_arc_id], fst_arcs[fst_arc_id].ac_score, fst_arcs[fst_arc_id].lm_score))

# print HTK lattice
print("N=%d  L=%d" % (len(htk_nodes), len(htk_arcs)))

# print nodes
for htk_node_id in htk_nodes:
  print("I=%d     t=%.2f	 W=%s	 v=1" % (htk_node_id, htk_nodes[htk_node_id].time, htk_nodes[htk_node_id].word))

# print arcs
for htk_arc_id in range(len(htk_arcs)):
  print("J=%d      S=%d      E=%d      a=%f	 l=%f	 r=0" % (htk_arc_id, htk_arcs[htk_arc_id].start_node, htk_arcs[htk_arc_id].end_node, htk_arcs[htk_arc_id].ac_score, htk_arcs[htk_arc_id].lm_score))

