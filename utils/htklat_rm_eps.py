#!/usr/bin/env python
#
# This script takes in a HTK format lattice and removes all <eps> nodes.
# The scores and times of the <eps> nodes are merged into the previous nodes.

import sys
import re
import copy

eps = '<eps>'
sentence_end = '</s>'

class htkarc:
  start_node=-1
  end_node=-1
  ac_score=0.0
  lm_score=0.0
  r_score=0.0
  others=""

  def __init__(self, start_node, end_node, ac_score, lm_score, r_score, others):
    self.start_node = start_node
    self.end_node = end_node
    self.ac_score = ac_score
    self.lm_score = lm_score
    self.r_score = r_score
    self.others = others

class htknode:
  time=0.0
  word=""
  others=""
  in_arcs = []
  out_arcs = []
  in_arcs_from_eps = []
  out_arcs_to_eps = []
  num_processed_inputs = 0

  def __init__(self, time, word, others):
    self.time = time
    self.word = word
    self.others = others
    self.in_arcs = []
    self.out_arcs = []
    self.in_arcs_from_eps = []
    self.out_arcs_to_eps = []
    self.num_processed_inputs

  def add_in_arc(self, arc_id):
    self.in_arcs.append(arc_id)

  def add_out_arc(self, arc_id):
    self.out_arcs.append(arc_id)

  def add_in_arc_from_eps(self, arc_id):
    self.in_arcs_from_eps.append(arc_id)

  def add_out_arc_to_eps(self, arc_id):
    self.out_arcs_to_eps.append(arc_id)

  def del_in_arc_from_eps(self, arc_id):
    self.in_arcs_from_eps.remove(arc_id)

  def del_out_arc_to_eps(self, arc_id):
    self.out_arcs_to_eps.remove(arc_id)

# create a new arc from start_node, by by-passing the next <eps> node
def process_eps_arc(arc_id, start_node, num_arcs, nodes, eps_nodes, arcs):
  new_arc_id = num_arcs
  old_arc = arcs[arc_id]

  # create new arcs
  for arc in eps_nodes[old_arc.end_node].out_arcs:
    arcs[new_arc_id] = htkarc(start_node, arcs[arc].end_node, arcs[arc].ac_score, arcs[arc].lm_score, arcs[arc].r_score, arcs[arc].others)
    nodes[start_node].add_out_arc(new_arc_id)
    if arc in nodes[arcs[arc].end_node].in_arcs_from_eps:
      if eps_nodes[arcs[arc].start_node].num_processed_inputs == len(eps_nodes[arcs[arc].start_node].in_arcs_from_eps)-1:
        nodes[arcs[arc].end_node].del_in_arc_from_eps(arc)
    nodes[arcs[arc].end_node].add_in_arc(new_arc_id)
    new_arc_id = new_arc_id + 1
  for arc in eps_nodes[old_arc.end_node].out_arcs_to_eps:
    arcs[new_arc_id] = htkarc(start_node, arcs[arc].end_node, arcs[arc].ac_score, arcs[arc].lm_score, arcs[arc].r_score, arcs[arc].others)
    nodes[start_node].add_out_arc_to_eps(new_arc_id)
    if arc in eps_nodes[arcs[arc].end_node].in_arcs_from_eps:
      if eps_nodes[arcs[arc].start_node].num_processed_inputs == len(eps_nodes[arcs[arc].start_node].in_arcs_from_eps)-1:
        eps_nodes[arcs[arc].end_node].del_in_arc_from_eps(arc)
    eps_nodes[arcs[arc].end_node].add_in_arc(new_arc_id)
    new_arc_id = new_arc_id + 1

  # add old arc scores to input arc
  nodes[start_node].time = eps_nodes[old_arc.end_node].time
  for arc in nodes[start_node].in_arcs:
    arcs[arc].ac_score = arcs[arc].ac_score + old_arc.ac_score
    arcs[arc].lm_score = arcs[arc].lm_score + old_arc.lm_score
    arcs[arc].r_score = arcs[arc].r_score + old_arc.r_score
  for arc in nodes[start_node].in_arcs_from_eps:
    arcs[arc].ac_score = arcs[arc].ac_score + old_arc.ac_score
    arcs[arc].lm_score = arcs[arc].lm_score + old_arc.lm_score
    arcs[arc].r_score = arcs[arc].r_score + old_arc.r_score

  return new_arc_id

# make a copy of node_id, without copying the outgoing arcs
def split_node_without_outarcs(node_id, num_arcs, num_nodes, nodes, eps_nodes, arcs):
  new_node_id = num_nodes
  new_arc_id = num_arcs
  old_node = nodes[node_id]
  nodes[new_node_id] = htknode(old_node.time, old_node.word, old_node.others)
  for in_arc in old_node.in_arcs:
    arcs[new_arc_id] = htkarc(arcs[in_arc].start_node, new_node_id, arcs[in_arc].ac_score, arcs[in_arc].lm_score, arcs[in_arc].r_score, arcs[in_arc].others)
    nodes[arcs[in_arc].start_node].add_out_arc(new_arc_id)
    nodes[new_node_id].add_in_arc(new_arc_id)
    new_arc_id = new_arc_id + 1
  for in_arc in old_node.in_arcs_from_eps:
    arcs[new_arc_id] = htkarc(arcs[in_arc].start_node, new_node_id, arcs[in_arc].ac_score, arcs[in_arc].lm_score, arcs[in_arc].r_score, arcs[in_arc].others)
    eps_nodes[arcs[in_arc].start_node].add_out_arc(new_arc_id)
    nodes[new_node_id].add_in_arc_from_eps(new_arc_id)
    new_arc_id = new_arc_id + 1

  new_node_id = new_node_id + 1
  return new_arc_id, new_node_id, new_node_id-1

# this function looks at a node just before an <eps> node, and removes all outgoing <eps> nodes
def process_prev_node(num_arcs, num_nodes, nodes, eps_nodes, arcs, node_id, processed_nodes, nodes_to_delete, eps_to_sentence_end):
  if node_id in processed_nodes:
    return num_arcs, num_nodes

  new_arc_id = num_arcs
  new_node_id = num_nodes
  node = nodes[node_id]

  # check whether ac_score, lm_score, r_score, t of next nodes match. If not, this node needs to be split
  if len(node.out_arcs)==0 and len(node.out_arcs_to_eps)==1:
    assert not eps_to_sentence_end[arcs[node.out_arcs_to_eps[0]].end_node]
    new_arc_id = process_eps_arc(node.out_arcs_to_eps[0], node_id, new_arc_id, nodes, eps_nodes, arcs)
  else:
    for out_arc in node.out_arcs_to_eps:
      if not eps_to_sentence_end[arcs[out_arc].end_node]:
        if node.time!=eps_nodes[arcs[out_arc].end_node].time or arcs[out_arc].ac_score!=0.0 or arcs[out_arc].lm_score!=0.0 or arcs[out_arc].r_score!=0.0:
          if len(node.out_arcs) > 0:
            new_arc_id, new_node_id, split_node = split_node_without_outarcs(node_id, new_arc_id, new_node_id, nodes, eps_nodes, arcs)
            new_arc_id = process_eps_arc(out_arc, split_node, new_arc_id, nodes, eps_nodes, arcs)
            processed_nodes.append(split_node)
          elif len(node.out_arcs_to_eps) > 1:
            if eps_nodes[arcs[node.out_arcs_to_eps[0]].end_node].time!=eps_nodes[arcs[out_arc].end_node].time or arcs[node.out_arcs_to_eps[0]]!=arcs[out_arc].ac_score or arcs[node.out_arcs_to_eps[0]]!=arcs[out_arc].lm_score or arcs[node.out_arcs_to_eps[0]]!=arcs[out_arc].r_score:
              new_arc_id, new_node_id, split_node = split_node_without_outarcs(node_id, new_arc_id,   new_node_id, nodes, eps_nodes, arcs)
              new_arc_id = process_eps_arc(out_arc, split_node, new_arc_id, nodes,eps_nodes, arcs)
              processed_nodes.append(split_node)
          else:
            new_arc_id = process_eps_arc(out_arc, node_id, new_arc_id, nodes,eps_nodes, arcs)
        else:
          new_arc_id = process_eps_arc(out_arc, node_id, new_arc_id, nodes,eps_nodes, arcs)

  out_arcs_to_eps = copy.deepcopy(node.out_arcs_to_eps)
  for out_arc in out_arcs_to_eps:
    if not eps_to_sentence_end[arcs[out_arc].end_node]:
      node.del_out_arc_to_eps(out_arc)

  # remove node if there are no more outgoing arcs. Since this function only processes previous nodes, this node should never be </s>.
  if len(node.out_arcs)==0 and len(node.out_arcs_to_eps)==0:
    nodes_to_delete.append(node_id)

  processed_nodes.append(node_id)
  return new_arc_id, new_node_id

# This function merges an <eps> node with its outgoing </s> nodes
def merge_eps_with_sentence_end(num_arcs, nodes, arcs, node_id):
  node = eps_nodes[node_id]
  new_arc_id = num_arcs
  assert len(node.in_arcs_from_eps) == 0
  assert len(node.out_arcs_to_eps) == 0

  for in_arc in node.in_arcs:
    for out_arc in node.out_arcs:
      arcs[new_arc_id] = htkarc(arcs[in_arc].start_node, arcs[out_arc].end_node, arcs[in_arc].ac_score+arcs[out_arc].ac_score, arcs[in_arc].lm_score+arcs[out_arc].lm_score, arcs[in_arc].r_score+arcs[out_arc].r_score, arcs[in_arc].others)
      nodes[arcs[in_arc].start_node].add_out_arc(new_arc_id)
      nodes[arcs[out_arc].end_node].add_in_arc(new_arc_id)
      new_arc_id = new_arc_id + 1

  for in_arc in node.in_arcs:
    nodes[arcs[in_arc].start_node].del_out_arc_to_eps(in_arc)
  for out_arc in node.out_arcs:
    nodes[arcs[out_arc].end_node].del_in_arc_from_eps(out_arc)

  return new_arc_id

# This function recursively removes <eps> nodes
def remove_eps_node(num_arcs, num_nodes, nodes, eps_nodes, arcs, node_id, processed_nodes, nodes_to_delete, eps_to_sentence_end):
  new_arc_id = num_arcs
  new_node_id = num_nodes
  node = eps_nodes[node_id]

  # ensure that all previous <eps> nodes have been removed. This is needed, because the lattice is not assumed to be topologically sorted
  in_arcs_from_eps = node.in_arcs_from_eps
  if len(in_arcs_from_eps)!=0 or len(node.out_arcs_to_eps)!=0:
    sys.stderr.write ('WARNING: sequence of consecutive <eps> detected!\n')
  while len(node.in_arcs_from_eps) > 0:
  #for in_arc in node.in_arcs_from_eps:
    new_arc_id, new_node_id = remove_eps_node(new_arc_id, new_node_id, nodes, eps_nodes, arcs, arcs[node.in_arcs_from_eps[0]].start_node, processed_nodes, nodes_to_delete, eps_to_sentence_end)
    del node.in_arcs_from_eps[0]

  if eps_to_sentence_end[node_id]:
    # if <eps> node has outgoing arcs only to </s> nodes, then merge <eps> with </s>.
    new_arc_id = merge_eps_with_sentence_end(new_arc_id, nodes, arcs, node_id)
  else:
    # if <eps> node has outgoing arcs to non </s> nodes, then merge <eps> with previous nodes
    in_arcs = node.in_arcs
    for in_arc in in_arcs:
      new_arc_id, new_node_id = process_prev_node(new_arc_id, new_node_id, nodes, eps_nodes, arcs, arcs[in_arc].start_node, processed_nodes, nodes_to_delete, eps_to_sentence_end)
      node.num_processed_inputs = node.num_processed_inputs + 1

  return new_arc_id, new_node_id

if (len(sys.argv) != 2):
  print("Remove <eps> from HTK format lattice.")
  print("Usage: htklat_rm_eps.py htk_lat_in > htk_lat_out")
  print("       cat htk_lat_in | fst2htklat.py - > htk_lat_out")
  sys.exit(1)

if (sys.argv[1] == "-"):
  f = sys.stdin
else:
  f = open(sys.argv[1])

# read in HTK lattice
nodes = {}
eps_nodes = {}
arcs = {}
with f as readfile:
  for line in readfile:
    line = line.rstrip()
    match = re.compile(r'^\s*(\S)=.+$').search(line)
    if match:
      if match.group(1) == 'I':
        # read node
        match = re.compile(r'^\s*I=(\d+)\s+t=(\S+)\s+W=(\S+)(\s*\S*.*\s*)$').search(line)
        if match:
          node_id = int(match.group(1))
          start_time = float(match.group(2))
          word = match.group(3)
          others = match.group(4)

          if word != eps:
            if (not node_id in eps_nodes) and (not node_id in nodes):
              nodes[node_id] = htknode(start_time, word, others)
            else:
              sys.stderr.write ('Error: multiple definitions for node %d\n' % node_id)

          else:
            if (not node_id in eps_nodes) and (not node_id in nodes):
              eps_nodes[node_id] = htknode(start_time, word, others)
            else:
              sys.stderr.write ('Error: multiple definitions for node %d\n' % node_id)

        else:
          sys.stderr.write ('Error: unrecognised line format in file: %s\n' % line)

      elif match.group(1) == 'J':
        # read arc
        match = re.compile(r'^\s*J=(\d+)\s+S=(\d+)\s+E=(\d+)\s+a=(\S+)\s+l=(\S+)\s+r=(\S+)(\s*\S*.*\s*)$').search(line)
        if match:
          arc_id = int(match.group(1))
          start_node = int(match.group(2))
          end_node = int(match.group(3))
          ac_score = float(match.group(4))
          lm_score = float(match.group(5))
          r_score = float(match.group(6))
          others = match.group(7)

          if not arc_id in arcs:
            arcs[arc_id] = htkarc(start_node, end_node, ac_score, lm_score, r_score, others)
          else:
            sys.stderr.write ('Error: multiple definitions for arc %d\n' % arc_id)

        else:
          sys.stderr.write ('Error: unrecognised line format in file: %s\n' % line)

      elif match.group(1) == 'N':
        match = re.compile(r'^\s*N=(\d+)\s+L=(\d+)\s*$').search(line)
        if match:
          num_nodes = int(match.group(1))
          num_arcs = int(match.group(2))
        else:
          sys.stderr.write ('Error: unrecognised line format in file: %s\n' % line)

f.close()

# find the in and out arcs for all nodes
for arc_id, arc in arcs.items():
  if arc.start_node in eps_nodes:
    if arc.end_node in eps_nodes:
      eps_nodes[arc.start_node].add_out_arc_to_eps(arc_id)
    else:
      eps_nodes[arc.start_node].add_out_arc(arc_id)
  else:
    if arc.end_node in eps_nodes:
      nodes[arc.start_node].add_out_arc_to_eps(arc_id)
    else:
      nodes[arc.start_node].add_out_arc(arc_id)

  if arc.end_node in eps_nodes:
    if arc.start_node in eps_nodes:
      eps_nodes[arc.end_node].add_in_arc_from_eps(arc_id)
    else:
      eps_nodes[arc.end_node].add_in_arc(arc_id)
  else:
    if arc.start_node in eps_nodes:
      nodes[arc.end_node].add_in_arc_from_eps(arc_id)
    else:
      nodes[arc.end_node].add_in_arc(arc_id)

processed_nodes = [] # this keeps track of whether a node just before an <eps> node has already been processed to remove all outgoing <eps> nodes
nodes_to_delete = [] # these should be nodes that have multiple outgoing <eps> arcs, but no outgoing non-eps arcs. We create copies of the node, and delete the original.

# find <eps> nodes that have outgoing arcs to only </s> nodes
eps_to_sentence_end = {}
for node_id, node in eps_nodes.items():
  if len(node.out_arcs_to_eps) > 0:
    eps_to_sentence_end[node_id] = False
  else:
    found = False
    for out_arc in node.out_arcs:
      if nodes[arcs[out_arc].end_node].word != sentence_end:
        found = True
        break
    if found:
      eps_to_sentence_end[node_id] = False
    else:
      eps_to_sentence_end[node_id] = True

# remove <eps> nodes
for node_id in eps_nodes:
  num_arcs, num_nodes = remove_eps_node(num_arcs, num_nodes, nodes, eps_nodes, arcs, node_id, processed_nodes, nodes_to_delete, eps_to_sentence_end)

# get new node IDs, by excluding <eps> nodes
new_node_ids = {}
new_nodes = {}
new_node_id = 0
for node_id, node in nodes.items():
  if node_id not in nodes_to_delete:
    new_node_ids[node_id] = new_node_id
    new_nodes[new_node_id] = node
    new_node_id = new_node_id + 1

# remove arcs connected to <eps> nodes
new_arcs = {}
new_arc_id = 0
for arc_id, arc in arcs.items():
  if not ((arc.start_node in eps_nodes) or (arc.end_node in eps_nodes) or (arc.start_node in nodes_to_delete) or (arc.end_node in nodes_to_delete)):
    new_arcs[new_arc_id] = arc
    x = arc.start_node
    y = arc.end_node
    new_arcs[new_arc_id].start_node = new_node_ids[arc.start_node]
    new_arcs[new_arc_id].end_node = new_node_ids[arc.end_node]
    new_arc_id = new_arc_id + 1

# print new HTK lattice
print('N=%d  L=%d' % (len(new_node_ids), len(new_arcs)))
for node_id in range(len(new_nodes)):
  print('I=%d\t t=%0.2f\t W=%s%s' % (node_id, new_nodes[node_id].time, new_nodes[node_id].word, new_nodes[node_id].others))
for arc_id in range(len(new_arcs)):
  print('J=%d\t S=%d\t E=%d\t a=%f\t l=%f\t r=%f%s' % (arc_id, new_arcs[arc_id].start_node, new_arcs[arc_id].end_node, new_arcs[arc_id].ac_score, new_arcs[arc_id].lm_score, new_arcs[arc_id].r_score, new_arcs[arc_id].others))

