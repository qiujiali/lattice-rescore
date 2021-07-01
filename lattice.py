#!/usr/bin/env python3

import itertools
import gzip
import re
import logging
import numpy as np

LOGZERO = -np.inf
MAXINT = np.iinfo(np.int).max
FRATE = 100
NULL = '!NULL'
SOS = '<s>'
EOS = '</s>'
UNK = '<unk>'
OOV = '[oov]'
SPECIAL_KEYS = [NULL, SOS, EOS, UNK, OOV]


class Lattice(object):
    """Directed acyclic graph representation of a phone/word lattice."""

    class Node(object):
        """
        Node in a DAG representation of a phone/word lattice.
        :param sym: Word corresponding to this node.  All arcs out of
                    this node represent hypothesised instances of this
                    word starting at frame `entry`.
        :type sym: string
        :param entry: Entry frame for this node.
        :type entry: int
        :param var: Pronunciation variant number.
        :type var: int
        :param exits: List of arcs out of this node.
        :type exits: list of Dag.Arc
        :param entries: List of arcs into this node
        :type entries: list of Dag.Arc
        :param score: Viterbi (or other) score for this node, used in bestpath 
                      calculation.
        :type score: float
        :param post: Posterior probability of this node.
        :type post: float
        :param prev: Backtrace pointer for this node, used in bestpath
                     calculation.
        :type prev: object
        :param entropy: Log entropy of this node.
        :type entropy: float
        :param fan: Temporary fan counter used in edge traversal.
        :type fan: int
        :param expnode: Expanded node list for lattice expansion.
        :type expnode: list of Dag.Node
        :param exparc: Expanded arc list for lattice expansion.
        :type exparc: list of Dag.Arc
        :param lmstate: A tuple of lm state and ngram history.
        :type lmstate: tuple
        """
        __slots__ = ('sym', 'entry', 'var', 'exits', 'entries', 'score',
                     'post', 'prev', 'entropy', 'fan', 'expnodes', 'exparcs',
                     'lmstate')
        def __init__(self, sym, entry, var):
            self.sym = sym
            self.entry = entry
            self.var = var
            self.exits = []
            self.entries = []
            self.score = LOGZERO
            self.post = LOGZERO
            self.prev = None
            self.entropy = None
            self.fan = 0
            self.expnodes = []
            self.exparcs = []
            self.lmstate = ([], 0)

        def __str__(self):
            return '<Node: %s[%d]>' % (self.sym, self.entry)

        def subnode(self):
            return type(self)(self.sym, self.entry, self.var)


    class Arc(object):
        """
        Arc in DAG representation of a phone/word lattice.
        :param src: Start node for this arc.
        :type src: Dag.Node
        :param dest: End node for this arc.
        :type dest: Dag.Node
        :param ascr: Acoustic score for this arc.
        :type ascr: float
        :param lscr: Best language model score for this arc.
        :type lscr: float
        :param nscr: RNNLM score for this arc.
        :type nscr: list
        :param iscr: ISCA score for this arc.
        :type iscr: list
        :type alpha: float
        :param beta: Conditional log-prob of all paths following this arc.
        :type beta: float
        :param post: Posterior log-prob of this arc.
        :type post: float
        :param prev: Previous arc in best path.
        :type prev: Dag.Arc
        """
        __slots__ = ('src', 'dest', 'ascr', 'lscr', 'nscr', 'iscr', 'alpha',
                     'beta', 'post', 'prev')
        def __init__(self, src, dest, ascr, lscr, nscr, iscr,
                     alpha=LOGZERO, beta=LOGZERO, post=LOGZERO):
            self.src = src
            self.dest = dest
            self.ascr = ascr
            self.lscr = lscr
            self.nscr = nscr
            self.iscr = iscr
            self.alpha = alpha
            self.beta = beta
            self.post = post
            self.prev = None

        def __str__(self):
            return ('<Arc: %s[%d]-->%s[%d], a=%f, l=%f>' % (
                self.src.sym,
                self.src.entry,
                self.dest.sym,
                self.dest.entry,
                self.ascr,
                self.lscr,
            ))

        def subarc(self, src, dest):
            return type(self)(src, dest, self.ascr, self.lscr, self.nscr.copy(),
                              self.iscr.copy())


    def __init__(self, file_path=None, file_type='htk', header=None,
                 nframes=None):
        """
        Construct a DAG, optionally loading contents from a file.
        :param file_path: HTK SLF format word lattice file to load (optionally).
        :type file_path: string
        :param file_type: Either `htk` or `kaldi`.
        :type file_type: str
        :param header: Header of HTK lattice file.
        :type header: dict
        """
        self.header = header
        self.nframes = nframes
        self.nodes = []
        self.arcs = []
        self.start = None
        self.end = None
        if file_path:
            if file_type == 'htk':
                self.htk2dag(file_path)
            elif file_type == 'kaldi':
                self.kaldi2dag(file_path)
            else:
                raise ValueError('file_type must be either htk or kaldi')

    def kaldi2dag(self, file_path):
        """Read a Kaldi format lattice file to populate a DAG."""
        raise NotImplementedError

    def htk2dag(self, file_path):
        """Read an HTK format lattice file to populate a DAG."""
        field_re = re.compile(r'(\S+)=(?:"((?:[^\\"]+|\\.)*)"|(\S+))')
        open_fn = gzip.open if file_path.endswith('.gz') else open
        with open_fn(file_path, 'rt', encoding='utf-8') as fh:
            self.header = {}
            self.nframes = 0
            state = 'header'
            # Read everything
            for spam in fh:
                if spam.startswith('#'):
                    continue
                fields = dict(map(lambda t: (t[0], t[1] or t[2]),
                              field_re.findall(spam.rstrip())))
                # Number of nodes and arcs
                if 'N' in fields:
                    num_nodes = int(fields['N'])
                    self.nodes = [None] * num_nodes
                    num_arcs = int(fields['L'])
                    self.arcs = [None] * num_arcs
                    state = 'items'
                if state == 'header':
                    self.header.update(fields)
                else:
                    # This is a node
                    if 'I' in fields:
                        idx = int(fields['I'])
                        frame = int(float(fields['t']) * FRATE)
                        var = int(fields['v']) if 'v' in fields else None
                        node = self.Node(
                            fields['W'].replace('\\', ''), frame, var)
                        self.nodes[idx] = node
                        if frame > self.nframes:
                            self.nframes = frame
                    # This is an arc
                    elif 'J' in fields:
                        idx = int(fields['J'])
                        start_node = self.nodes[int(fields['S'])]
                        end_node = self.nodes[int(fields['E'])]
                        ascr = float(fields.get('a', 0))
                        lscr = float(fields.get('l', 0))
                        nscr = fields.get('n', [])
                        if isinstance(nscr, str):
                            nscr = [float(n) for n in nscr.split(',')]
                        iscr = fields.get('i', [])
                        if isinstance(iscr, str):
                            iscr = [float(i) for i in iscr.split(',')]
                        arc = self.Arc(
                            start_node, end_node, ascr, lscr, nscr, iscr)
                        self.arcs[idx] = arc
                        # Link up existing nodes
                        start_node.exits.append(arc)
                        end_node.entries.append(arc)

        self.sort_nodes()

    def dag2htk(self, file_path):
        """Write out a lattice in HTK format."""
        open_fn = gzip.open if file_path.endswith('.gz') else open
        with open_fn(file_path, 'wb') as fh:
            for k, v in self.header.items():
                string = '%s=%s\n' % (k, v)
                fh.write(string.encode())
            fh.write(('N=%d\tL=%d\n' % (
                self.num_nodes(), self.num_arcs())).encode())
            mapping = {}
            for idx, node in enumerate(self.nodes):
                if node.var:
                    string = 'I=%d\tt=%.2f\tW=%s\tv=%d\n' % (
                        idx, node.entry/FRATE, node.sym, node.var)
                else:
                    string = 'I=%d\tt=%.2f\tW=%s\n' % (
                        idx, node.entry/FRATE, node.sym)
                fh.write(string.encode())
                mapping[node] = idx
            for idx, arc in enumerate(self.arcs):
                string = 'J=%d\tS=%d\tE=%d\ta=%.2f\tl=%.3f' % (
                    idx,
                    mapping[arc.src],
                    mapping[arc.dest],
                    arc.ascr,
                    arc.lscr,
                )
                if arc.nscr:
                    string += '\tn=' + ','.join(
                        ['{:.3f}'.format(n) for n in arc.nscr])
                if arc.iscr:
                    string += '\ti=' + ','.join(
                        ['{:.3f}'.format(i) for i in arc.iscr])
                string += '\n'
                fh.write(string.encode())

    def dag2dot(self, file_path):
        with open(file_path, 'w') as fh:
            fh.write('digraph lattice {\n\trankdir=LR;\n')
            node_id = {}
            fh.write('\tnode [shape=circle];')
            for i, u in enumerate(self.nodes):
                node_id[u] = '\'[%d]%s/%d\'' % (i, u.sym, u.entry)
                if u != self.end:
                    fh.write(' %s' % node_id[u])
            fh.write(';\n\tnode [shape=doublecircle]; %s;\n\n'
                     % node_id[self.end])
            for x in self.arcs:
                label = 'a=%.2f,l=%.3f' % (x.ascr, x.lscr)
                if x.nscr:
                    label += ',n=' + ','.join(
                        ['{:.3f}'.format(n) for n in x.nscr])
                if x.iscr:
                    label += ',i=' + ','.join(
                        ['{:.3f}'.format(i) for i in x.iscr])
                fh.write('\t%s -> %s [label=\'%s\'];\n'
                         % (node_id[x.src], node_id[x.dest], label))
            fh.write('}\n')

    def num_nodes(self):
        """Return the number of nodes in the DAG.

        :return: Number of nodes in the DAG.
        :rtype: int
        """
        return len(self.nodes)

    def num_arcs(self):
        """Return the number of arcs in the DAG.

        :return: Number of arcs in the DAG.
        :rtype: int
        """
        return len(self.arcs)

    def sort_nodes(self):
        """Find start & end nodes, sort nodes & arcs by timestamps."""
        non_terminal_nodes = []
        for node in self.nodes:
            if not node.entries:
                assert self.start is None, (
                    'there are more than one node with no incoming arcs')
                self.start = node
            elif not node.exits:
                assert self.end is None, (
                    'there are more than one node with no outgoing arcs')
                self.end = node
            else:
                non_terminal_nodes.append(node)
        assert self.start is not None and self.end is not None, (
            'no start or end node')
        self.nodes = ([self.start]
                      + sorted(non_terminal_nodes,
                               key=lambda x: (x.entry, x.sym))
                      + [self.end])
        for n in self.nodes:
            n.exits.sort(key=lambda x: (x.dest.entry, x.dest.sym))

    def remove_nodes(self, nodes):
        """Remove dangling nodes recursively."""
        for node in nodes:
            for arc in node.entries:
                arc.src.exits.remove(arc)
                self.arcs.remove(arc)
            for arc in node.exits:
                arc.dest.entries.remove(arc)
                self.arcs.remove(arc)
            self.nodes.remove(node)
        dangling_nodes = []
        for node in self.nodes:
            if node == self.start or node == self.end:
                pass
            else:
                if not node.exits or not node.entries:
                    dangling_nodes.append(node)
        if dangling_nodes:
            self.remove_nodes(dangling_nodes)

    def remove_unk_oov(self):
        """Remove all nodes and arcs of UNK and OOV."""
        unk_oov_nodes = []
        for node in self.nodes:
            if node.sym in [UNK, OOV]:
                unk_oov_nodes.append(node)
        self.remove_nodes(unk_oov_nodes)

    def density(self):
        """Compute lattice density, i.e. number of arcs per second.

        :return: Lattice density.
        :rtype: float
        """
        return self.num_arcs() / (self.nframes / FRATE)

    def traverse_arcs_topo(self, start=None, end=None, reverse=False):
        """
        Traverse arcs in topological order (all predecessors to a given
        edge have been traversed before that edge);
        or in reversed topological order (all successors to a given
        edge have been traversed before that edge).
        """
        for w in self.nodes:
            w.fan = 0
        if start is None:
            start = self.start
        if end is None:
            end = self.end
        if not reverse:
            # forward topological order
            for x in self.arcs:
                x.dest.fan += 1
            # Agenda of closed arcs
            Q = start.exits[:]
            while Q:
                e = Q[0]
                del Q[0]
                yield e
                e.dest.fan -= 1
                if e.dest.fan == 0:
                    if e.dest == end:
                        break
                    Q.extend(e.dest.exits)
        else:
            # backward topological order
            for x in self.arcs:
                x.src.fan += 1
            Q = end.entries[:]
            while Q:
                e = Q[0]
                del Q[0]
                yield e
                e.src.fan -= 1
                if e.src.fan == 0:
                    if e.src == start:
                        break
                    Q.extend(e.src.entries)

    def forward(self, aw, lw):
        """
        Compute forward variable for all arcs in the lattice.
        Store alpha on each arc.
        """
        # This can be accelerated by storing alpha for nodes without recomputing
        for wx in self.traverse_arcs_topo(reverse=False):
            # If wx.src has no predecessors the previous alpha is 1.0
            if len(wx.src.entries) == 0:
                alpha = 0
            else:
                alpha = LOGZERO
                # For each predecessor node to wx.src
                for vx in wx.src.entries:
                    # Accumulate alpha for this arc
                    alpha = np.logaddexp(alpha, vx.alpha)
            wx.alpha = alpha + wx.ascr * aw + wx.lscr * lw

    def backward(self, aw, lw):
        """
        Compute backward variable for all arcs in the lattice.
        Store beta value on each arc.
        """
        # This can be accelerated by storing beta for nodes without recomputing
        for vx in self.traverse_arcs_topo(reverse=True):
            # Beta for arcs into </s> = 1.0
            if len(vx.dest.exits) == 0:
                beta = 0
            else:
                beta = LOGZERO
                # For each outgoing arc from vx.dest
                for wx in vx.dest.exits:
                    # Accumulate beta for this arc
                    beta = np.logaddexp(beta, wx.beta)
            # Update beta for this arc
            vx.beta = beta + vx.ascr * aw + vx.lscr * lw

    def posterior(self, aw=1.0, lw=1.0):
        """
        Compute arc posterior probabilities.
        Store posterior on each node and each arc.
        """
        # Clear alphas, betas, and posteriors
        for w in self.nodes:
            for wx in w.exits:
                wx.alpha = wx.beta = wx.post = LOGZERO
        # Run forward and backward algorithm
        self.forward(aw, lw)
        self.backward(aw, lw)
        # Sum over alpha for arcs entering the end node to get normaliser
        fwd_norm = LOGZERO
        for vx in self.end.entries:
            fwd_norm = np.logaddexp(fwd_norm, vx.alpha)
        # Sum over beta for arcs exiting the start node to get normaliser
        bwd_norm = LOGZERO
        for wx in self.start.exits:
            bwd_norm = np.logaddexp(bwd_norm, wx.beta)
        # Sanity check: relative difference of fwd & bwd norms
        if (fwd_norm - bwd_norm) / bwd_norm > 0.01:
            logging.warning('Forward %.8f disagrees with Backward %.8f'
                            % (fwd_norm, bwd_norm))
        # Iterate over all arcs and normalize
        for w in self.nodes:
            w.post = LOGZERO
            for wx in w.exits:
                wx.post = (wx.alpha + wx.beta - fwd_norm
                           - (wx.ascr * aw + wx.lscr * lw))
                w.post = np.logaddexp(w.post, wx.post)

    def entropy(self, aw=1.0, lw=1.0):
        """Compute lattice entropy."""
        self.posterior(aw=aw, lw=lw)
        # Clear all entropy values
        for w in self.nodes:
            w.entropy = None
        # Set the end node to have zero entropy
        self.end.entropy = 0
        # Loop through the graph in the reversed topological order
        for e in self.traverse_arcs_topo(reverse=True):
            w = e.src
            if w.entropy is None:
                try:
                    outgoing_arcs = [
                        (wx.post - w.post, wx.dest.entropy) for wx in w.exits]
                    w.entropy = sum([
                        np.exp(post)*(ent-post) for post, ent in outgoing_arcs])
                except TypeError:
                    pass
        return self.start.entropy

    def onebest(self, aw=1.0, lw=1.0, nw=[], iw=[], ip=0.0):
        """Find best path in the lattice using Viterbi algorithm."""
        if not hasattr(nw, '__len__'):
            nw = np.ones_like(self.arcs[0].nscr) * nw
        if not hasattr(iw, '__len__'):
            iw = np.ones_like(self.arcs[0].iscr) * iw
        # Clear node score and prev
        for w in self.nodes:
            w.score = LOGZERO
            w.prev = None
        self.start.score = 0
        # Run Viterbi from the start node
        for w in self.nodes[1:]:
            scores = [e.src.score + e.ascr * aw + e.lscr * lw - ip
                      + np.dot(e.nscr, nw)
                      + np.dot(e.iscr, iw) for e in w.entries]
            max_idx = np.argmax(scores)
            w.score = scores[max_idx]
            w.prev = w.entries[max_idx]
        # Backtrace
        end = self.end.prev
        best_path = []
        while end:
            best_path.append(end)
            end = end.src.prev
        best_path = list(reversed(best_path))
        return best_path

    def nbest(self, n, aw=1.0, lw=1.0, ip=0.0):
        """Find N-best paths in the lattice using Viterbi algorithm."""
        # Clear node score and prev
        for w in self.nodes:
            w.score = []
            w.prev = []
        self.start.score = [0]
        self.start.prev = [(None, None)]
        # Keep path with the highest score if same history exists.
        def remove_repetition(node, n):
            pruned_scores, pruned_prevs = [], []
            existing_hyps = set()
            for score, prev in zip(node.score, node.prev):
                # Backtrace
                arc, idx = prev
                hyp = []
                while arc:
                    hyp.append(arc.dest.sym)
                    arc, idx = arc.src.prev[idx]
                hyp = ' '.join(list(hyp))
                # Check existing history
                if hyp not in existing_hyps:
                    pruned_scores.append(score)
                    pruned_prevs.append(prev)
                    existing_hyps.add(hyp)
                # Cut off for nbest
                if len(pruned_scores) >= n:
                    break
            # Update attributes
            node.score = pruned_scores
            node.prev = pruned_prevs
        # Run Viterbi but keep top n paths & pointers
        for w in self.nodes[1:]:
            for e in w.entries:
                arc_score = e.ascr * aw + e.lscr * lw - ip
                w.score.extend([i + arc_score for i in e.src.score])
                w.prev.extend([(e, idx) for idx in range(len(e.src.prev))])
            w.score, w.prev = zip(*sorted(
                zip(w.score, w.prev), key=lambda x: x[0], reverse=True))
            remove_repetition(w, n)
        # Backtrace
        best_paths = []
        for end_ in self.end.prev:
            arc, idx = end_
            best_path = []
            while arc:
                best_path.append(arc)
                arc, idx = arc.src.prev[idx]
            best_paths.append(list(reversed(best_path)))
        return best_paths

    def in_lattice(self, ref):
        """Check wheather ref sequence is in the lattice."""
        assert ref[0] == self.start.sym, 'The first word is not null.'
        cur_node = set([self.start])
        for word in ref[1:]:
            next_node = set()
            for i in cur_node:
                for j in i.exits:
                    if word == j.dest.sym:
                        next_node.add(j.dest)
            if not next_node:
                return False
            else:
                cur_node = next_node
        if sum([i == self.end for i in cur_node]) == 0:
            return False
        return True

    def oracle_wer(self, ref):
        """Compute the oracle word error rate of a lattice.

        :param ref: Reference list of strings (without start/end of sentence).
        :type ref: list
        :return: Word errors, alignment of oracle string to reference.
        :rtype: tuple
        """
        # Add start and end to ref
        ref = [NULL, SOS] + ref.split() + [EOS]
        # Most lattices contain the correct path, so check that first
        if self.in_lattice(ref):
            return (0, [(i, i) for i in ref])
        # Initialize the alignment matrix
        align_matrix = np.ones((len(ref),len(self.nodes)), 'i') * MAXINT
        # And the backpointer matrix
        bp_matrix = np.zeros((len(ref),len(self.nodes)), 'O')
        # Figure out the minimum distance to each node from the start
        # of the lattice, and construct a node to ID mapping
        nodeid = {}
        for i,u in enumerate(self.nodes):
            u.score = MAXINT
            nodeid[u] = i
        self.start.score = 1
        for u in self.nodes:
            for x in u.exits:
                dist = u.score + 1
                if dist < x.dest.score:
                    x.dest.score = dist
        def find_pred(ii, jj):
            bestscore = MAXINT
            bestp = -1
            if len(self.nodes[jj].entries) == 0:
                return bestp
            for e in self.nodes[jj].entries:
                k = nodeid[e.src]
                if align_matrix[ii,k] < bestscore:
                    bestp = k
                    bestscore = align_matrix[ii,k]
            return bestp
        # Now fill in the alignment matrix
        for i, w in enumerate(ref):
            for j, u in enumerate(self.nodes):
                # Insertion = cost(w, prev(u)) + 1
                if u == self.start: # start node
                    bestp = -1
                    inscost = i + 2 # Distance from start of ref
                else:
                    # Find best predecessor in the same reference position
                    bestp = find_pred(i, j)
                    inscost = align_matrix[i,bestp] + 1
                # Deletion  = cost(prev(w), u) + 1
                if i == 0: # start symbol
                    delcost = u.score + 1 # Distance from start of hyp
                else:
                    delcost = align_matrix[i-1,j] + 1
                # Substitution = cost(prev(w), prev(u)) + (w != u)
                if i == 0 and bestp == -1: # Start node, start of ref
                    subcost = int(w != u.sym)
                elif i == 0: # Start of ref
                    subcost = (self.nodes[bestp].score
                               + int(w != u.sym))
                elif bestp == -1: # Start node
                    subcost = i - 1 + int(w != u.sym)
                else:
                    # Find best predecessor in the previous reference position
                    bestp = find_pred(i-1, j)
                    subcost = (align_matrix[i-1,bestp]
                               + int(w != u.sym))
                align_matrix[i,j] = min(subcost, inscost, delcost)
                # Now find the argmin
                if align_matrix[i,j] == subcost:
                    bp_matrix[i,j] = (i-1, bestp)
                elif align_matrix[i,j] == inscost:
                    bp_matrix[i,j] = (i, bestp)
                else:
                    bp_matrix[i,j] = (i-1, j)
        # Find last node's index
        last = nodeid[self.end]
        # Backtrace to get an alignment
        i = len(ref)-1
        j = last
        bt = []
        while True:
            ip,jp = bp_matrix[i,j]
            if ip == i: # Insertion
                bt.append(('**INS**', '*%s*' % self.nodes[j].sym))
            elif jp == j: # Deletion
                bt.append(('*%s' % ref[i], '**DEL**'))
            else:
                if ref[i] == self.nodes[j].sym:
                    bt.append((ref[i], self.nodes[j].sym))
                else:
                    bt.append((ref[i], '*%s*' % self.nodes[j].sym))
            # If we consume both ref and hyp, we are done
            if ip == -1 and jp == -1:
                break
            # If we hit the beginning of the ref, fill with insertions
            if ip == -1:
                while True:
                    bt.append(('**INS**', self.nodes[jp].sym))
                    bestp = find_pred(i,jp)
                    if bestp == -1:
                        break
                    jp = bestp
                break
            # If we hit the beginning of the hyp, fill with deletions
            if jp == -1:
                while ip >= 0:
                    bt.append((ref[ip], '**DEL**'))
                    ip = ip - 1
                break
            # Follow the pointer
            i,j = ip,jp
        bt.reverse()
        return align_matrix[len(ref)-1,last], bt
