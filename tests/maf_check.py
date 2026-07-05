"""Independent Maximum-Agreement-Forest checker and oracle for PACE 2026.

This module does NOT share any code with the C++ solver, so it can be used to
catch the two failure modes that disqualify an exact-track submission:

  * INFEASIBLE output  -> validate_forest() rejects it.
  * SUBOPTIMAL output  -> brute_force_optimum() gives the true optimum on small
                          instances for comparison.

Leaf labels are the integers 1..n (each exactly once), encoded internally as
0-based bit indices in a Python int bitmask.

Definitions used (rooted binary MAF):
  A set of components (a partition of the leaves) is a valid agreement forest iff
    1. the components partition the whole leaf set,
    2. every component induces the SAME rooted topology in every input tree
       (children unordered, degree-<=1 vertices suppressed), and
    3. the components' minimal spanning subtrees are edge-disjoint in each tree
       (for binary trees this is equivalent to node-disjointness).
The optimum is the minimum number of components.
"""

import re
import sys

# Newick of a deep caterpillar recurses to depth ~n; allow large real instances.
sys.setrecursionlimit(1_000_000)


class Tree:
    """A rooted binary phylogenetic tree parsed from a Newick string.

    Nodes are numbered in creation order; because the parser builds children
    before their parent, iterating node ids 0..N-1 is a valid post-order.
    """

    def __init__(self, newick, n):
        self.n = n
        self.left = []
        self.right = []
        self.label = []  # 0-based leaf label, or -1 for internal nodes
        s = newick.strip()
        if s.endswith(";"):
            s = s[:-1]
        self._s = s
        self._pos = 0
        self.root = self._parse_node()
        if self._pos != len(self._s):
            raise ValueError(f"unparsed Newick suffix: {self._s[self._pos:]!r}")
        self.N = len(self.left)
        # Leaf-set bitmask per node, computed in post-order.
        self.ls = [0] * self.N
        for v in range(self.N):
            if self.left[v] == -1:
                self.ls[v] = 1 << self.label[v]
            else:
                self.ls[v] = self.ls[self.left[v]] | self.ls[self.right[v]]

    def _new_leaf(self, lab):
        i = len(self.left)
        self.left.append(-1)
        self.right.append(-1)
        self.label.append(lab)
        return i

    def _new_internal(self, l, r):
        i = len(self.left)
        self.left.append(l)
        self.right.append(r)
        self.label.append(-1)
        return i

    def _parse_node(self):
        s = self._s
        if self._pos >= len(s):
            raise ValueError("truncated Newick")
        if s[self._pos] == "(":
            self._pos += 1
            l = self._parse_node()
            if self._pos >= len(s) or s[self._pos] != ",":
                raise ValueError("expected ','")
            self._pos += 1
            r = self._parse_node()
            if self._pos >= len(s) or s[self._pos] != ")":
                raise ValueError("expected ')'")
            self._pos += 1
            return self._new_internal(l, r)
        j = self._pos
        while self._pos < len(s) and s[self._pos].isdigit():
            self._pos += 1
        if j == self._pos:
            raise ValueError(f"expected leaf label at {self._pos}")
        return self._new_leaf(int(s[j:self._pos]) - 1)

    def canon(self, mask):
        """Canonical rooted topology of this tree restricted to leaf-set `mask`.

        Returns a hashable nested tuple; two trees agree on a component iff their
        canon values are equal. Children are sorted so ordering is irrelevant and
        unary (degree-<=1) vertices are suppressed.
        """
        c = [None] * self.N
        for v in range(self.N):  # post-order
            if self.left[v] == -1:
                lab = self.label[v]
                c[v] = ("L", lab) if (mask >> lab) & 1 else None
            else:
                a = c[self.left[v]]
                b = c[self.right[v]]
                if a is None:
                    c[v] = b
                elif b is None:
                    c[v] = a
                else:
                    c[v] = ("N",) + tuple(sorted((a, b)))
        return c[self.root]

    def _lca(self, mask):
        v = self.root
        while self.left[v] != -1:
            li = self.ls[self.left[v]] & mask
            ri = self.ls[self.right[v]] & mask
            if li and ri:
                return v
            v = self.left[v] if li else self.right[v]
        return v

    def edge_set(self, mask):
        """Edge ids (child node ids) of the minimal subtree spanning `mask`."""
        if mask & (mask - 1) == 0:  # 0 or 1 leaf -> no edges
            return frozenset()
        es = set()
        stack = [self._lca(mask)]
        while stack:
            v = stack.pop()
            for ch in (self.left[v], self.right[v]):
                if ch != -1 and (self.ls[ch] & mask):
                    es.add(ch)
                    stack.append(ch)
        return frozenset(es)


def parse_instance(text):
    """Parse a PACE 2026 instance. Returns (n, t, [Tree, ...])."""
    n = t = None
    trees = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if len(line) >= 2 and line[1] == "p":
                parts = line[2:].split()
                t, n = int(parts[0]), int(parts[1])
            # #a / #s / #x and comments are ignored by an exact-track solver
            continue
        if n is None:
            raise ValueError("tree line before #p header")
        trees.append(Tree(line, n))
    if n is None:
        raise ValueError("missing #p header")
    if len(trees) != t:
        raise ValueError(f"#p says {t} trees, found {len(trees)}")
    return n, t, trees


def output_to_masks(n, output_text):
    """Turn solver stdout into a list of component leaf-set bitmasks.

    Each non-comment line containing digits is treated as one component; the
    leaves are the integers appearing on that line.
    """
    masks = []
    for raw in output_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        labels = [int(x) for x in re.findall(r"\d+", line)]
        if not labels:
            continue
        mask = 0
        for lab in labels:
            if lab < 1 or lab > n:
                raise ValueError(f"label {lab} out of range 1..{n}")
            mask |= 1 << (lab - 1)
        masks.append(mask)
    return masks


def validate_forest(n, trees, comps):
    """Check that `comps` (list of bitmasks) is a valid agreement forest.

    Returns (ok: bool, reason: str).
    """
    full = (1 << n) - 1
    covered = 0
    for c in comps:
        if c == 0:
            return False, "empty component"
        if c & covered:
            return False, "components overlap (not a partition)"
        covered |= c
    if covered != full:
        return False, "components do not cover all leaves"

    # (2) topology agreement across trees, per component
    for c in comps:
        base = trees[0].canon(c)
        for tr in trees[1:]:
            if tr.canon(c) != base:
                return False, "component topology disagrees across trees"

    # (3) edge-disjointness within each tree
    for tr in trees:
        used = set()
        for c in comps:
            e = tr.edge_set(c)
            if e & used:
                return False, "components share an edge in some tree"
            used |= e
    return True, "ok"


def _set_partitions(items):
    if not items:
        yield []
        return
    first = items[0]
    for rest in _set_partitions(items[1:]):
        for i in range(len(rest)):
            yield rest[:i] + [[first] + rest[i]] + rest[i + 1:]
        yield [[first]] + rest


def brute_force_optimum(n, trees):
    """True MAF optimum by exhaustive partition search. Only for small n."""
    best = n  # singletons are always a valid forest
    for part in _set_partitions(list(range(n))):
        if len(part) >= best:
            continue
        comps = [sum(1 << x for x in blk) for blk in part]
        ok, _ = validate_forest(n, trees, comps)
        if ok:
            best = len(part)
    return best
