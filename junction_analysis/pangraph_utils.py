from collections import defaultdict, Counter
from typing import Optional, Iterable, List, Tuple, Union

class Node:
    """Combination of block id and strandedness, node id can be added optionally but equality doesn't depend on it."""

    def __init__(self, bid: str, strand: bool, nid: Optional[str] = None) -> None:
        self.id = bid
        self.strand = strand
        self.nid = nid

    def invert(self) -> "Node":
        return Node(self.id, not self.strand, self.nid)

    def __eq__(self, other: object) -> bool:
        return self.id == other.id and self.strand == other.strand

    def __hash__(self) -> int:
        return hash((self.id, self.strand))

    def __repr__(self) -> str:
        s = "+" if self.strand else "-"
        return f"[{self.id}|{s}]"

    def to_str_id(self):
        s = "f" if self.strand else "r"
        return f"{self.id}_{s}"

    @staticmethod
    def from_str_id(t) -> "Node":
        bid = t.split("_")[0]
        strand = True if t.split("_")[1] == "f" else False
        return Node(bid, strand)
    
class DeduplicatedNode:
    """Combination of block id, strandedness and context in path (closest, non-duplicated block to the left), node id can be added optionally but equality doesn't depend on it."""

    def __init__(self, bid: str, strand: bool, context: str, nid: Optional[str] = None,) -> None:
        self.id = bid
        self.strand = strand
        self.context = context
        self.nid = nid

    # for now context is always block that is never inverted, that's why we can just keep the context in the case of an inversion
    def invert(self) -> "DeduplicatedNode":
        return DeduplicatedNode(self.id, not self.strand, self.context, self.nid)

    def __eq__(self, other: object) -> bool:
        return self.id == other.id and self.strand == other.strand and self.context == other.context

    def __hash__(self) -> int:
        return hash((self.id, self.strand, self.context))

    def __repr__(self) -> str:
        s = "+" if self.strand else "-"
        return f"[{self.id}|{s}|{self.context}]"

    def to_str_id(self):
        s = "f" if self.strand else "r"
        return f"{self.id}_{s}_{self.context}"

    @staticmethod
    def from_str_id(t) -> "DeduplicatedNode":
        bid, strand, context = t.split("_")
        strand = True if strand == "f" else False
        return DeduplicatedNode(bid, strand, context)


class Path:
    """A path is a list of nodes"""

    def __init__(self, nodes=None) -> None:
        self.nodes = nodes if nodes is not None else []

    def __iter__(self):
        return iter(self.nodes)

    def add_left(self, node: None) -> None:
        self.nodes.insert(0, node)

    def add_right(self, node: None) -> None:
        self.nodes.append(node)

    def invert(self) -> "Path":
        return Path([n.invert() for n in self.nodes[::-1]])

    # TODO: add check for context dependent paths to consider a path equal to its fully inverted path
    def __eq__(self, o: object) -> bool:
        return self.nodes == o.nodes

    def __hash__(self) -> int:
        return hash(tuple(self.nodes))

    def __repr__(self) -> str:
        return "_".join([str(n) for n in self.nodes])

    def __len__(self) -> int:
        return len(self.nodes)

    def to_list(self):
        return [n.to_str_id() for n in self.nodes]

    @staticmethod
    def from_list(path_list, type = 'node') -> "Path":
        if type == 'deduplicatednode':
            return Path([DeduplicatedNode.from_str_id(nid) for nid in path_list])
        elif type == 'node':
            return Path([Node.from_str_id(nid) for nid in path_list])
    
    @staticmethod
    def from_tuple_list(path_list, type: str = "node") -> "Path":
        """
        Accepts tuples with or without nid:
        - Node: (bid, strand) or (bid, strand, nid)
        - DeduplicatedNode: (bid, strand, context) or (bid, strand, context, nid)
        """
        nodes = []
        if type == "deduplicatednode":
            for t in path_list:
                if len(t) == 3:
                    bid, strand, context = t
                    nid = None
                elif len(t) == 4:
                    bid, strand, context, nid = t
                else:
                    raise ValueError("DeduplicatedNode tuple must have 3 or 4 elements")
                nodes.append(DeduplicatedNode(bid, strand, context, nid))
        elif type == "node":
            for t in path_list:
                if len(t) == 2:
                    bid, strand = t
                    nid = None
                elif len(t) == 3:
                    bid, strand, nid = t
                else:
                    raise ValueError("Node tuple must have 2 or 3 elements")
                nodes.append(Node(bid, strand, nid))
        else:
            raise ValueError(f"Unknown node type: {type}")

        return Path(nodes)


class Edge:
    """Oriented link between two nodes/paths"""

    def __init__(self, left, right) -> None:
        self.left = left
        self.right = right

    def invert(self) -> "Edge":
        return Edge(self.right.invert(), self.left.invert())

    def __side_eq__(self, o: object) -> bool:
        # comparison of edges doesn't depend on context
        return self.left.id == o.left.id and self.left.strand == o.left.strand and self.right.id == o.right.id and self.right.strand == o.right.strand

    def __eq__(self, o: object) -> bool:
        return self.__side_eq__(o) or self.__side_eq__(o.invert())

    def __side_hash__(self) -> int:
        return hash((self.left.id, self.left.strand, self.right.id, self.right.strand))

    def __hash__(self) -> int:
        return self.__side_hash__() ^ self.invert().__side_hash__()

    def __repr__(self) -> str:
        return f"{self.left} <--> {self.right}"

    def __to_str_id(self) -> str:
        return "__".join([self.left.to_str_id(), self.right.to_str_id()])

    def to_str_id(self) -> str:
        A = self.__to_str_id()
        B = self.invert().__to_str_id()
        return A if A < B else B

    @staticmethod
    def from_str_id(t) -> "Edge":
        left, right = t.split("__")
        if len(left.split("_") == 3):
            return Edge(DeduplicatedNode.from_str_id(left), DeduplicatedNode.from_str_id(right))
        else:
            return Edge(Node.from_str_id(left), Node.from_str_id(right))


class Junction:
    """A junction is a combination of a node (or path) flanked by two nodes, with reverse-complement simmetry"""

    def __init__(self, left: Node, center: Path, right: Node) -> None:
        self.left = left
        self.center = center
        self.right = right

    def invert(self) -> "Junction":
        return Junction(self.right.invert(), self.center.invert(), self.left.invert())

    def flanks_bid(self, bid) -> bool:
        return (self.left.id == bid) or (self.right.id == bid)

    def __side_eq__(self, o: object) -> bool:
        return self.left == o.left and self.center == o.center and self.right == o.right

    def __eq__(self, o: object) -> bool:
        return self.__side_eq__(o) or self.__side_eq__(o.invert())

    def __side_hash__(self) -> int:
        return hash((self.left, self.center, self.right))

    def __hash__(self) -> int:
        return self.__side_hash__() ^ self.invert().__side_hash__()

    def __repr__(self) -> str:
        return f"{self.left} <-- {self.center} --> {self.right}"

    def to_list(self):
        return [self.left.to_str_id(), self.center.to_list(), self.right.to_str_id()]

    def flanking_edge(self) -> Edge:
        return Edge(self.left, self.right)

    @staticmethod
    def from_list(t) -> "Junction":
        return Junction(
            Node.from_str_id(t[0]), Path.from_list(t[1]), Node.from_str_id(t[2])
        )


def pangraph_to_path_dict(pan):
    """Creates a dictionary isolate -> path objects"""
    res = {}
    for path in pan.paths:
        name = path.name
        B = path.block_ids
        S = path.block_strands
        nodes = [Node(b, s) for b, s in zip(B, S)]
        res[name] = Path(nodes)
    return res


def filter_paths(paths, keep_f):
    """Given a filter function, removes nodes that fail the condition from
    the path dictionaries."""
    res = {}
    for iso, path in paths.items():
        filt_path = Path([node for node in path.nodes if keep_f(node.id)])
        res[iso] = filt_path
    return res


def path_categories(paths):
    """Returns a list of touples, one per non-empty path, with the following info:
    (count, path, [list of isolates])"""
    iso_list = defaultdict(list)
    n_paths = defaultdict(int)
    nodes = {}
    for iso, path in paths.items():
        if len(path.nodes) > 0:
            n_paths[path] += 1
            iso_list[path].append(iso)
            nodes[path] = path.nodes

    # sort by count
    path_cat = [(count, nodes[path], iso_list[path]) for path, count in n_paths.items()]
    path_cat.sort(key=lambda x: x[0], reverse=True)
    return path_cat


def path_junction_split(path, is_core):
    """Given a path and a boolean function of node ids, it splits the path in a set of
    Junctions, with flanking "core" blocks for which the condition is true."""
    junctions = []

    current = []
    left_node = None
    for node in path.nodes:
        if is_core(node.id):
            J = Junction(left_node, Path(current), node)
            junctions.append(J)
            left_node = node
            current = []
        else:
            current.append(node)

    # complete periodic boundary
    J = junctions[0]
    J.left = left_node
    J.center = Path(current + J.center.nodes)
    junctions[0] = J

    return junctions


def path_edge_count(paths):
    """Count internal edges of paths"""
    ct = Counter()
    for iso, p in paths.items():
        L = len(p.nodes)
        for i in range(L):
            e = Edge(p.nodes[i], p.nodes[(i + 1) % L])
            ct.update([e])
    return dict(ct)


def path_block_count(paths):
    """Count internal blocks of paths"""
    ct = Counter()
    for iso, p in paths.items():
        for node in p.nodes:
            ct.update([node.id])
    return dict(ct)


def find_mergers(paths):
    """Create a dictionary source -> sinks of block-ids to be merged"""
    edge_ct = path_edge_count(paths)
    block_ct = path_block_count(paths)

    mergers = {}
    for e, ec in edge_ct.items():
        bl, br = e.left.id, e.right.id
        if (ec == block_ct[bl]) and (ec == block_ct[br]):
            # merge
            if bl in mergers:
                if br in mergers:
                    source = mergers[br]
                    sink = mergers[bl]
                    for k in mergers:
                        if mergers[k] == source:
                            mergers[k] = sink
                else:
                    mergers[br] = mergers[bl]
            elif br in mergers:
                mergers[bl] = mergers[br]
            else:
                mergers[br] = bl
                mergers[bl] = bl
    return mergers