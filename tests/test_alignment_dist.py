import sys
import os
import numpy as np
import pytest

# ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from junction_analysis import block_alignment as ba

# -------- avg_pairwise_distance tests --------

def test_avg_pairwise_distance_identical_sequences():
    """All sequences identical -> distance 0."""
    seqs = ["AAA", "AAA", "AAA"]
    arr = np.array([list(s) for s in seqs])
    arr = np.char.upper(arr)

    d = ba.avg_pairwise_distance(arr)
    assert d == 0.0


def test_avg_pairwise_distance_simple_case():
    """
    Seqs: AAA, AAT, ATT

    Pairwise distances (ignoring gaps, only A/C/G/T):
      S1 vs S2: AAA vs AAT -> 1 mismatch / 3 = 1/3
      S1 vs S3: AAA vs ATT -> 2 mismatches / 3 = 2/3
      S2 vs S3: AAT vs ATT -> 1 mismatch / 3 = 1/3

    Average = (1/3 + 2/3 + 1/3) / 3 = 4/9
    """
    seqs = ["AAA", "AAT", "ATT"]
    arr = np.array([list(s) for s in seqs])
    arr = np.char.upper(arr)

    d = ba.avg_pairwise_distance(arr)
    expected = 4.0 / 9.0
    assert np.isclose(d, expected)


def test_avg_pairwise_distance_ignores_gaps_and_N():
    """
    Gaps and N should be ignored.

    Seqs: A-C, ATC
      pos0: A vs A -> valid, match
      pos1: - vs T -> ignore (gap)
      pos2: C vs C -> valid, match

    Distance = 0 / 2 = 0
    """
    seqs = ["A-C", "ATC"]
    arr = np.array([list(s) for s in seqs])
    arr = np.char.upper(arr)

    d = ba.avg_pairwise_distance(arr)
    assert np.isclose(d, 0.0)

    # also check that N is ignored and does not blow up
    seqs_with_N = ["AN", "AT"]
    arrN = np.array([list(s) for s in seqs_with_N])
    arrN = np.char.upper(arrN)

    dN = ba.avg_pairwise_distance(arrN)
    # only position 0 is valid (A vs A), so distance 0
    assert np.isclose(dN, 0.0)


# -------- avg_distance_to_consensus tests --------

def test_avg_distance_to_consensus_basic():
    """
    Seqs: AAA, AAT, ATT
    With ba.calc_consensus_seq, consensus should be "AAT".

    Distances to consensus:
      AAA vs AAT -> 1/3
      AAT vs AAT -> 0
      ATT vs AAT -> 1/3

    Average = (1/3 + 0 + 1/3) / 3 = 2/9
    """
    seqs = ["AAA", "AAT", "ATT"]
    arr = np.array([list(s) for s in seqs])
    arr = np.char.upper(arr)

    cons = ba.calc_consensus_seq(arr)
    assert cons == "AAT"

    d = ba.avg_distance_to_consensus(arr, cons)
    expected = 2.0 / 9.0
    assert np.isclose(d, expected)


def test_avg_distance_to_consensus_ignores_gaps():
    """
    Seqs: A-C, ATC
    consensus (plurality) = "ATC"

    Distances (ignoring gaps in seq or consensus):
      A-C vs ATC:
        pos0: A vs A -> valid, match
        pos1: - vs T -> ignored
        pos2: C vs C -> valid, match
        -> 0 mismatches / 2 valid = 0

      ATC vs ATC -> 0

    Average = 0
    """
    seqs = ["A-C", "ATC"]
    arr = np.array([list(s) for s in seqs])
    arr = np.char.upper(arr)

    cons = ba.calc_consensus_seq(arr)
    assert cons == "ATC"

    d = ba.avg_distance_to_consensus(arr, cons)
    assert np.isclose(d, 0.0)


def test_avg_distance_to_consensus_all_identical():
    """If all sequences equal to consensus, distance must be 0."""
    seqs = ["ACGT", "ACGT", "ACGT", "ACGT"]
    arr = np.array([list(s) for s in seqs])
    arr = np.char.upper(arr)

    cons = ba.calc_consensus_seq(arr)
    d = ba.avg_distance_to_consensus(arr, cons)

    assert cons == "ACGT"
    assert d == 0.0
