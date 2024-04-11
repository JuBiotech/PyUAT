"""Module containing tracking metric computation"""

import networkx as nx
import numpy as np


def compute_positive_division(
    pred_lc_lineage: nx.DiGraph, gt_lc_lineage: nx.DiGraph
) -> tuple:
    """Computes the counts of true positive (tp) and false positive (fp) cell divisions

    Args:
        pred_lc_lineage (nx.DiGraph): predicted life cycle lineage (every cell track is one node)
        gt_lc_lineage (nx.DiGraph): ground truth life cycle lineage (every cell track is one node)

    Returns:
        tuple: true-positive (tp) and false positive (fp) counts
    """

    tp = 0
    fp = 0

    # create cycle lookup dict
    gt_lc_lookup = {}
    for gt_lc_node in gt_lc_lineage:
        for gt_node in gt_lc_lineage.nodes[gt_lc_node]["cycle"]:
            gt_lc_lookup[gt_node] = gt_lc_node

    for pred_node in pred_lc_lineage.nodes:
        pred_successors = list(pred_lc_lineage.successors(pred_node))
        pred_cycle = pred_lc_lineage.nodes[pred_node]["cycle"]

        if len(pred_successors) > 1:
            # this is a division

            # identify the respective gt node by using the last
            gt_node = gt_lc_lookup[pred_cycle[-1]]
            gt_cycle = gt_lc_lineage.nodes[gt_node]["cycle"]
            gt_successors = list(gt_lc_lineage.successors(gt_node))

            # direct parent (last cell detection in the cycle) must be the same
            match = True

            match &= pred_cycle[-1] == gt_cycle[-1]

            # direct children (first cell detection in the cycle) must be the same
            pred_direct_children = {
                pred_lc_lineage.nodes[succ]["cycle"][0] for succ in pred_successors
            }
            gt_direct_children = {
                gt_lc_lineage.nodes[succ]["cycle"][0] for succ in gt_successors
            }

            match &= pred_direct_children == gt_direct_children

            if match:
                # same division exists in gt
                tp += 1
            else:
                # division does not exist in gt
                fp += 1

    return tp, fp


def compute_bc_stats(pred_lc_lineage: nx.DiGraph, gt_lc_lineage: nx.DiGraph) -> dict:
    """Compute the branch correctness (bc) statistics

    Args:
        pred_lc_lineage (nx.DiGraph): predicted life cycle lineage (every cell track is one node)
        gt_lc_lineage (nx.DiGraph): ground truth life cycle lineage (every cell track is one node)

    Returns:
        dict: dictionary with true positive (tp), false positive (fp) and false negative (fn) counts
    """

    tp = 0
    fp = 0
    fn = 0

    # compare pred divisions with gt divisions
    tp, fp = compute_positive_division(
        pred_lc_lineage=pred_lc_lineage, gt_lc_lineage=gt_lc_lineage
    )
    # do it the other way round
    tp2, fn = compute_positive_division(
        pred_lc_lineage=gt_lc_lineage, gt_lc_lineage=pred_lc_lineage
    )

    # make sure that tps agree
    np.testing.assert_equal(tp, tp2)

    return dict(tp=tp, fp=fp, fn=fn)


def compute_positive_cts(pred_lc_lineage, gt_lc_lineage) -> tuple:
    """Computes the counts of true positive (tp) and false positive (fp) cell tracks (cts)

    Args:
        pred_lc_lineage (nx.DiGraph): predicted life cycle lineage (every cell track is one node)
        gt_lc_lineage (nx.DiGraph): ground truth life cycle lineage (every cell track is one node)

    Returns:
        tuple: tp and fp counts
    """
    tp = 0
    fp = 0

    gt_lc_lookup = {}
    for gt_lc_node in gt_lc_lineage:
        for gt_node in gt_lc_lineage.nodes[gt_lc_node]["cycle"]:
            gt_lc_lookup[gt_node] = gt_lc_node

    for cell_cycle_node in pred_lc_lineage.nodes:
        pred_cc = list(pred_lc_lineage.nodes[cell_cycle_node]["cycle"])

        match = False

        # get the gt cell cycle by the first node
        gt_cc_node = gt_lc_lookup[pred_cc[0]]
        gt_cc = list(gt_lc_lineage.nodes[gt_cc_node]["cycle"])

        if pred_cc == gt_cc:
            # they cover the same cell track

            match = True

        if match is False:
            # no matching cell track was found -> it is a false positive
            fp += 1
        else:
            # matching cell track was found!
            tp += 1

    return tp, fp


def compute_ct_stats(pred_lc_lineage, gt_lc_lineage):
    """Compute the correct cell track (bc) statistics. A cell track is correct if from cell birth to cell division/death/disappearance all edges are correctly reconstructed. It does not deal with division edges.

    Args:
        pred_lc_lineage (nx.DiGraph): predicted life cycle lineage (every cell track is one node)
        gt_lc_lineage (nx.DiGraph): ground truth life cycle lineage (every cell track is one node)

    Returns:
        _type_: _description_
    """
    tp = 0
    fp = 0
    fn = 0

    # compare gt tracks with predicition
    tp, fp = compute_positive_cts(
        pred_lc_lineage=pred_lc_lineage, gt_lc_lineage=gt_lc_lineage
    )
    # compare pred tracks with gt
    tp2, fn = compute_positive_cts(
        pred_lc_lineage=gt_lc_lineage, gt_lc_lineage=pred_lc_lineage
    )

    # make sure the tps agree
    np.testing.assert_equal(tp, tp2)

    return dict(tp=tp, fp=fp, fn=fn)


def compute_f1(tp: int, fp: int, fn: int) -> dict:
    """Compute f1 score from count statistics

    Args:
        tp (int): true positive (tp) count
        fp (int): false positive (fp) count
        fn (int): false negative (fn) count

    Returns:
        dict: dictionary with precision, recall and f1 scores
    """
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 / (1 / precision + 1 / recall)

    return dict(precision=precision, recall=recall, f1=f1)
