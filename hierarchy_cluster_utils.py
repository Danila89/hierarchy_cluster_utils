import numpy as np
from scipy.cluster.hierarchy import leaders, ClusterNode, to_tree
from typing import Optional, Tuple, List


def get_node(
    linkage_matrix: np.ndarray,
    clusters_array: np.ndarray,
    cluster_num: int
) -> ClusterNode:
    """
    Returns ClusterNode (the node of the cluster tree) corresponding to the given cluster number.
    :param linkage_matrix: linkage matrix
    :param clusters_array: array of cluster numbers for each point
    :param cluster_num: id of cluster for which we want to get ClusterNode
    :return: ClusterNode corresponding to the given cluster number
    """
    L, M = leaders(linkage_matrix, clusters_array)
    idx = L[M == cluster_num]
    tree = to_tree(linkage_matrix)
    result = search_for_node(tree, idx)
    assert result
    return result


def search_for_node(
    cur_node: Optional[ClusterNode],
    target: int
) -> Optional[ClusterNode]:
    """
    Searches for the node with the given id of the cluster in the given subtree.
    :param cur_node: root of the cluster subtree to search for target node
    :param target: id of the target node (cluster)
    :return: ClusterNode with the given id if it exists in the subtree, None otherwise
    """
    if cur_node is None:
        return None
    if cur_node.get_id() == target:
        return cur_node
    left = search_for_node(cur_node.get_left(), target)
    if left:
        return left
    return search_for_node(cur_node.get_right(), target)


def dfs_get_parent_for_node(
    root: ClusterNode,
    node: ClusterNode
) -> Optional[ClusterNode]:
    """
    Returns parent of the given ClusterNode.
    :param root: root of the cluster tree
    :param node: ClusterNode for which we want to get parent
    :return: parent of the given ClusterNode
    """
    if root is None or root.is_leaf():
        return None
    if root.get_left().get_id() == node.get_id() or root.get_right().get_id() == node.get_id():
        return root
    left = dfs_get_parent_for_node(root.get_left(), node)
    if left:
        return left
    return dfs_get_parent_for_node(root.get_right(), node)


def get_parent_and_sibling_for_node(
    linkage_matrix: np.ndarray,
    node: ClusterNode
) -> Tuple[Optional[ClusterNode], Optional[ClusterNode]]:
    """
    Returns parent and sibling of the given ClusterNode.
    :param linkage_matrix: linkage matrix
    :param node: ClusterNode for which we want to get parent and sibling
    :return: tuple of two ClusterNodes: parent and sibling
    """
    parent = dfs_get_parent_for_node(to_tree(linkage_matrix), node)
    if parent is None:  # node is root
        return None, None
    sibling = None
    if parent.get_left().get_id() == node.get_id():
        sibling = parent.get_right()
    elif parent.get_right().get_id() == node.get_id():
        sibling = parent.get_left()
    return parent, sibling


def get_leaves_ids(node: ClusterNode) -> List[int]:
    """
    Returns ids of all samples (leaf nodes) that belong to the given ClusterNode (belong to the node's subtree).
    :param node: ClusterNode for which we want to get ids of samples
    :return: list of ids of samples that belong to the given ClusterNode
    """
    res = []

    def dfs(cur: Optional[ClusterNode]):
        if cur is None:
            return
        if cur.is_leaf():
            res.append(cur.get_id())
            return
        dfs(cur.get_left())
        dfs(cur.get_right())
    dfs(node)
    return res


def get_distances_and_counts(
    linkage_matrix: np.ndarray,
    clusters_array: np.ndarray
) -> Tuple[List[float], List[int]]:
    """
    Calculates intracluster distances and cluster sizes for each cluster according to the given linkage matrix.
    :param linkage_matrix: linkage matrix
    :param clusters_array: array of cluster numbers for each point
    :return: tuple of two lists: intracluster distances and cluster sizes
    """
    L, M = leaders(linkage_matrix, clusters_array)
    distances = {}
    counts = {}
    tree = to_tree(linkage_matrix)
    for i in range(len(L)):
        node = search_for_node(tree, L[i])
        distances[M[i]] = node.dist
        counts[M[i]] = node.get_count()
    return [distances[i] for i in clusters_array], [counts[i] for i in clusters_array]
