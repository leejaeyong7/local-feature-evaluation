import torch

def pdist(descs1, descs2):
    '''

    Given two matrix of shape N1xC, N2xC, compute pairwise distance

    Args:
        descs1 (torch.Tensor): N1 x C array, N1 descriptors of size C
        descs2 (torch.Tensor): N2 x C array, N2 descriptors of size C

    Returns:
        (torch.Tensor): N1 x N2 distance matrix (squared distance)
    '''

    N1, C = descs1.shape
    N2, C2 = descs2.shape

    # sanity check
    assert C == C2

    diffs = descs1.view(N1, 1, C) - descs2.view(1, N2, C)
    return (diffs ** 2).sum(2)

def match_descriptors(descs1, descs2, max_dist_ratio):
    '''

    Given two matrix of shape N1xC, N2xC, compute pairwise distance

    Args:
        descs1 (torch.Tensor): N1 x C array, N1 descriptors of size C
        descs2 (torch.Tensor): N2 x C array, N2 descriptors of size C
        max_dist_ratio(float): Maximum distance ratio between first and 
            second best matches.

    Returns:
        (torch.Tensor): M x 2 uint32 array representing pairwise matches
    '''
    N1, C = descs1.shape
    N2, C2 = descs2.shape

    # Exhaustively compute distances between all descriptors.
    dists = pdist(descs1, descs2)

    # Find the first best matches.
    idxs1 = torch.tensor(list(range(N1))).to(descs1.device)
    fmin_dists12, idxs12 = dists.min(2)
    fmin_dists21, idxs21 = dists.min(1)
    idxs121 = idxs21[idxs12]


    # compute second best matches
    dists[idxs1][idxs12] = float('inf')
    smin_dists12 = dists.min(2)[0]
    dist_ratios12 = fmin_dists12 / smin_dists12

    mask = (dist_ratios12 <= max_dist_ratio) & (idxs1 == idxs121)
    valid_idxs1 = idxs1(mask)
    valid_idxs2 = idxs12(mask)

    return torch.stack((valid_idxs1, valid_idxs2))
