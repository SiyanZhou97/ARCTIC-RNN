import copy
import scipy.stats
try:
    import cupy as np
    a=np.array([0])
except:
    import numpy as np
#import torch

def create_cosine_bumps(x, centers, widths):
    """Create cosine bumps.

    Modified from Tseng, et al. Neuron. (2022)

    Args:
        x (ndarray of shape (n_samples, )): x positions to evaluate the cosine bumps on.
        centers (ndarray of shape (number of bumps, )): contains center positions of bumps.
        widths (ndarray of shape (number of bumps, )): the width of each bump.

    Returns:
        bases (ndarray of shape (n_samples, number of bumps)): basis functions.

    """
    # Sanity check
    assert centers.shape == widths.shape, 'Centers and widths should have same number of elements'
    x_reshape = x.reshape(-1,)
    # Create empty array for basis functions
    bases = np.full((x.shape[0], centers.shape[0]), np.nan)
    # Loop over center positions
    for idx, cent in enumerate(centers):
        bases[:, idx] = (np.cos(2 * np.pi * (x_reshape - cent) / widths[idx]) * 0.5 + 0.5) * \
                    (np.absolute(x_reshape - cent) < widths[idx] / 2)
    return bases

def position_expansion(pos,n_pos_bases=5,start_pos=0,end_pos=1):
    """Expand a scalar position to the vector representation by bases

    Modified from Tseng, et al. Neuron. (2022)

    Args:
        pos (ndarray of shape (n_samples, )): actual position
        n_pos_bases (scalar, optional): required number of bases
        start_pos (scalar, optional): required starting position
        end_pos (scalar, optional): required ending position

    Returns:
        pos_bases (ndarray of shape (n_samples, number of bumps)): ector representation by bases

    """
    # Linearly space the centers of the position bases
    pos_centers = np.linspace(start_pos, end_pos, n_pos_bases)
    # Set width of the position bases as 2 times spacing
    width_to_spacing_ratio = 2
    try:
        pos_width = width_to_spacing_ratio * scipy.stats.mode(np.asnumpy(np.diff(pos_centers)))[0][0]
    except:
        try:
            pos_width = width_to_spacing_ratio * scipy.stats.mode(np.asnumpy(np.diff(pos_centers)))[0].item()
        except:
            try:
                pos_width = width_to_spacing_ratio * scipy.stats.mode(np.diff(pos_centers))[0][0]
            except:
                pos_width = width_to_spacing_ratio * scipy.stats.mode(np.diff(pos_centers))[0].item()
    # Evaluate the basis expanded position at each basis
    pos_bases = create_cosine_bumps(pos, pos_centers, pos_width * np.ones_like(pos_centers)) # position basis expansion
    return pos_bases
