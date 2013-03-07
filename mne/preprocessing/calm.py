"""CALM method for noise reduction using reference sensors


"""

from numpy import dot, empty_like
from scipy.linalg import lstsq


def calm(data, refs, buffer_size=6001):
    """Apply CALM method to reduce environment noise in MEG data

    Parameters
    ----------
    data : array [n_samples, n_sensors]
        data
    refs : array [n_samples, n_ref_sensors]
        reference channels
    buffer_size : int
        Buffer length in samples
    """
    buf = int((buffer_size - 1) / 2)
    N = len(data)

    # check parameters
    if len(refs) != N:
        err = ("Reference sensor data must have the same number of samples as"
               "data (got %i, %i)" % (len(refs), N))
        raise ValueError(err)

    out = empty_like(data)

    for s in xrange(len(data)):
        tmin = max(0, s - buf)
        tmax = min(N, s + buf)
        A = refs[tmin:tmax]  # t x ref
        B = data[tmin:tmax]  # t x ch
        x, _, _, _ = lstsq(A, B)  # ref x ch
        out[s] = data[s] - dot(refs[s], x)

    return out
