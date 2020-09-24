import numpy as np
from scipy.interpolate import UnivariateSpline


def calc_cv(orientation_curve, orientation):
    vect_sum = orientation_curve.dot(np.exp(1j * 2 * orientation / 180 * np.pi))
    osi = np.absolute(vect_sum) / np.sum(np.absolute(orientation_curve))
    return 1 - osi


def calc_bw(orientation_curve, orientation, filt_type='hanning', thrsh=0.5, mode='full'):
    or_ext = np.hstack((orientation - 180, orientation, orientation + 180))
    or_curve_ext = np.tile(orientation_curve, (1, 3))

    if filt_type == 'hanning':
        w = np.array([0, 2/5, 1, 2/5, 0])
    elif filt_type == 'flat':
        w = np.array([1, 1, 1, 1, 1])
    elif filt_type == 'smooth':
        w = np.array([0, 1/5, 1, 1/5, 0])

    if filt_type is not None:
        or_curve_ext = np.convolve(w / w.sum(), np.squeeze(or_curve_ext), mode='same')
    or_curve_spl = UnivariateSpline(or_ext, or_curve_ext, s=0.)

    or_full = np.linspace(-180, 359, 540)
    or_curve_full = or_curve_spl(or_full)
    pref_or_fit = np.argmax(or_curve_full[181:360])
    or_curve_max = or_curve_full[pref_or_fit + 181]

    try:
        less = np.where(or_curve_full <= or_curve_max * thrsh)[0][:]
        p1 = or_full[less[np.where(less < pref_or_fit + 181)[0][-1]]]
        p2 = or_full[less[np.where(less > pref_or_fit + 181)[0][0]]]
        bw = (p2 - p1)
        if bw > 180:
            bw = np.nan
    except:
        bw = np.nan
    if mode is 'half':
        bw = bw / 2
    return bw, pref_or_fit, or_full[181:360], or_curve_full[181:360]


def calc_opr(orientation_curve, orientation):
    pref_orientation = np.argmax(orientation_curve)
    orth_orientation = pref_orientation + int(len(orientation)/2)
    if orth_orientation >= len(orientation):
        orth_orientation -= len(orientation)
    opr = orientation_curve[orth_orientation] / orientation_curve[pref_orientation]
    return opr
