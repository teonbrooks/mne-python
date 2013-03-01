"""Conversion tool from SQD to FIF

RawKIT class is adapted from Denis Engemann et al.'s mne_bti2fiff.py

"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import logging
from os import SEEK_CUR
from struct import unpack
import time
import warnings

import numpy as np
from scipy.linalg import norm

from ...transforms.coreg import fit_matched_pts
from ...utils import verbose
from ..raw import Raw
from ..constants import FIFF
from .constants import KIT, KIT_NY, KIT_AD
from .coreg import get_dig_points, read_elp, read_hsp, read_mrk, read_sns, \
                   get_neuromag_transform, transform_ALS_to_RAS

logger = logging.getLogger('mne')


class RawKIT(Raw):
    """Raw object from KIT SQD file adapted from bti/raw.py

    Parameters
    ----------
    input_fname : str
        Absolute path to the sqd file.
    mrk_fname : str
        Absolute path to marker coils file.
    elp_fname : str
        Absolute path to elp digitizer laser points file.
    hsp_fname : str
        Absolute path to elp digitizer head shape points file.
    sns_fname : str
        Absolute path to sensor information file.
    stim : list
        List of trigger channels.
    data : bool | array-like
        Array-like data to use in lieu of data from sqd file.
    stimthresh : float
        The threshold level for accepting voltage change as a trigger event.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.fiff.Raw : Documentation of attribute and methods.
    """
    @verbose
    def __init__(self, input_fname, stim, sns_fname, mrk=None, elp=None,
                 hsp=None, stimthresh=3.5, verbose=None):

        logger.info('Extracting SQD Parameters from %s...' % input_fname)
        sqd = _get_sqd_params(input_fname)
        logger.info('Creating Raw.info structure...')

        # Raw attributes
        self.verbose = verbose
        self._preloaded = True
        self.fids = list()

        # Create raw.info dict for raw fif object with SQD data
        self.info = {}
        self.info['meas_id'] = None
        self.info['file_id'] = None
        self.info['meas_date'] = int(time.time())
        self.info['projs'] = []
        self.info['comps'] = []
        self.info['lowpass'] = sqd['lowpass']
        self.info['highpass'] = sqd['highpass']
        self.info['sfreq'] = float(sqd['sfreq'])
        self.info['nchan'] = sqd['nchan'] + 1  # adds synthetic channel
        self.info['bads'] = []
        self.info['acq_pars'], self.info['acq_stim'] = None, None
        self.info['filename'] = None
        self.info['ctf_head_t'] = None
        self.info['dev_ctf_t'] = []
        self.info['filenames'] = []
        self.info['dev_head_t'] = {}
        self.info['dev_head_t']['from'] = FIFF.FIFFV_COORD_DEVICE
        self.info['dev_head_t']['to'] = FIFF.FIFFV_COORD_HEAD

        if (mrk and elp and hsp):
            self.set_dig(mrk, elp, hsp)
        elif (mrk or elp or hsp):
            err = ("mrk, elp and hsp need to be provided as a group (all or "
                   "none)")
            raise ValueError(err)

        # Creates a list of dicts of meg channels for raw.info
        logger.info('Setting channel info structure...')
        ch_names = {}
        ch_names['MEG'] = ['MEG %03d' % ch for ch
                                in range(1, sqd['KIT'].n_sens + 1)]
        ch_names['MISC'] = ['MISC %03d' % ch for ch
                                 in range(1, sqd['KIT'].nmiscchan + 1)]
        ch_names['STIM'] = ['STI 014']
        locs = read_sns(sns_fname=sns_fname)
        chan_locs = transform_ALS_to_RAS(locs[:, :3])
        chan_angles = locs[:, 3:]
        self.info['chs'] = []
        for idx, ch_info in enumerate(zip(ch_names['MEG'], chan_locs,
                                          chan_angles), 1):
            ch_name, ch_loc, ch_angles = ch_info
            chan_info = {}
            chan_info['cal'] = KIT.CALIB_FACTOR
            chan_info['logno'] = idx
            chan_info['scanno'] = idx
            chan_info['range'] = KIT.RANGE
            chan_info['unit_mul'] = KIT.UNIT_MUL
            chan_info['ch_name'] = ch_name
            chan_info['unit'] = FIFF.FIFF_UNIT_T
            chan_info['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
            if idx <= sqd['KIT'].nmegchan:
                chan_info['coil_type'] = FIFF.FIFFV_COIL_KIT_GRAD
                chan_info['kind'] = FIFF.FIFFV_MEG_CH
            else:
                chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
                chan_info['kind'] = FIFF.FIFFV_REF_MEG_CH

            # create three orthogonal vector
            # ch_angles[0]: theta, ch_angles[1]: phi
            ch_angles = np.radians(ch_angles)
            x = np.sin(ch_angles[0]) * np.cos(ch_angles[1])
            y = np.sin(ch_angles[0]) * np.sin(ch_angles[1])
            z = np.cos(ch_angles[0])
            vec_z = np.array([x, y, z])
            length = norm(vec_z)
            vec_z /= length
            vec_x = np.zeros(vec_z.size, dtype=float)
            if vec_z[1] < vec_z[2]:
                if vec_z[0] < vec_z[1]:
                    vec_x[0] = 1.0
                else:
                    vec_x[1] = 1.0
            elif vec_z[0] < vec_z[2]:
                vec_x[0] = 1.0
            else:
                vec_x[2] = 1.0
            vec_x -= np.sum(vec_x * vec_z) * vec_z
            length = norm(vec_x)
            vec_x /= length
            vec_y = np.cross(vec_z, vec_x)
            # transform to Neuromag like coordinate space
            vecs = np.vstack((vec_x, vec_y, vec_z))
            vecs = transform_ALS_to_RAS(vecs, unit='m')
            chan_info['loc'] = np.vstack((ch_loc, vecs)).ravel()
            self.info['chs'].append(chan_info)

        # label trigger and misc channels
        for idy, ch_name in enumerate(ch_names['MISC'] + ch_names['STIM'],
                                      sqd['KIT'].n_sens):
            chan_info = {}
            chan_info['cal'] = KIT.CALIB_FACTOR
            chan_info['logno'] = idy
            chan_info['scanno'] = idy
            chan_info['range'] = 1.0
            chan_info['unit'] = FIFF.FIFF_UNIT_V
            chan_info['unit_mul'] = 0  # default is 0 mne_manual p.273
            chan_info['ch_name'] = ch_name
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['loc'] = np.zeros(12)
            if ch_name.startswith('STI'):
                chan_info['unit'] = FIFF.FIFF_UNIT_NONE
                chan_info['kind'] = FIFF.FIFFV_STIM_CH
            else:
                chan_info['kind'] = FIFF.FIFFV_MISC_CH
            self.info['chs'].append(chan_info)
        self.info['ch_names'] = (ch_names['MEG'] + ch_names['MISC'] +
                                 ch_names['STIM'])

        logger.info('Reading raw data from %s...' % input_fname)
        self._data = _get_sqd_data(rawfile=input_fname, sqd=sqd)
        assert len(self._data) == self.info['nchan']

        # Create a synthetic channel
        trig_chs = self._data[stim, :]
        trig_chs = trig_chs > stimthresh
        trig_vals = np.array(2 ** np.arange(len(stim)), ndmin=2).T
        trig_chs = trig_chs * trig_vals
        stim_ch = trig_chs.sum(axis=0)
        self._data[-1, :] = stim_ch

        # Add time info
        self.first_samp, self.last_samp = 0, self._data.shape[1] - 1
        self._times = np.arange(self.first_samp, self.last_samp + 1)
        self._times /= self.info['sfreq']
        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs'
                    % (self.first_samp, self.last_samp,
                       float(self.first_samp) / self.info['sfreq'],
                       float(self.last_samp) / self.info['sfreq']))
        logger.info('Ready.')

    def set_dig(self, elp, hsp, mrk):
        """
        Fill in the digitizer data using points in Polhemus space

        Parameters
        ----------
        mrk : None | array, shape = (5, 3)
            Marker points used to estimate the device head transform. If trans
            is provided, mrk is not used and can be None).
        max_hsp_n : None | int
            Maximum number of head shape points to keep.
        """
        if isinstance(hsp, basestring):
            hsp = read_hsp(hsp)
            n_pts = len(hsp)
            if n_pts > KIT.DIG_POINTS:
                msg = ("The selected head shape contains %i points, which is "
                       "more than recommended (%i)" % (n_pts, KIT.DIG_POINTS))
                warnings.warn(msg)

        if isinstance(elp, basestring):
            elp = read_elp(elp)

        if isinstance(mrk, basestring):
            mrk = read_mrk(mrk)

        hsp = transform_ALS_to_RAS(hsp)
        elp = transform_ALS_to_RAS(elp)
        mrk = transform_ALS_to_RAS(mrk, unit='m')

        nasion, lpa, rpa = elp[:3]
        nmtrans = get_neuromag_transform(nasion, lpa, rpa).T
        elp = np.dot(elp, nmtrans)
        hsp = np.dot(hsp, nmtrans)

        trans = fit_matched_pts(tgt_pts=mrk, src_pts=elp)

        self.set_transformed_dig(elp, hsp, trans)

    def set_transformed_dig(self, elp, hsp, trans):
        """
        Fill in the digitizer data using points that are already transformed to
        neuromag space

        Parameters
        ----------
        mrk : array, shape = (5, 3)
            Marker points used to estimate the device head transform. If trans
            is provided, mrk is not used and can be None).
        trans : None | array, shape = (4, 4)
            Device head transformation.
        """
        self.info['dig'] = get_dig_points(elp, hsp)

        trans = np.asarray(trans)
        if not trans.shape == (4, 4):
            raise ValueError("trans needs to be 4 by 4 array")
        self.info['dev_head_t']['trans'] = trans



def read_raw_kit(input_fname, mrk, elp, hsp, sns_fname,
                 stim, data=None, stimthresh=3.5, verbose=None):
    """Reader function for KIT conversion to FIF

    Parameters
    ----------
    input_fname : str
        Absolute path to the sqd file.
    mrk : str | array, shape = (5, 3)
        Absolute path to marker coils file, or array of points.
    elp : str | array, shape = (8, 3)
        Absolute path to elp digitizer laser points file, or array with points.
    hsp : str | array, shape = (n_pts, 3)
        Absolute path to elp digitizer head shape points file, or array with
        points.
    sns_fname : str
        Absolute path to sensor information file.
    stim : list
        List of trigger channels.
    data : bool | array-like
        Array-like data to use in lieu of data from sqd file.
    stimthresh : float
        The threshold level for accepting voltage change as a trigger event.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    return RawKIT(input_fname=input_fname, mrk=mrk, elp=elp, hsp=hsp,
                  sns_fname=sns_fname, stim=stim, data=data,
                  stimthresh=stimthresh, verbose=verbose)


def _get_sqd_params(rawfile):
    """Extracts all the information from the sqd file.

    Parameters
    ----------
    rawfile : str
        Raw sqd file to be read.

    Returns
    -------
    sqd : dict
        A dict containing all the sqd parameter settings.
    """
    sqd = {}
    sqd['rawfile'] = rawfile
    sqd['KIT'] = KIT
    with open(rawfile, 'r') as fid:
        fid.seek(sqd['KIT'].BASIC_INFO)
        basic_offset = unpack('i', fid.read(sqd['KIT'].INT))[0]
        fid.seek(basic_offset)
        # skips version, revision, sysid
        fid.seek(sqd['KIT'].INT * 3, SEEK_CUR)
        # basic info
        sysname = unpack('128s', fid.read(sqd['KIT'].STRING))
        sysname = sysname[0].split('\n')[0]
        fid.seek(sqd['KIT'].STRING, SEEK_CUR)  # skips modelname
        sqd['nchan'] = unpack('i', fid.read(sqd['KIT'].INT))[0]

        if sysname == 'New York University Abu Dhabi':
            sqd['KIT'] = KIT_AD
        elif sysname == 'NYU 160ch System since Jan24 2009':
            sqd['KIT'] = KIT_NY
        else:
            raise NotImplementedError

        # amplifier gain
        fid.seek(sqd['KIT'].AMPLIFIER_INFO)
        amp_offset = unpack('i', fid.read(sqd['KIT'].INT))[0]
        fid.seek(amp_offset)
        amp_data = unpack('i', fid.read(sqd['KIT'].INT))[0]

        gain1 = sqd['KIT'].GAINS[(sqd['KIT'].GAIN1_MASK & amp_data)
                                    >> sqd['KIT'].GAIN1_BIT]
        gain2 = sqd['KIT'].GAINS[(sqd['KIT'].GAIN2_MASK & amp_data)
                                    >> sqd['KIT'].GAIN2_BIT]
        if sqd['KIT'].GAIN3_BIT:
            gain3 = sqd['KIT'].GAINS[(sqd['KIT'].GAIN3_MASK & amp_data)
                                     >> sqd['KIT'].GAIN3_BIT]
            sqd['amp_gain'] = gain1 * gain2 * gain3
        else:
            sqd['amp_gain'] = gain1 * gain2

        # filter settings
        sqd['lowpass'] = sqd['KIT'].LPFS[(sqd['KIT'].LPF_MASK & amp_data)
                                    >> sqd['KIT'].LPF_BIT]
        sqd['highpass'] = sqd['KIT'].HPFS[(sqd['KIT'].HPF_MASK & amp_data)
                                    >> sqd['KIT'].HPF_BIT]
        sqd['notch'] = sqd['KIT'].BEFS[(sqd['KIT'].BEF_MASK & amp_data)
                                    >> sqd['KIT'].BEF_BIT]

        # only sensor channels requires gain. the additional misc channels
        # (trigger channels, audio and voice channels) are passed
        # through unaffected

        fid.seek(sqd['KIT'].CHAN_SENS)
        sens_offset = unpack('i', fid.read(sqd['KIT'].INT))[0]
        fid.seek(sens_offset)
        sens = np.fromfile(fid, dtype='d', count=sqd['nchan'] * 2)
        sensitivities = (np.reshape(sens, (sqd['nchan'], 2))
                         [:sqd['KIT'].n_sens, 1])
        sqd['sensor_gain'] = np.ones(sqd['nchan'] + 1)  # extra ch for STI 014
        sqd['sensor_gain'][:sqd['KIT'].n_sens] = sensitivities

        fid.seek(sqd['KIT'].SAMPLE_INFO)
        acqcond_offset = unpack('i', fid.read(sqd['KIT'].INT))[0]
        fid.seek(acqcond_offset)
        acq_type = unpack('i', fid.read(sqd['KIT'].INT))[0]
        if acq_type == 1:
            sqd['sfreq'] = unpack('d', fid.read(sqd['KIT'].DOUBLE))[0]
            _ = fid.read(sqd['KIT'].INT)  # initialized estimate of samples
            sqd['nsamples'] = unpack('i', fid.read(sqd['KIT'].INT))[0]
        else:
            raise NotImplementedError
    return sqd


def _get_sqd_data(rawfile, sqd):
    """Extracts the data from the sqd file.

    Parameters
    ----------
    rawfile : str
        Raw sqd file to be read.
    sqd : dict
        A dict of parameters for the rawfile.

    Returns
    -------
    sqd : dict
        A dict containing all the sqd parameter settings.
    """
    with open(rawfile, 'r') as fid:
        # extract data
        fid.seek(sqd['KIT'].DATA_OFFSET)
        # data offset info
        data_offset = unpack('i', fid.read(sqd['KIT'].INT))[0]

        fid.seek(data_offset)
        data = np.empty((sqd['nsamples'], sqd['nchan'] + 1))
        count = sqd['nsamples'] * sqd['nchan']
        data[:, :sqd['nchan']] = np.fromfile(fid, dtype='h', count=count
                                             ).reshape((sqd['nsamples'],
                                                        sqd['nchan']))
        # amplifier applies only to the sensor channels
        sqd['sensor_gain'][:sqd['KIT'].n_sens] /= sqd['amp_gain']
        conv_factor = np.array((sqd['KIT'].VOLTAGE_RANGE /
                                sqd['KIT'].DYNAMIC_RANGE) *
                               sqd['sensor_gain'], ndmin=2)
        data *= conv_factor
    return data.T
