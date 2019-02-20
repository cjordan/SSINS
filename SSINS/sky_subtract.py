"""
The sky_subtract (SS) class is defined here. This is the backbone of the analysis
pipeline when working with raw datafiles or UVData objects.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from pyuvdata import UVData
import os
from SSINS import util
from SSINS import INS
from SSINS import MF
from SSINS import VDH
from SSINS import ES
import scipy.stats
import warnings
import time


class SS(UVData):

    """
    Defines the SS class.
    """

    def __init__(self):

        """
        """
        super(SS, self).__init__()

    def read(self, filename, diff=True, flag_choice=None, INS=None, custom=None,
             **kwargs):
        super(SS, self).read(filename, **kwargs)
        if (self.data_array is not None) and diff:
            self.diff()
            self.apply_flags(flag_choice=flag_choice)

    def apply_flags(self, flag_choice=None, INS=None, custom=None):
        """
        A function which applies flags to the data via numpy masked arrays. Also
        changes the SS.flag_choice attribute, which will affect saved outputs,
        so it is convenient to change flags using this function.

        keywords: choice: Options are None, 'original', 'INS', and 'custom'

                          None: No flags are applied to the data

                          'original': The "differenced flags" from the original
                                      flag_array are applied to the data

                          'custom': A custom flag array will be applied to the
                                    data.

                          'INS': A flag_array developed from an INS will be
                                 applied to the data. All flags in the INS will
                                 be extended across the baseline axis of the SS
                                 data array.

                  custom: The custom flags to be applied. Must be used in
                          conjunction with choice='custom'

                  INS: The INS whose flags will be applied. Must be used in
                       conjunction with choice='INS'
        """
        self.flag_choice = flag_choice
        if flag_choice is 'original':
            self.data_array.mask = self.flag_array
        elif flag_choice is 'INS':
            ind = np.where(INS.data.mask)
            self.data_array[ind[0], :, ind[1], ind[2], ind[3]] = np.ma.masked
        elif flag_choice is 'custom':
            if custom is not None:
                self.data_array[custom] = np.ma.masked
            else:
                warnings.warn('Custom flags were chosen, but custom flags were None type. Not applying flags.')
        elif np.any(self.data_array.mask) or (not self.data_array.mask):
            self.data_array.mask = np.zeros(self.data_array.shape, dtype=bool)

    def diff(self):

        assert self.Nblts == self.Nbls * self.Ntimes, 'Nblts != Nbls * Ntimes'
        cond = np.all([self.baseline_array[:self.Nbls] == self.baseline_array[k * self.Nbls:(k + 1) * self.Nbls]
                       for k in range(1, self.Ntimes - 1)])
        assert cond, 'Baseline array slices do not match in each time! The baselines are out of order.'

        # Difference in time and OR the flags
        self.data_array = np.ma.masked_array(self.data_array[self.Nbls:] - self.data_array[:-self.Nbls])
        self.flag_array = np.logical_or(self.flag_array[self.Nbls:], self.flag_array[:-self.Nbls])

        # Adjust the UVData attributes.
        self.Nblts -= self.Nbls
        self.ant_1_array = self.ant_1_array[:-self.Nbls]
        self.ant_2_array = self.ant_2_array[:-self.Nbls]
        self.baseline_array = self.baseline_array[:-self.Nbls]
        self.integration_time = self.integration_time[self.Nbls:] + self.integration_time[:-self.Nbls]
        self.Ntimes -= 1
        self.nsample_array = 0.5 * (self.nsample_array[self.Nbls:] + self.nsample_array[:-self.Nbls])
        self.time_array = 0.5 * (self.time_array[self.Nbls:] + self.time_array[:-self.Nbls])
        self.uvw_array = 0.5 * (self.uvw_array[self.Nbls:] + self.uvw_array[:-self.Nbls])
        super(SS, self).set_lsts_from_time_array()

    def INS_prepare(self, history='', label='', order=0):

        """
        Prepares an INS. Passes all possible relevant non-conflicting attributes.
        """

        self.INS = INS(self, history='', label='', order=order)

    def MLE_calc(self):

        self.MLE = np.sqrt(0.5 * np.mean(np.absolute(self.data_array)**2, axis=(0, 1, -1)))

    def mixture_prob(self, bins):
        """
        Calculates the probabilities of landing in each bin for a given set of
        bins.

        args:
            bins: The bin edges of the bins to calculate the probabilities for.
        """

        if not hasattr(self, 'MLE'):
            self.MLE_calc()

        N_spec = np.sum(np.logical_not(self.data_array.mask), axis=(0, 1, -1))
        N_total = np.sum(N_spec)

        # Calculate the fraction belonging to each frequency
        chi_spec = N_spec / N_total

        # initialize the probability array
        prob = np.zeros(len(bins) - 1)
        # Calculate the mixture distribution
        # If this could be vectorized over frequency, that would be better.
        for chan in range(self.Nfreqs):
            if self.MLE[chan] > 0:
                quants = scipy.stats.rayleigh.cdf(bins, scale=self.MLE[chan])
                prob += chi_spec[chan] * (quants[1:] - quants[:-1])

        return(prob)

    def rev_ind(self, band):

        where_band = np.logical_and(np.absolute(self.data_array) > min(band),
                                    np.absolute(self.data_array) < max(band))
        where_band_mask = np.logical_and(np.logical_not(self.data_array.mask),
                                         where_band)
        shape = [self.Ntimes, self.Nbls, self.Nfreqs, self.Npols]
        return(np.sum(where_band_mask.reshape(shape), axis=1))

    def write(self, filename_out, file_type_out, UV=None, filename_in=None,
              read_kwargs={}, combine=True, nsample_default=1, write_kwargs={}):

        """
        Lets one write out a newly flagged file. Data is recovered by reading
        in the original file or using the original UV object. If passing a UV
        object, be careful that the original UV object was not changed by any
        operations due to typical confusing python binding issues. The operation
        used to make "differenced flags" is actually not invertible in some
        cases, so this just extends flags as much as possible.

        Keywords: outpath: The name of the file to write out to.

                  file_type_out: The file_type to write out to.

                  UV: If using this, make sure it is the original UV object
                      intended without any extra flagging or differencing or
                      reshaped arrays.

                  inpath: The file to read in to get the original data from.

                  read_kwargs: The UVData.read keyword dict for the original
                               UVData object

                  bad_time_indices: Bad time indices to remove from original
                                    UVData object.
        """

        if UV is None:
            UV = super(SS, self).read(filename_in, **read_kwargs)

        # Test that assumptions abouts blts axis are ok
        assert UV.Nblts == UV.Nbls * UV.Ntimes, 'Nblts != Nbls * Ntimes for UV object.'
        cond = np.all([UV.baseline_array[:UV.Nbls] == UV.baseline_array[k * UV.Nbls:(k + 1) * UV.Nbls]
                       for k in range(1, UV.Ntimes)])
        assert cond, 'Baseline array slices do not match in each time! The baselines are out of order.'

        # Check nsample_array for issue
        if np.any(UV.nsample_array == 0) and (file_type_out is 'uvfits'):
            warnings.warn("Writing uvfits file with some nsample == 0. This will"
                          " result in a failure to propagate flags. Changing "
                          " nsample value to nsample_default parameter (default is 1)")
            UV.nsample_array[UV.nsample_array == 0] = nsample_default

        # Option to keep old flags
        if not combine:
            UV.flag_array[:] = 0

        # Propagate the new flags
        for i in range(self.Ntimes):
            # This actually does not invert properly but I think it's the best way
            UV.flag_array[i * self.Nbls: (i + 1) * self.Nbls][self.data_array.mask[i * self.Nbls: (i + 1) * self.Nbls]] = 1
            UV.flag_array[(i + 1) * self.Nbls: (i + 2) * self.Nbls][self.data_array.mask[i * self.Nbls: (i + 1) * self.Nbls]] = 1

        # Write file
        getattr(UV, 'write_%s' % file_type_out)(file_name_out, **write_kwargs)
