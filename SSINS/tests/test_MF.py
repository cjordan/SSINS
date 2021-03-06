from SSINS import MF, INS
from SSINS.data import DATA_PATH
import os
import numpy as np
import pytest


def test_init():

    freq_path = os.path.join(DATA_PATH, 'MWA_Highband_Freq_Array.npy')
    freqs = np.load(freq_path)

    # Make a shape that encompasses the first five channels
    ch_width = freqs[1] - freqs[0]
    shape = [freqs[0] - 0.1 * ch_width, freqs[4] + 0.1 * ch_width]
    shape_dict = {'shape': shape}
    sig_thresh = {'narrow': 10, 'shape': 5, 'streak': 5}

    mf_1 = MF(freqs, sig_thresh, shape_dict=shape_dict)

    assert mf_1.slice_dict['shape'] == slice(0, 5), "It did not set the shape correctly"
    assert mf_1.slice_dict['narrow'] is None, "narrow did not get set correctly"
    assert mf_1.slice_dict['streak'] == slice(0, 384), "streak did not get set correctly"
    assert mf_1.sig_thresh == sig_thresh

    # Test disabling streak/narrow
    mf_2 = MF(freqs, sig_thresh, shape_dict=shape_dict, narrow=False, streak=False)

    assert 'narrow' not in mf_2.slice_dict, "narrow is still in the shape_dict"
    assert 'streak' not in mf_2.slice_dict, "streak still in shape_dict"
    assert 'shape' in mf_2.slice_dict, "shape not in shape_dict"

    # Test if error gets raised with bad shape_dict
    try:
        mf_3 = MF(freqs, sig_thresh, shape_dict={}, streak=False, narrow=False)
    except ValueError:
        pass

    # Test if error gets raised with missing significances
    try:
        mf_4 = MF(freqs, {'narrow': 5}, shape_dict={'shape': shape})
    except KeyError:
        pass

    # Test passing a number to shape_dict
    mf_5 = MF(freqs, 5, shape_dict={'shape': shape})
    assert mf_5.sig_thresh == {'shape': 5, 'narrow': 5, 'streak': 5}


def test_match_test():

    obs = '1061313128_99bl_1pol_half_time'
    insfile = os.path.join(DATA_PATH, '%s_SSINS.h5' % obs)

    ins = INS(insfile)

    # Mock a simple metric_array and freq_array
    ins.metric_array = np.ma.ones([10, 20, 1])
    ins.weights_array = np.copy(ins.metric_array)
    ins.freq_array = np.zeros([1, 20])
    ins.freq_array = np.arange(20)

    # Make a shape dictionary for a shape that will be injected later
    shape = [7.9, 12.1]
    shape_dict = {'shape': shape}
    sig_thresh = {'shape': 5, 'narrow': 5, 'streak': 5}
    mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict)

    # Inject a shape, narrow, and streak event
    ins.metric_array[3, 5] = 10
    ins.metric_array[5] = 10
    ins.metric_array[7, 7:13] = 10
    ins.metric_ms = ins.mean_subtract()

    t_max, f_max, R_max, shape_max = mf.match_test(ins)
    print(shape_max)

    assert t_max == 5, "Wrong time"
    assert f_max == slice(0, 20), "Wrong freq"
    assert shape_max == 'streak', "Wrong shape"


def test_apply_match_test():

    obs = '1061313128_99bl_1pol_half_time'
    insfile = os.path.join(DATA_PATH, '%s_SSINS.h5' % obs)

    ins = INS(insfile)

    # Mock a simple metric_array and freq_array
    ins.metric_array = np.ma.ones([10, 20, 1])
    ins.weights_array = np.copy(ins.metric_array)
    ins.freq_array = np.zeros([1, 20])
    ins.freq_array = np.arange(20)

    # Make a shape dictionary for a shape that will be injected later
    shape = [7.9, 12.1]
    shape_dict = {'shape': shape}
    sig_thresh = {'shape': 5, 'narrow': 5, 'streak': 5}
    mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict)

    # Inject a shape, narrow, and streak event
    ins.metric_array[3, 5] = 10
    ins.metric_array[5] = 10
    ins.metric_array[7, 7:13] = 10
    ins.metric_ms = ins.mean_subtract()
    ins.sig_array = np.ma.copy(ins.metric_ms)

    mf.apply_match_test(ins, event_record=True)

    # Check that the right events are flagged
    test_mask = np.zeros(ins.metric_array.shape, dtype=bool)
    test_mask[3, 5] = 1
    test_mask[5] = 1
    test_mask[7, 7:13] = 1

    assert np.all(test_mask == ins.metric_array.mask), "Flags are incorrect"

    test_match_events_slc = [(5, slice(0, 20), 'streak'),
                             (7, slice(7, 13), 'shape'),
                             (3, slice(5, 6), 'narrow')]

    for i, event in enumerate(test_match_events_slc):
        assert ins.match_events[i][:-1] == test_match_events_slc[i], "%ith event is wrong" % i

    assert not np.any([ins.match_events[i][-1] < 5 for i in range(3)]), "Some significances were less than 5"

    # Test a funny if block that is required when the last time in a shape is flagged
    ins.metric_array[1:, 7:13] = np.ma.masked
    ins.metric_ms[0, 7:13] = 10

    mf.apply_match_test(ins, event_record=True)

    assert np.all(ins.metric_ms.mask[:, 7:13]), "All the times were not flagged for the shape"


def test_samp_thresh():

    obs = '1061313128_99bl_1pol_half_time'
    insfile = os.path.join(DATA_PATH, '%s_SSINS.h5' % obs)
    out_prefix = os.path.join(DATA_PATH, '%s_test' % obs)
    match_outfile = '%s_SSINS_match_events.yml' % out_prefix

    ins = INS(insfile)

    # Mock a simple metric_array and freq_array
    ins.metric_array = np.ma.ones([10, 20, 1])
    ins.weights_array = np.copy(ins.metric_array)
    ins.metric_ms = ins.mean_subtract()
    ins.sig_array = np.ma.copy(ins.metric_ms)
    ins.freq_array = np.zeros([1, 20])
    ins.freq_array = np.arange(20)

    # Arbitrarily flag enough data in channel 10
    sig_thresh = {'narrow': 5}
    mf = MF(ins.freq_array, sig_thresh, streak=False, N_samp_thresh=5)
    ins.metric_array[3:, 10] = np.ma.masked
    ins.metric_array[3:, 9] = np.ma.masked
    # Put in an outlier so it gets to samp_thresh_test
    ins.metric_array[0, 11] = 10
    ins.metric_ms = ins.mean_subtract()
    bool_ind = np.zeros(ins.metric_array.shape, dtype=bool)
    bool_ind[:, 10] = 1
    bool_ind[:, 9] = 1
    bool_ind[0, 11] = 1

    mf.apply_match_test(ins, event_record=True, apply_samp_thresh=True)
    test_match_events = [(0, slice(11, 12), 'narrow')]
    test_match_events += [(ind, slice(9, 10), 'samp_thresh') for ind in range(3)]
    test_match_events += [(ind, slice(10, 11), 'samp_thresh') for ind in range(3)]
    # Test stuff
    assert np.all(ins.metric_array.mask == bool_ind), "The right flags were not applied"
    for i, event in enumerate(test_match_events):
        assert ins.match_events[i][:-1] == event, "The events weren't appended correctly"

    # Test that writing with samp_thresh flags is OK
    ins.write(out_prefix, output_type='match_events')
    test_match_events_read = ins.match_events_read(match_outfile)
    os.remove(match_outfile)
    assert ins.match_events == test_match_events_read

    # Test that exception is raised when N_samp_thresh is too high
    with pytest.raises(ValueError):
        mf = MF(ins.freq_array, {'narrow': 5, 'streak': 5}, N_samp_thresh=100)
        mf.apply_samp_thresh_test(ins)
