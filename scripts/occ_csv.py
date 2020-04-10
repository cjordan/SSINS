from SSINS import INS, util
import argparse
import csv
import numpy as np


def csv_to_dict_list(csv_filepath):

    with open(csv_filepath, newline='') as csv_file:
        dict_reader = csv.DictReader(csv_file)
        dict_list = []
        for row in dict_reader:
            dict_list.append(row)

    return(dict_list)


def dict_list_to_csv(dict_list, outpath, fieldnames):

    with open(outpath, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames)

        writer.writeheader()
        for row in dict_list:
            writer.writerow(row)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--shapes', nargs='*', help='The shapes to calculate occupancies for')
    parser.add_argument('-c', '--csv', help='A csv whose columns are obsids, ins data files, ins masks, and yml files')
    parser.add_argument('-o', '--outfile', help='The name of the output csv that contains occupancy information')
    parser.add_argument('-i', '--ch_ignore', help='A text file of fine frequency channels to ignore in occupancy calculation')
    parser.add_argument('-t', '--time_ignore', help='Time indices to ignore when calculating occupancy')
    args = parser.parse_args()

    dict_list = csv_to_dict_list(args.csv)

    if args.ch_ignore is not None:
        fine_channels_ignore = util.make_obslist(args.ch_ignore)
        fine_channels_ignore = np.array(fine_channels_ignore).astype(int)
    else:
        fine_channels_ignore = None

    if args.time_ignore is not None:
        time_ignore = util.make_obslist(args.time_ignore)
        time_ignore = np.array(time_ignore).astype(int)
    else:
        time_ignore = None

    occ_dict_list = []
    for obs in dict_list:
        obsid = obs['obsid']
        ins = INS(obs['ins_file'], mask_file=obs['mask_file'],
                  match_events_file=obs['yml_file'])

        if fine_channels_ignore is not None:
            freq_chans = np.arange(len(ins.freq_array))
            freq_chans = np.delete(freq_chans, fine_channels_ignore)
            ins.select(freq_chans=freq_chans)
            # Also have to remove these freqs from match_events
            new_events = []
            for event in ins.match_events:
                if (event[1].start in fine_channels_ignore) and ((event[2] == 'narrow') or (event[2] == 'samp_thresh')):
                    continue
                else:
                    new_events.append(event)
            ins.match_events = new_events

        if time_ignore is not None:
            times = ins.time_array
            times = np.delete(times, times[time_ignore])
            ins.select(times=times)
            # Also have to remove these times from match_events
            new_events = []
            for event in match_events:
                if event[0] in time_ignore:
                    continue
                else:
                    new_events.append(event)
            ins.match_events = new_events

        total_occ = np.mean(ins.metric_array.mask[:, :, 0])

        obs_occ_dict = {'obsid': obsid, 'total_occ': total_occ}
        if args.shapes is not None:
            for shape in args.shapes:
                obs_occ_dict[shape] = len([event for event in ins.match_events if event[2] == shape]) / ins.metric_array.shape[0]
        occ_dict_list.append(obs_occ_dict)

    dict_list_to_csv(occ_dict_list, args.outfile, ['obsid', 'total_occ'] + [shape for shape in args.shapes])
