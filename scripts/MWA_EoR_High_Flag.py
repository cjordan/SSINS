from SSINS import INS, util, Catalog_Plot, MF
import argparse
from astropy.time import Time
from matplotlib import cm
import numpy as np
from time import time
from multiprocessing import Pool

# note on performance: enabling the -p plots flag takes approximately an order
# longer than only -w writing the data files: for raw processing of data, disabling
# -p is recommended.

# for best performance, set -t to the number of available cores on the machine;
# scaling after six cores is limited

# pool doesn't like being fed multiple arguments, so args is global

# note on multithread execution scaling: this script basically runs out of gains
# at more than six cores allocated because of how the Pool scheduler works:
# Pool allocates N workers to work on N elements on the list and then waits for
# them *all* to return before issuing new work to threads.

# Since the different datasets take different amounts of time to finish executing
# many threads are stalled; with >6 cores allocated, most of the cores sit idle
# as a batch of >6 data pieces is very likely to have a slow entry within it.

# A CPU with aggressive turbo bins (capable of scaling a single thread very fast
# on demand) may somewhat alleviate this problem.

global args

def execbody (ins_filepath):
    slash_ind = ins_filepath.rfind('/')
    obsid = ins_filepath[slash_ind + 1: slash_ind + 11]

    ins = INS(ins_filepath)
    ins.select(times=ins.time_array[3:-3])
    ins.metric_ms = ins.mean_subtract()
    shape_dict = {'TV6': [1.74e8, 1.81e8],
                  'TV7': [1.81e8, 1.88e8],
                  'TV8': [1.88e8, 1.95e8],
                  'TV9': [1.95e8, 2.02e8]}
    sig_thresh = {shape: 5 for shape in shape_dict}
    sig_thresh['narrow'] = 5
    sig_thresh['streak'] = 8
    mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict,
            N_samp_thresh=len(ins.time_array) // 2)

    ins.metric_array[ins.metric_array == 0] = np.ma.masked
    ins.metric_ms = ins.mean_subtract()
    ins.sig_array = np.ma.copy(ins.metric_ms)


    #write plots if command flagged to do so
    if args.plots:
        prefix = '%s/%s_trimmed_zeromask' % (args.outdir, obsid)
        freqs = np.arange(1.7e8, 2e8, 5e6)
        xticks, xticklabels = util.make_ticks_labels(freqs, ins.freq_array, sig_fig=0)
        yticks = [0, 20, 40]
        yticklabels = []
        for tick in yticks:
            yticklabels.append(Time(ins.time_array[tick], format='jd').iso[:-4])

        Catalog_Plot.INS_plot(ins, prefix, xticks=xticks, yticks=yticks,
                              xticklabels=xticklabels, yticklabels=yticklabels,
                              data_cmap=cm.plasma, ms_vmin=-5, ms_vmax=5,
                              title=obsid, xlabel='Frequency (Mhz)', ylabel='Time (UTC)')
        if args.verbose:
            print("wrote trimmed zeromask plot for "+obsid)

    mf.apply_match_test(ins, apply_samp_thresh=False)
    mf.apply_samp_thresh_test(ins, event_record=True)

    flagged_prefix = '%s/%s_trimmed_zeromask_MF_s8' % (args.outdir, obsid)

    #write data/mask/match/ if command flagged to do so
    if args.write:
        ins.write(flagged_prefix, output_type='data', clobber=True)
        ins.write(flagged_prefix, output_type='mask', clobber=True)
        ins.write(flagged_prefix, output_type='match_events')
        if args.verbose:
            print("wrote data/mask/match files for "+obsid)

    #write plots if command flagged to do so
    if args.plots:
        Catalog_Plot.INS_plot(ins, flagged_prefix, xticks=xticks, yticks=yticks,
                              xticklabels=xticklabels, yticklabels=yticklabels,
                              data_cmap=cm.plasma, ms_vmin=-5, ms_vmax=5,
                              title=obsid, xlabel='Frequency (Mhz)', ylabel='Time (UTC)')

        if args.verbose:
            print("wrote trimmed zeromask (w/ match filter) for "+obsid)

    #a hardcoded csv generator for occ_csv
    if args.gencsv is not None:
        csv = ""+obsid+","+flagged_prefix+"_SSINS_data.h5,"+flagged_prefix+"_SSINS_mask.h5,"+flagged_prefix+"_SSINS_match_events.yml\n"
        with open(args.gencsv, "a") as csvfile:
            csvfile.write(csv)
        print("wrote entry for "+obsid)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Processes a list of SSINS data files into plots, CSV for occ_csv, and/or processed data.')
    parser.add_argument('-i', '--ins_file_list', help="A text file of SSINS h5 data files.")
    parser.add_argument('-o', '--outdir', help="The output directory for the files.")
    parser.add_argument('-w', '--write', action='store_true', help="Toggles creation of output data files.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Toggles verbose console output of file processing progress.")
    parser.add_argument('-p', '--plots', action='store_true', help="Toggles creation of plots.")
    parser.add_argument('-g', '--gencsv', help="If nonnull, creates an output CSV file for occ_csv.py with the given name (pass with extension)")
    parser.add_argument('-n', '--numthreads', type=int, default=4, help="Sets the number of threads to use for evaluation (default: 4)")
    args = parser.parse_args()

    ins_file_list = util.make_obslist(args.ins_file_list)

    if args.gencsv is not None:
        f = open(args.gencsv, "w") #wipe old file
        f.write("obsid,ins_file,mask_file,yml_file\n")#header
        f.close()

    #time the full length of the run if -v passed
    if args.verbose:
        start = time()

    if args.numthreads == 1: #single thread fallback
        print("using single thread execution path")
        for ins_filepath in ins_file_list:
            execbody(ins_filepath)
    else:#multithreaded execution
        print("using multithreaded execution path: "+str(args.numthreads)+" cores used")
        filelist = []
        for ins_filepath in ins_file_list:
            filelist.append(ins_filepath)
        p = Pool(args.numthreads)
        p.map(execbody, filelist)

    #print out full length of run
    if args.verbose:
        print(f'Time taken: {time() - start}')
