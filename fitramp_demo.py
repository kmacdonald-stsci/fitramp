#!/usr/bin/env python
# coding: utf-8

import numpy as np
import fitramp
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time

################## DEBUG ##################
#                  HELP!!
import ipdb
import sys

sys.path.insert(1, "/Users/kmacdonald/code/common")
from general_funcs import DELIM, dbg_print

################## DEBUG ##################


def ndstr(arr, prec=4):
    return np.array2string(
            arr, precision=prec, max_line_width=np.nan, separator=", ")


def get_params():
    # Npix = 5000                          # pixels per row
    # nrows = 1000                         # number of rows of pixels

    Npix = 4  # pixels per row
    nrows = 1  # number of rows of pixels

    # countrate = 2 * np.ones((nrows, Npix))  # count rate for each pixel
    # sig = 20 * np.ones((nrows, Npix))  # uncertainty for each pixel

    countrate = 20 * np.ones((nrows, Npix))  # count rate for each pixel
    sig = 0.05 * np.ones((nrows, Npix))  # uncertainty for each pixel

    '''
    readtimes = [
        1,
        2,
        3,
        [4, 5],
        [6, 7, 8],
        [10, 11, 13],
        [15, 18],
        [21, 22],
        23,
        [25, 26],
    ]
    '''
    readtimes = [k+1 for k in range(10)]

    return Npix, nrows, countrate, sig, readtimes


def print_im_diff(im, diff, p=False):
    if p:
        for row in range(im.shape[1]):
            for col in range(im.shape[2]):
                im_str = ndstr(im[:, row, col])
                diff_str = ndstr(diff[:, row, col])
                dbg_print(f"    **** ({row}, {col}) ****")
                print(f"im: {im_str}")
                print(f"diff: {diff_str}\n")


def generate_data(Npix, nrows, countrate, sig, readtimes, C):
    # Generate dummy data
    im = np.empty((len(readtimes), nrows, Npix))

    # import ipdb; ipdb.set_trace()
    for i in range(nrows):
        im[:, i] = fitramp.get_ramps(countrate[i], sig[i], readtimes, nramps=Npix)
    # import ipdb; ipdb.set_trace()
    # dbg_print(f"im = \n{repr(im)}")
    print(DELIM)
    dbg_print(f"im = \n{im}")
    print(DELIM)

    diff = (im[1:] - im[:-1]) / C.delta_t[:, np.newaxis, np.newaxis]
    print_im_diff(im, diff, False)

    return im, diff


def ramp_fit_row_loop1(Npix, nrows, countrate, sig, readtimes, im, diff, C, t0, debias=True):
    for i in range(nrows):
        # diff[:, i] = diff[:, i, :], i.e., it's the ith row.
        # XXX WTF is happening here?  Fit each ramp on each row, then
        #     just drop the reult on the floor when fitting the ramp
        #     on the next row?  This does not make sense to me.
        result = fitramp.fit_ramps(diff[:, i], C, sig[i])
        print(DELIM)
        dbg_print(f"result = \n{repr(result)}")
        repr(result)
        print(DELIM)
        sys.exit(1)
        import ipdb; ipdb.set_trace()

        if debias:
            countrateguess = result.countrate * (result.countrate > 0)
            result = fitramp.fit_ramps(
                diff[:, i], C, sig[i], countrateguess=countrateguess
            )

    dbg_print(
        "Time per H4RG: %.3g seconds"
        % ((time.time() - t0) * 4096**2 / (Npix * nrows))
    )

    dbg_print(
        "Time per 1e8 pixel-resultants: %.3g seconds"
        % ((time.time() - t0) * 1e8 / np.prod(diff.shape))
    )


def ramp_fit_row_loop_with_ped(Npix, nrows, countrate, sig, readtimes, im, diff, C, debias=True):
    C_wped = fitramp.Covar(readtimes, pedestal=True)
    d_wped = np.zeros(im.shape)
    d_wped[0] = im[0] / C.mean_t[0]
    d_wped[1:] = (im[1:] - im[:-1]) / C.delta_t[:, np.newaxis, np.newaxis]

    # dbg_print(f"Loop: nrows = {nrows}")
    for i in range(nrows):
        result = fitramp.fit_ramps(
            d_wped[:, i], C_wped, sig[i], resetval=0, resetsig=np.inf
        )

    # dbg_print("Bias calculations")
    sig_bias = 10
    countrates = 10 ** (np.linspace(-2, 4, 200))
    cvec = np.ones(diff.shape[0])
    bias_constant_c = C.calc_bias(countrates, sig_bias, cvec)

    return sig_bias, countrates, cvec, bias_constant_c


def mask_jumps_row_loop1(Npix, nrows, countrate, sig, diff, C):
    t0 = time.time()

    # dbg_print(f"Loop: nrows = {nrows}")
    for i in range(nrows):

        diffs2use, countrates = fitramp.mask_jumps(
            diff[:, i], C, sig[i], threshold_oneomit=20.25, threshold_twoomit=23.8
        )

        result = fitramp.fit_ramps(
            diff[:, i],
            C,
            sig[i],
            diffs2use=diffs2use,
            countrateguess=countrates * (countrates > 0),
        )

    dbg_print(
        "Time per H4RG with jump detection & debiasing: %.3g seconds"
        % ((time.time() - t0) * 4096**2 / (nrows * Npix))
    )

    dbg_print(
        "Time per 1e8 pixel-resultants with jump detection & debiasing: %.3g seconds"
        % ((time.time() - t0) * 1e8 / np.prod(diff.shape))
    )


def main():
    Npix, nrows, countrate, sig, readtimes = get_params()

    # Compute covariance matrix
    C = fitramp.Covar(readtimes)

    im, diff = generate_data(Npix, nrows, countrate, sig, readtimes, C)

    t0 = time.time()

    debias = True
    ramp_fit_row_loop1(Npix, nrows, countrate, sig, readtimes, im, diff, C, t0, debias)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    # ipdb.set_trace()

    sig_bias, countrates, cvec, bias_constant_c = ramp_fit_row_loop_with_ped(
            Npix, nrows, countrate, sig, readtimes, im, diff, C)

    plt.figure(figsize=(6, 4))
    plt.title("Count Rate vs. Bias")
    rcParams['font.size'] = 13.5
    plt.semilogx(countrates, bias_constant_c, linewidth=3)
    plt.xlabel("Count Rate ($e^-{\\rm s}^{-1}$)")
    plt.ylabel("Bias ($e^-{\\rm s}^{-1}$)")
    plt.show(block=False)

    mask_jumps_row_loop1(Npix, nrows, countrate, sig, diff, C)

    ijump = 2
    x, y = np.meshgrid(np.arange(diff.shape[2]), np.arange(diff.shape[1]))
    diff = (im[1:] - im[:-1]) / C.delta_t[:, np.newaxis, np.newaxis]
    diff[ijump] += 5 * np.exp(
        -((x - x.mean()) ** 2 + (y - y.mean()) ** 2) / (2 * 200**2)
    )

    alljumps = np.zeros(diff.shape)
    alljumpsigs = np.zeros(diff.shape)

    # dbg_print(f"Loop: nrows = {nrows}")
    for i in range(nrows):
        diffs2use, countrates = fitramp.mask_jumps(
            diff[:, i], C, sig[i], threshold_oneomit=20.25, threshold_twoomit=23.8
        )
        ct = countrates * (countrates > 0)
        result = fitramp.fit_ramps(
            diff[:, i],
            C,
            sig[i],
            diffs2use=diffs2use,
            detect_jumps=True,
            countrateguess=ct,
        )

        alljumps[:, i] = result.jumpval_oneomit
        alljumpsigs[:, i] = result.jumpsig_oneomit

        for j in range(len(diff)):
            indx = diffs2use[j] == 0  # only need to redo these differences

            if np.sum(indx) == 0:
                continue

            # each time we'll make sure that this difference isn't masked
            mask = diffs2use[:, indx] * 1
            mask[j] = 1

            result = fitramp.fit_ramps(
                diff[:, i, indx],
                C,
                sig[i, indx],
                diffs2use=mask,
                detect_jumps=True,
                countrateguess=ct[indx],
            )

            # Overwrite the jump value if it was previously masked.
            alljumps[j, i, indx] = result.jumpval_oneomit[j]
            alljumpsigs[j, i, indx] = result.jumpsig_oneomit[j]

    plt.figure(figsize=(12, 4))
    maxval = np.amax(np.abs(alljumps[ijump]))*0.5
    dbg_print("imshow 1")
    plt.imshow(alljumps[ijump], origin='lower', cmap='seismic',
               vmin=-maxval, vmax=maxval)
    plt.title("Jump Value (from $\chi^2$ analysis)")
    plt.show(block=False)


    plt.figure(figsize=(12, 4))
    dbg_print("imshow 2")
    plt.imshow(diff[ijump] - np.median(diff, axis=0), origin='lower', cmap='seismic',
               vmin=-maxval, vmax=maxval)
    plt.title("Jump Value (from single difference)")
    plt.show()


if __name__ == "__main__":
    main()
