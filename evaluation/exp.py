#!/usr/bin/env python3

import os
import re
import sys
import glob
import subprocess

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def prepare(args):
    '''Generates input files used to evaluate the Count-Min Sketch.

    By default creates `20` input files with `10M` values each, sampled
    from a Zipf distribution with `alpha` parameter set to the values
    in `alpha`.

    That is `runs * len(alpha)` files will be generated and saved to
    `LOCATION`. The `LOCATION` will be used as input to `run()`.

    '''
    if len(args) < 1:
        print('usage: exp prepare [OPTIONS] <LOCATION>')
        print('                   [--runs RUNS]')
        return

    # output location
    location = args[-1]
    # number of runs.
    runs = 20
    # number of operations in each run.
    count = 10000000
    # Zipf distribution alpha parameter values.
    alpha = [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]

    if len(args) > 1 and args[0] == '--runs':
        runs = int(args[1])

    print('exp: creating {} hash file(s) under {}.'.format(runs, location))

    for a in alpha:
        for r in range(runs):
            print('exp: creating hash file: {} with alpha: {}.'.format(r, a))

            nums = np.random.zipf(a, count)

            filepath = os.path.join(location, 'zipf-a{}-r{}.dat'.format(a, r))

            with open(filepath, 'w') as f:
                f.write('\n'.join(str(num) for num in nums))


def run(args):
    '''Runs the evaluation program for each of the input files.

    The experiments are executed for each input file found in `INPUT LOCATION`
    and for different values of `epsilon`. The aggregated results from all the
    input files for a specific `epsilon`, `alpha` tuple are stored in
    `OUTPUT LOCATION`.

    The `OUTPUT LOCATION` will be used as input to `plot()`.

    '''
    if len(args) < 2:
        print('usage: exp run [OPTIONS] <INPUT LOCATION> <OUTPUT LOCATION>')
        print('               [--jobs JOBS] [--exe EXECUTABLE]')
        return

    # number of threads to use.
    jobs = 1
    # Rust executable location.
    exe = './target/release/evl'
    # input hash files location.
    input_location = args[-2]
    # output location.
    output_location = args[-1]
    # epsilon values, for different accuracies.
    epsilon = [0.01, 0.001, 0.0001]

    if args[0] == '--jobs':
        jobs = int(args[1])
    elif args[0] == '--exe':
        exe = args[1]

    if len(args) > 2 and args[2] == '--jobs':
        jobs = int(args[2])
    elif len(args) > 2 and args[2] == '--exe':
        exe = args[2]

    regex = re.compile('(?<=zipf-a)(.+)-.+$')

    filenames = list(glob.glob('{}/*.dat'.format(input_location)))

    alpha = set(regex.search(filename).group(1) for filename in filenames)

    for a in alpha:
        for e in epsilon:
            message = 'exp: running evaluation with: alpha: {}, epsilon: {}.'
            print(message.format(a, e))

            prefix = 'zipf-a{}-r'.format(a)

            command = [
                exe,
                "--jobs", str(jobs),
                "--output", output_location,
                "--eps", str(e),
            ]

            command.extend(f for f in filenames if prefix in f)

            subprocess.run(command)


def plot(args):
    '''Plots the experimental results.'''
    if len(args) < 1:
        print('usage: exp plot <INPUT LOCATION>')
        return

    # input location.
    location = args[-1]

    regex = re.compile('rerr-e(.+)-zipf-a(.+)-r.+$')

    data = defaultdict(list)

    for filename in glob.glob('{}/*.dat'.format(location)):
        epsilon = float(regex.search(filename).group(1))
        alpha = float(regex.search(filename).group(2))

        minres, maxres, avgres = [], [], []

        with open(filename, 'r') as f:
            # skip the header.
            f.readline()

            for line in f.readlines():
                entries = list(map(float, line.split()))

                minres.append(entries[0])
                maxres.append(entries[1])
                avgres.append(entries[2])

        data[epsilon].append(
            (alpha, np.mean(minres), np.mean(maxres), np.mean(avgres))
        )

    fix, ax = plt.subplots()

    for (epsilon, data) in data.items():
        data_sorted = sorted(data, key=lambda x: x[0])

        alpha, minres, maxres, avgres = [], [], [], []

        for entry in data_sorted:
            alpha.append(entry[0])
            minres.append(entry[1])
            maxres.append(entry[2])
            avgres.append(entry[3])

        ax.plot(alpha, avgres, '-', linewidth=0.5,
                label=r'$\epsilon = {}$'.format(epsilon))

        ax.fill_between(alpha, minres, maxres, alpha=0.2)

    ax.set_xlim(1.1, 2.6)
    ax.set_ylim(0, 0.27)

    plt.grid(linestyle='--')

    plt.xlabel(r'Zipf: `$\alpha$` parameter (skew)')
    plt.ylabel('Mean Relative Error')

    plt.title(r'Count-Min Sketch Accuracy')

    plt.legend()

    plt.show()


def usage():
    print('usage: exp <prepare/run/plot> [OPTIONS]')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    if sys.argv[1] == 'prepare':
        prepare(sys.argv[2:])
    elif sys.argv[1] == 'run':
        run(sys.argv[2:])
    elif sys.argv[1] == 'plot':
        plot(sys.argv[2:])
    else:
        usage()
