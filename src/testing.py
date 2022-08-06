import numpy as np
import matplotlib.pyplot as plt

from build_lib import *
from probaBLAST import *


# load the library sequence and confidence values
# Data is a portion (~600 kilobases) of the predicted chromosome 22 of an ancestral Boreoeutherian (a sub-type of placental mammals)
with open('../data/sequence.txt') as f1:
    library_seq = f1.readline().upper()

with open('../data/confidence.txt') as f2:
    conf = f2.readline().split()
    conf = np.asarray([float(x) for x in conf])
    conf_comp = (1 - conf)/3      # (comp --> complement). The probability for each of the non-predicted nucleotides. Assumes equal probabilities for each.

'''Run this if you don't already have a library saved as a file!'''
# lib = build_library(library_seq, conf, 7, p=0.8)
# save_library(lib, '../data/test_lib_w7.json', indent=4)

lib = load_library('../data/test_lib_w7.json')


# test on long sequence (5000 bp). Takes about 2 minutes on Colab, from my testing. Requires ~2.5 GB of memory.
def test_long_seq():
    seed = 22329
    rng = np.random.RandomState(seed=seed)
    query, true_inds = get_rand_subseq(library_seq, conf, 5000, 5001, indel_coeff=0.04, added_mut_chance=0.0, rng=rng, verbose=True)
    print(query)
    print(library_seq[true_inds[0]:true_inds[1]])
    print()
    likely_candidates = ProbaBLAST(query, library_seq, lib, conf)

    for alignment in likely_candidates:
        header = (alignment[0], alignment[1], alignment[2])
        print_alignment(alignment[3][0], alignment[3][1], header=header)


# test, with the runtimes of each phase of the algorithm.
def test_runtimes():
    seed = 2370089
    rng = np.random.RandomState(seed=seed)
    query, true_inds = get_rand_subseq(library_seq, conf, 20, 1000, indel_coeff=0.15, rng=rng, verbose=True)
    print('Sequence generated from library location: ', true_inds, '\t\tQuery length: ', len(query))
    print(query)
    print(library_seq[true_inds[0]:true_inds[1]])
    print()
    likely_candidates = ProbaBLAST(query, library_seq, lib, conf, verbose=True)

    for alignment in likely_candidates:
        score = alignment[0]
        query_inds = alignment[1]
        algn_inds = alignment[2]
        header = (score, query_inds, algn_inds)
        print_alignment(alignment[3][0], alignment[3][1], header=header)
        print('Coverage score:', coverage(true_inds, algn_inds))


# Get a baseline accuracy using the default random sequence parameters, based on 100 random sequences
def baseline_acc():
    seed = 123456789
    rng = np.random.RandomState(seed=seed)
    accuracies = []

    for i in range(100):
        query, true_inds = get_rand_subseq(library_seq, conf, 50, 500, indel_coeff=0.04, rng=rng, verbose=False)
        likely_candidates = ProbaBLAST(query, library_seq, lib, conf, verbose=False)
        best_cov = 0.0
        for alignment in likely_candidates:
            algn_score = alignment[0]
            algn_inds = alignment[2]
            cov = coverage(true_inds, algn_inds)
            if cov > best_cov:
                best_cov = cov
        accuracies.append(best_cov)
        if i % 10 == 0 and i > 0:
            print('...done', i)

    print('average coverage =', np.average(accuracies), 'with standard deviation =', np.std(accuracies))
    plt.hist(accuracies)
    plt.ylabel('Count')
    plt.xlabel('Alignment Coverage')
    plt.title('Baseline ProbaBLAST Performance')
    plt.show()


# Test increasing mutation frequencies (no indels).
def mutation_freqs():
    levels = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    seed = 123456789
    accs_per_level = []
    std_devs_per_level = []

    for entry in levels:
        rng = np.random.RandomState(seed=seed)
        accuracies = []
        for i in range(100):
            query, true_inds = get_rand_subseq(library_seq, conf, 50, 500, indel_coeff=0.0, added_mut_chance=entry, rng=rng, verbose=False)
            likely_candidates = ProbaBLAST(query, library_seq, lib, conf, verbose=False)
            best_cov = 0.0
            for alignment in likely_candidates:
                algn_score = alignment[0]
                algn_inds = alignment[2]
                cov = coverage(true_inds, algn_inds)
                if cov > best_cov:
                    best_cov = cov
            accuracies.append(best_cov)
        accs_per_level.append(np.average(accuracies))
        std_devs_per_level.append(np.std(accuracies))
        # print(sorted(accuracies))
        print('done', entry)

    plt.errorbar(levels, accs_per_level, yerr=std_devs_per_level, capsize=3.0, fmt='-o')
    plt.ylabel('Average Coverage')
    plt.xlabel('Added Mutation Chance')
    plt.title('Performance with Increasing Mutation Chance')
    plt.show()


# Test increasing indel rates
def indel_rates():
    levels = [0, 0.02, 0.04, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1]
    seed = 123456789
    accs_per_level = []
    std_devs_per_level = []

    for entry in levels:
        rng = np.random.RandomState(seed=seed)
        accuracies = []
        for i in range(100):
            query, true_inds = get_rand_subseq(library_seq, conf, 50, 500, indel_coeff=entry, rng=rng, verbose=False)
            likely_candidates = ProbaBLAST(query, library_seq, lib, conf, gap=-1, verbose=False)
            best_cov = 0.0
            for alignment in likely_candidates:
                algn_score = alignment[0]
                algn_inds = alignment[2]
                cov = coverage(true_inds, algn_inds)
                if cov > best_cov:
                    best_cov = cov
            accuracies.append(best_cov)
        accs_per_level.append(np.average(accuracies))
        std_devs_per_level.append(np.std(accuracies))
        print('done', entry)

    plt.errorbar(levels, accs_per_level, yerr=std_devs_per_level, capsize=3.0, fmt='-o')
    plt.ylabel('Average Coverage')
    plt.xlabel('Indel Rate')
    plt.title('Performance with Increasing Indel Rates')
    plt.show()


# Test increasing average indel lengths
def indel_lengths():
    levels = [2, 3, 4, 5, 7, 10, 15, 20, 30]
    seed = 123456789
    accs_per_level = []
    std_devs_per_level = []

    for entry in levels:
        rng = np.random.RandomState(seed=seed)
        accuracies = []
        for i in range(100):
            query, true_inds = get_rand_subseq(library_seq, conf, 50, 500, avg_indel_len=entry, indel_coeff=0.04*entry, rng=rng, verbose=False)
            likely_candidates = ProbaBLAST(query, library_seq, lib, conf, verbose=False)
            best_cov = 0.0
            for alignment in likely_candidates:
                algn_score = alignment[0]
                algn_inds = alignment[2]
                cov = coverage(true_inds, algn_inds)
                if cov > best_cov:
                    best_cov = cov
            accuracies.append(best_cov)
        accs_per_level.append(np.average(accuracies))
        std_devs_per_level.append(np.std(accuracies))
        print('done', entry)

    plt.errorbar(levels, accs_per_level, yerr=std_devs_per_level, capsize=3.0, fmt='-o')
    plt.ylabel('Average Coverage')
    plt.xlabel('Average Indel Length')
    plt.title('Performance with Increasing Indel Length')
    plt.show()


# Produces a graph showing alternative coefficient-generating functions
def coeff_graph():
    x = np.linspace(0.25,1, 100)
    base_func = (4/3) * x - (1/3)
    square = base_func ** 2
    root = np.sqrt(base_func)
    log4 = 1 + np.log(x)/np.log(4)
    plt.plot(x, base_func, label = 'linear')
    plt.plot(x, square, label = 'squared')
    plt.plot(x, root, label = 'square root')
    plt.plot(x, log4, label = '1 - log4')
    plt.xlabel('Confidence')
    plt.ylabel('Coefficient')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_long_seq()
    test_runtimes()
    baseline_acc()
    mutation_freqs()
    indel_rates()
    indel_lengths()
    coeff_graph()
