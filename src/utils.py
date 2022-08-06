import numpy as np
import math


# randomly mutates the given nucleotide (nuc) into another nucleotide (all with equal probability)
# used when testing the algorithm
def mutate(nuc, conf_level, added_mut_chance=0, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    nucleotides = ['A', 'C', 'G', 'T']
    nucleotides.remove(nuc)
    if rng.rand() + added_mut_chance > conf_level:
        return nucleotides[rng.randint(len(nucleotides))]
    else:
        return nuc


# Finds interesting stats about the probabilites in the dataset for a given word size
# returns (mean, std_dev, min, max, %of_words_below_p) of the probabilities of words
def word_stats(sequence, confidence, w, p=0.70):
    min_prob = 1
    max_prob = 0
    num_below_p = 0
    all_probs = []
    for i in range(0, len(sequence) - w + 1):
        new_prob = np.prod(confidence[i: i + w])
        if new_prob < min_prob:
            min_prob = new_prob
        if new_prob > max_prob:
            max_prob = new_prob
        if new_prob < p:
            num_below_p += 1
        all_probs.append(new_prob)
    mean = np.average(all_probs)
    stddev = np.std(all_probs)
    percent_below_p = num_below_p/len(all_probs)
    return mean, stddev, min_prob, max_prob, percent_below_p


def print_alignment(seq1, seq2, header=None, line_len=80):
    if header is not None:
        print('Alingment of query segment', header[1], 'and library segment', header[2], 'with score =', '{:.3f}'.format(header[0]))
    for chunk in range(0, math.ceil(len(seq1) / line_len)):
        chunk_start = chunk * line_len
        chunk_end = min(len(seq1), (chunk + 1) * line_len)
        print('query:\t\t', seq1[chunk_start:chunk_end], chunk_end)
        match_string = ''
        for i in range(chunk_start, chunk_end):
            if seq1[i] == seq2[i]:
                match_string = match_string + '|'
            else:
                match_string = match_string + ' '
        print('\t\t', match_string)
        print('library:\t', seq2[chunk_start:chunk_end], chunk_end)
        print()
    print()


# I use "coverage" to determine the quality of an alignment
# It measures what proportion of the true range (i.e., the one used to generate the query) overlaps with the one found by probaBLAST.
def coverage(true_range, algn_range):
    true_start, true_stop = true_range
    algn_start, algn_stop = algn_range
    prefix_error = abs(true_start - algn_start)
    suffix_error = abs(true_stop - algn_stop)
    overlap = max(0, (min(true_stop, algn_stop) - max(true_start, algn_start)))
    cov = overlap/(prefix_error + suffix_error + overlap)
    return cov


# generates a subsequence with a length between min_len and max_len (both inclusive)
def get_subseq_indices(seq, min_len, max_len, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    length = rng.randint(min_len, high=max_len + 1)
    start_pos = rng.randint(0, high=len(seq) - length + 1)
    end_pos = start_pos + length
    return start_pos, end_pos


# attempts to mutate a subsequence at every nucleotide, with the probability of a mutation equal to (1 - confidence level) at that nucleotide
def randomize_subseq(seq, conf, start_pos, end_pos, added_mut_chance=0, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    seq = seq[start_pos: end_pos]
    conf = conf[start_pos: end_pos]
    length = end_pos - start_pos
    lst = []
    for i in range(0, length):
        lst.append(mutate(seq[i], conf[i], added_mut_chance=added_mut_chance, rng=rng))
    output = ''.join(lst)
    return output


# returns a mutated subsequence with length between min_len and max_len (both inclusive), with mutation frequency at each nucleotide inversly proportional to the confidence at that nucleotide
# also returns the indices of the library used to generate the sequence, for testing the accuracy of the alignment.
def get_rand_subseq(seq, conf, min_len, max_len, rng=None, indel_coeff=0.04, avg_indel_len=3, added_mut_chance=0,
                    verbose=False):
    if rng is None:
        rng = np.random.RandomState()
    start_pos, end_pos = get_subseq_indices(seq, min_len, max_len, rng=rng)
    mutated = randomize_subseq(seq, conf, start_pos, end_pos, added_mut_chance=added_mut_chance, rng=rng)
    gapped = gapify(mutated, indel_coeff, rng=rng, avg_length=avg_indel_len)
    if verbose:
        print((start_pos, end_pos))
    return gapped, (start_pos, end_pos)


# generates random nucleotides of a given length (for simulating insertions)
def rand_nucleotides(length, rng=None, p=None):
    if rng is None:
        rng = np.random.RandomState()
    nucleotides = ['A', 'C', 'G', 'T']
    sample = rng.choice(nucleotides, size=length, p=p).tolist()
    return ''.join(sample)


# randomly adds and removes nucleotides from a sequence to simulate insertions and deletions
#
# InDel mutation chance paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4563830/
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2734402/
def gapify(seq, avg_indel_proportion, rng=None, avg_length=3):
    if avg_indel_proportion <= 0:
        return seq
    if rng is None:
        rng = np.random.RandomState()
    avg_num_indels = (len(seq) * avg_indel_proportion) / avg_length
    num_indels = rng.negative_binomial(avg_num_indels, 0.5)
    insert_or_delete = rng.randint(0, 2,
                                   size=num_indels)  # currently, probabilites for insertion and deletion are equal. 0 => insert, 1 => delete

    indices = rng.choice(len(seq), size=num_indels, replace=False).tolist()
    indices = [0] + indices
    indices.sort()
    split = [seq[i:j] for i, j in zip(indices, indices[1:] + [None])]

    for i in range(0, num_indels):
        length = rng.poisson(lam=avg_length)  # length of the indel
        if insert_or_delete[i] == 0:
            # insertion
            split[i + 1] = rand_nucleotides(length, rng=rng) + split[i + 1]
        else:
            # deletion
            split[i + 1] = split[i + 1][length:]
    output = ''.join(split)
    return output

