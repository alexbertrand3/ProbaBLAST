import time
import numpy as np

from utils import *


# coefficient for matches
# conf[i] - (1 - conf[i])/3
# = (4/3)conf[i] - (1/3)
# Equals 1 at conf[i] = 1, 0 at conf[i] = 0.25 (i.e., confidence is no better than random)
# Can also raise the above to a power to change the shape of the curve without affecting its important properties
def score_coeff(conf):
    return ((4 / 3) * conf) - (1 / 3)


def score_word(query, lib_seq, conf):
    out = 0
    for i in range(0, len(query)):
        out += score_coeff(conf[i]) * score(query[i], lib_seq[i])
    return out


# scores a nucleotide pair
def score(n1, n2, match=1, mismatch=-2):
    if n1 != n2:
        return mismatch
    else:
        return match


# Searches the library/database to find indices of all the words in the query sequence.
# filters out library matches that are overlapping to save computation time
def search_lib(query, lib, w):
    matches = []
    for i in range(0, len(query) - w + 1):
        word = query[i:i + w]
        if word in lib:
            matches.append((i, lib[word]))  # a tuple of (index of word in query, [indices of word in database])
    return matches


def ungapped_extension(query, lib_seq, conf, lib_i, q_i, step, D=10):
    best_i = (q_i - step, lib_i - step)  # the indices of the max score in the query and library sequence
    cur_score = 0
    max_score = 0
    while (
            (0 <= q_i < len(query)) and  # ensure indices are in bounds
            (0 <= lib_i < len(lib_seq)) and  # ensure indices are in bounds
            (max_score - cur_score < D)  # stop extension if score drops too far below the maximum
    ):
        cur_score += score_coeff(conf[lib_i]) * score(lib_seq[lib_i], query[q_i])
        if cur_score > max_score:
            max_score = cur_score
            best_i = (q_i, lib_i)
        q_i += step
        lib_i += step
    return max_score, best_i


# A version of the Needleman-Wunsch algorithm that accounts for confidence, for gapped extension
def needleman_wunsch(S, T, T_conf, gap=-2):
    S = '-' + S
    T = '-' + T
    T_conf = np.concatenate(([0], T_conf))

    best_index = (0, 0)
    best_score = 0

    # Initialization: M is scoring matrix, trace is trace-back matrix
    M = np.full((len(S), len(T)), -float('inf'))
    trace = np.empty((len(S), len(T)), dtype=object)
    for i in range(0, len(S)):
        M[i, 0] = i * gap
        trace[i, 0] = (i - 1, 0)
    for j in range(0, len(T)):
        M[0, j] = j * gap
        trace[0, j] = (0, j - 1)

    # Performing NW algorithm
    for i in range(1, len(S)):
        for j in range(1, len(T)):
            score_match = M[i - 1, j - 1] + (score_coeff(T_conf[j]) * score(S[i], T[j]))
            score_T_gap = M[i - 1, j] + gap
            score_S_gap = M[i, j - 1] + gap
            max_score = max(score_match, score_T_gap, score_S_gap)
            M[i, j] = max_score
            if max_score >= best_score:
                best_score = max_score
                best_index = (i, j)

            if score_match == max_score:
                trace[i, j] = (i - 1, j - 1)
            elif score_T_gap == max_score:
                trace[i, j] = (i - 1, j)
            elif score_S_gap == max_score:
                trace[i, j] = (i, j - 1)

    # Trace back and return the aligned sequence
    i, j = best_index
    S_str = ''
    T_str = ''
    while i > 0 or j > 0:
        from_i, from_j = trace[i, j]
        if i - from_i == 1:
            if j - from_j == 1:
                S_str = S[i] + S_str
                T_str = T[j] + T_str
            else:
                T_str = '-' + T_str
                S_str = S[i] + S_str
        else:
            S_str = '-' + S_str
            T_str = T[j] + T_str
        i, j = trace[i, j]
    return best_score, (S_str, T_str)


def ProbaBLAST(query, lib_seq, lib, conf, w=7, get_top=5, fit_threshold=0.25, gap=-2, verbose=False):
    start_time = time.time()
    query = query.upper()
    matches = search_lib(query, lib, w)

    lib_time = time.time()
    if verbose:
        print('Searching database completed in ', lib_time - start_time, 'seconds')

    all_scores = []
    for q_i, indices in matches:
        for lib_i in indices:
            score_init = score_word(query[q_i:q_i + w], lib_seq[lib_i:lib_i + w], conf[lib_i:lib_i + w])
            score_L, l_i = ungapped_extension(query, lib_seq, conf, lib_i - 1, q_i - 1, -1)  # Search left of the word
            score_R, r_i = ungapped_extension(query, lib_seq, conf, lib_i + w, q_i + w, 1)  # Search right of the word
            score_tot = (score_init + score_R + score_L)
            query_indices = (l_i[0], r_i[0])
            library_indices = (l_i[1], r_i[1])
            all_scores.append((score_tot, query_indices, library_indices))
    ug_time = time.time()
    if verbose:
        print('Ungapped extension completed in ', ug_time - lib_time, 'seconds')

    all_scores = sorted(all_scores, key=lambda x: (
    x[1][0], x[2][0]))  # sort by the indices used, so repeated entries (generated from nearby words) can be removed
    # remove repeated entries to save computation
    temp = [all_scores[0]]
    i = 1
    while i < len(all_scores):
        if (all_scores[i][1] != temp[-1][1] or all_scores[i][2] != temp[-1][2]):
            temp.append(all_scores[i])
        i += 1

    all_scores = sorted(temp, key=lambda x: x[0],
                        reverse=True)  # sort by scores descending, so we only need to look at the best ones

    threshold = 15  # Arbitary threshold that an HSP must be above to consider for gapped extension
    outputs = []
    for entry in all_scores:
        if entry[0] >= threshold:

            hsp_start, hsp_stop = entry[1]  # start and end indices of the HSP in the query
            lib_start, lib_stop = entry[2]  # start and end indices of the HSP in the library

            # if the current segment is enclosed by one on which needleman-wunsch was performed, skip it to save computation
            # This can result in some alignments being *slightly* suboptimal
            # (from my testing it did not significantly change the indices aligned, just occasionally changed which residues were matched/where gaps were inserted)
            enclosed_hsp = False
            for previous_nw in outputs:
                if previous_nw[1][0] <= hsp_start and hsp_stop <= previous_nw[1][1]:
                    enclosed_hsp = True
            if enclosed_hsp:
                continue

            # prefix of query must be reversed to work l with needleman wunsch implementation
            query_prefix = query[0:hsp_start][::-1]

            # searches for up to (1.5x + 5) times the length of the query segment to ensure the best match is found. This is arbitrary and may be somewhat computationally wasteful, and could surely be improved.
            # the 5 is needed for cases where the segment being searched is very short
            lib_prefix = lib_seq[max(0, lib_start - int(1.5 * len(query_prefix)) - 5): lib_start][::-1]
            conf_prefix = conf[max(0, lib_start - int(1.5 * len(query_prefix)) - 5): lib_start][::-1]
            pre_score, (pre_query_str, pre_lib_str) = needleman_wunsch(query_prefix, lib_prefix, conf_prefix, gap=gap)

            # re-reverse the strings to return them to the proper order. Size is the length without gaps, used to find the exact query/library indices used in the optimal alignment
            pre_query_str = pre_query_str[::-1]
            pre_query_size = len(pre_query_str) - pre_query_str.count('-')
            pre_lib_str = pre_lib_str[::-1]
            pre_lib_size = len(pre_lib_str) - pre_lib_str.count('-')
            query_suffix = query[hsp_stop:len(query)]

            lib_suffix = lib_seq[lib_stop: lib_stop + min(len(lib_seq), int(1.5 * len(query_suffix))) + 5]
            conf_suffix = conf[lib_stop: lib_stop + min(len(lib_seq), int(1.5 * len(query_suffix))) + 5]
            post_score, (post_query_str, post_lib_str) = needleman_wunsch(query_suffix, lib_suffix, conf_suffix,
                                                                          gap=gap)

            post_query_size = len(post_query_str) - post_query_str.count('-')
            post_lib_size = len(post_lib_str) - post_lib_str.count('-')

            query_aligned_indices = (hsp_start - pre_query_size, hsp_stop + post_query_size)
            lib_aligned_indices = (lib_start - pre_lib_size, lib_stop + post_lib_size)

            score_tot = pre_score + entry[0] + post_score

            # string representations of the alignment indicating gaps
            query_aligned = pre_query_str + query[hsp_start:hsp_stop] + post_query_str
            lib_aligned = pre_lib_str + lib_seq[lib_start:lib_stop] + post_lib_str

            # a version that shows the separation between the ungapped and needleman-wunsch portions of the algorithm, useful for debugging
            # query_aligned = pre_query_str + '|' + query[hsp_start:hsp_stop] + '|' + post_query_str
            # lib_aligned = pre_lib_str + '|' + lib_seq[lib_start:lib_stop] + '|' + post_lib_str

            outputs.append((score_tot, query_aligned_indices, lib_aligned_indices, (query_aligned, lib_aligned)))
        else:
            break
    nw_time = time.time()
    if verbose:
        print('Needleman Wunsch extensions completed in ', nw_time - ug_time, 'seconds')

    outputs = sorted(outputs, key=lambda x: x[0], reverse=True)
    # for out in outputs[0:get_top]:
    # print(out[0:3])
    # print('query seq: \t', out[3][0])
    # print('library: \t', out[3][1])
    # print('\n')
    return outputs