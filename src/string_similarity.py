from fuzzywuzzy import fuzz as fw
import pyxdameraulevenshtein as lev
import numpy as np
import time as t

#related to fuzzy matching
PESO_PARTIAL_RATIO = 1.2
HIGH_AVERAGE_FUZZY = 95
LOW_AVERAGE_FUZZY = 85
HIGH_SUBSTRING_FUZZY = 95
LOW_SUBSTRING_FUZZY = 85

#related to Levensthein
HIGH_LEV_DIFFERENCE = 20
LOW_LEV_DIFFERENCE = 5
LEV_TOLLERANCE = 1


# It computes the fuzzy matching score between two words
def single_fuzzmatch(w1, w2):

    w1 = w1.lower()
    w2 = w2.lower()

    fuz1 = fw.ratio(w1, w2)
    fuz2 = fw.partial_ratio(w1, w2)
    fuz3 = fw.token_set_ratio(w1, w2)

    fuzAverage = (fuz1 + fuz2 + fuz3) // 3
    return fuzAverage

# It computes the Levensthein score between two words
def single_lev(w1, w2):
    w1 = w1.lower().replace(" ", "")
    w2 = w2.lower().replace(" ", "")
    return lev.damerau_levenshtein_distance(w1, w2)

# It computes the fuzzy matching score between two set of words
def matrix_fuzzmatch(words):
    start = t.time()
    matrix = np.array([[single_fuzzmatch(w1, w2) for w1 in words] for w2 in words])
    end = t.time()
    return matrix, end-start

# It computes the Levensthein score between two set of words
def matrix_lev(words):
    start = t.time()
    matrix = np.array([[single_lev(w1, w2) for w1 in words] for w2 in words])
    end = t.time()
    return matrix, end-start

# It combines both the Fuzzy Matchind and the Levensthein scores to compute the similarity between two words
def single_wombocombo(w1, w2, dictionary,
                      high_average_fuzzy=HIGH_AVERAGE_FUZZY,
                      low_average_fuzzy=LOW_AVERAGE_FUZZY,
                      high_substring_fuzzy=HIGH_SUBSTRING_FUZZY,
                      low_substring_fuzzy=LOW_SUBSTRING_FUZZY,
                      lev_tollerance=LEV_TOLLERANCE):
    w1 = w1.lower().strip()
    w2 = w2.lower().strip()

    lev_d = single_lev(w1, w2)

    # if are distinct and both valid values, they are considered very distant to each other
    if dictionary.get(w1.lower()) is not None and dictionary.get(w2) is not None and w1 != w2:
        return lev_d + HIGH_LEV_DIFFERENCE

    # it forces very close similarity among words
    if lev_d <= lev_tollerance:
        return 0

    # if at most one word is correct, we compute the fuzzy matching score
    fuz1 = fw.ratio(w1, w2)
    fuz2 = fw.partial_ratio(w1, w2)
    fuz3 = fw.token_set_ratio(w1, w2)

    fuzAverage = (fuz1 + (fuz2*PESO_PARTIAL_RATIO) + fuz3)//3

    # it forces very close similarity among words
    if (fuzAverage >= high_average_fuzzy): 
        return 0

    # it forces large distance among words
    if (fuzAverage < low_average_fuzzy): 
        lev_d = lev_d + HIGH_LEV_DIFFERENCE

    return lev_d + LOW_LEV_DIFFERENCE

# It combines both the Fuzzy Matchind and the Levensthein scores to compute the similarity between two set of words
def wombo_combo(words, dictionary, high_average_fuzzy=HIGH_AVERAGE_FUZZY,
                low_average_fuzzy=LOW_AVERAGE_FUZZY, high_substring_fuzzy=HIGH_SUBSTRING_FUZZY,
                low_substring_fuzzy=LOW_SUBSTRING_FUZZY,
                lev_tollerance=LEV_TOLLERANCE):
    start = t.time()
    matrix = np.array([[single_wombocombo(w1, w2, dictionary, high_average_fuzzy, low_average_fuzzy,
                                          high_substring_fuzzy, low_substring_fuzzy, lev_tollerance) for w1 in words] for w2 in words])

    end = t.time()
    return matrix, end-start

def wombo_combo_matrix(words, dictionary, high_average_fuzzy=HIGH_AVERAGE_FUZZY,
                low_average_fuzzy=LOW_AVERAGE_FUZZY, high_substring_fuzzy=HIGH_SUBSTRING_FUZZY,
                low_substring_fuzzy=LOW_SUBSTRING_FUZZY,
                lev_tollerance=LEV_TOLLERANCE):
    start = t.time()
    matrix = np.matrix([[single_wombocombo(w1, w2, dictionary, high_average_fuzzy, low_average_fuzzy,
                                          high_substring_fuzzy, low_substring_fuzzy, lev_tollerance) for w1 in words] for w2 in words])

    end = t.time()
    return matrix, end-start

def single_wombocombo_word_dictionary(w1, w2):
    w1 = w1.lower().strip()
    w2 = w2.lower().strip()

    lev_d = single_lev(w1, w2)

    fuz1 = fw.ratio(w1, w2)
    fuz2 = fw.partial_ratio(w1, w2)
    fuz3 = fw.token_set_ratio(w1, w2)

    fuzAverage = (fuz1 + (fuz2*PESO_PARTIAL_RATIO) + fuz3)//3

    # it forces very close similarity among words
    if (fuzAverage >= HIGH_AVERAGE_FUZZY): 
        return 0

    # it forces large distance among words
    if (fuzAverage < LOW_AVERAGE_FUZZY): 
        lev_d = lev_d + HIGH_LEV_DIFFERENCE

    return lev_d + LOW_LEV_DIFFERENCE

def wombo_combo_word_dictionary(words, dictionary):
    start = t.time()
    matrix = np.array([[single_wombocombo_word_dictionary(w1, w2) for w1 in dictionary] for w2 in words])

    end = t.time()
    return matrix, end-start

def lev_distance_word_dictionary(words, dictionary):
    start = t.time()
    matrix = np.array([[single_lev(w1, w2) for w1 in words] for w2 in words])
    end = t.time()
    return matrix, end-start

# It counts the number of perfect matchings
def perfect_matching(words, dictionary):

    n_matching = 0

    for word in np.unique(words):
        if isinstance(word, str):
            word = word.lower()

        if dictionary.get(word) is not None:
            n_matching += 1

    return n_matching, len(np.unique(words))

def get_wrong_words(words, dictionary):
    wrong_words = []

    for word in np.unique(words):
        if isinstance(word, str):
            word = word.lower()
        
        word = word.strip()

        if dictionary.get(word) is None:
            wrong_words.append(word)

    return wrong_words

def cluster_range(words, dictionary):
    w = np.array([x.lower() if isinstance(x, str) else x for x in words])

    cluster_count = 0

    for word in np.unique(w):
        if dictionary.get(word) is not None:
            cluster_count += 1
    if cluster_count == 0:
        cluster_count = 1
    return cluster_count, len(np.unique(words))

def max_clusters(words):
    w = np.array([x.lower() if isinstance(x, str) else x for x in words])
    return len(np.unique(w))
