from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as AC
from itertools import product
import time as t
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import editdistance as edist

import string_similarity


def agglomerative_propagation(matrix, n_cluster, words):

    start = t.time()
    affinity = AC(affinity="precomputed", n_clusters=n_cluster, linkage="complete", compute_full_tree=True)
    affinity.fit(matrix)
    clusters = []

    for index in range(0, n_cluster):
        lista = []
        clusters.append(lista)

    for index in range(0, len(words)):
        clusters[affinity.labels_[index]].append(words[index])
    end = t.time()

    return affinity, clusters, end-start


# it returns the 'centroid', i.e., the correct word or the most common one
def find_samples(column, uniques, dictionary):
    maxcount = 0
    maxw = ""
    column = [x.lower() for x in column]

    for w in uniques:
        w = w.lower()
        if dictionary.get(w) is not None:
            return w
        count = column.count(w)
        if count > maxcount:
            maxcount = count
            maxw = w
    return maxw

# True if clusters should be collapsed (similar words in different clusters), False otherwise
def collapse(clusters, dictionary,
             high_average_fuzzy=string_similarity.HIGH_AVERAGE_FUZZY,
             low_average_fuzzy=string_similarity.LOW_AVERAGE_FUZZY,
             high_substring_fuzzy=string_similarity.HIGH_SUBSTRING_FUZZY,
             low_substring_fuzzy=string_similarity.LOW_SUBSTRING_FUZZY,
             lev_tollerance=string_similarity.LEV_TOLLERANCE):
    samples = []
    for i, group in enumerate(clusters):
        samples.append(find_samples(group, np.unique(group), dictionary))

    for i, w1 in enumerate(samples):
        for j, w2 in enumerate(samples):
            if i == j: continue
            w1 = w1.lower()
            w2 = w2.lower()
            if dictionary.get(w1) is not None and dictionary.get(w2) is not None and w1 != w2:
                continue
            if string_similarity.single_wombocombo(w1, w2, dictionary, high_average_fuzzy, low_average_fuzzy,
                                                   high_substring_fuzzy, low_substring_fuzzy, lev_tollerance) == 0:
                #print(i, "-", w1, " e ", j, "-", w2, " simili ma in cluster diversi")
                return False

    return True


# True if a cluster mix words that should be divided, False otherwise
def split(clusters, dictionary,
          high_average_fuzzy=string_similarity.HIGH_AVERAGE_FUZZY,
          low_average_fuzzy=string_similarity.LOW_AVERAGE_FUZZY,
          high_substring_fuzzy=string_similarity.HIGH_SUBSTRING_FUZZY,
          low_substring_fuzzy=string_similarity.LOW_SUBSTRING_FUZZY,
          lev_tollerance=string_similarity.LEV_TOLLERANCE
          ):

    present = ""

    for i, group in enumerate(clusters):
         g = np.unique([x.lower() for x in group])
         for w in g:
             w = w.lower()
             if dictionary.get(w) is not None:
                if present == "":
                    present = w
                else:
                    return True

    for i, group in enumerate(clusters):
        g = np.unique([x.lower() for x in group])
        for w1 in g:
            for w2 in g:
                if string_similarity.single_wombocombo(w1.lower(), w2.lower(), dictionary,
                                                       high_average_fuzzy, low_average_fuzzy,
                                                       high_substring_fuzzy, low_substring_fuzzy,
                                                       lev_tollerance) != 0:
                    return True


    return False

# It returns -1 if clusters should be collapsed (i.e., decreased), 0 if they are ok, 1 if they should be split
def check_clusters(clusters, dictionary, high_average_fuzzy=string_similarity.HIGH_AVERAGE_FUZZY,
                   low_average_fuzzy=string_similarity.LOW_AVERAGE_FUZZY,
                   high_substring_fuzzy=string_similarity.HIGH_SUBSTRING_FUZZY,
                   low_substring_fuzzy=string_similarity.LOW_SUBSTRING_FUZZY,
                   lev_tollerance=string_similarity.LEV_TOLLERANCE):

    if split(clusters, dictionary, high_average_fuzzy, low_average_fuzzy, high_substring_fuzzy, low_substring_fuzzy, lev_tollerance):
        return 1

    if collapse(clusters, dictionary, high_average_fuzzy, low_average_fuzzy, high_substring_fuzzy, low_substring_fuzzy, lev_tollerance):
        return -1

    return 0

# It assumes that clusters are well-formed, i.e., there is one and only one correct word for cluster
def propose_correction(clusters, dictionary):
    start = t.time()
    sample = ""
    maxval = 0

    corrections = {}

    for i, group in enumerate(clusters):
        g = np.unique(group)

        # find correct word in cluster
        for w in g:
            if dictionary.get(w.lower()) is not None:
                sample = w
                break

        #alternative approach to find the 'best' correction
        if sample == "":
            for w, d in product(g, dictionary):
                if string_similarity.single_wombocombo(w, d, dictionary) == 0:

                    val = string_similarity.single_lev(w, d)

                    if (val >1):
                        val = string_similarity.single_fuzzmatch(w, d)
                    else:
                        val = 100

                    if sample == "":
                        sample = dictionary.get(d)
                        maxval = val
                    elif maxval < val:
                        sample = dictionary.get(d)
                        maxval = val

        # it corrects the cluster by the sample word
        if sample != "":
            for j, el in enumerate(group):

                if clusters[i][j].lower().strip() != sample.lower().strip():
                    corrections[clusters[i][j]] = sample
                    #print("Corretto", clusters[i][j], "con", sample)
                clusters[i][j] = str(sample)

        sample = ""

    end = t.time()
    correction_time = end-start

    return clusters, corrections, correction_time


#?????? fare versione gerenale, nel caso in cui ci sono 2 parole nel dizionario che corrispondono
def propose_correction_general(clusters, dictionary):

    samples = []
    for i, group in enumerate(clusters):
        g = np.unique(group)

        # 1st case: no cluster element in the dictionary

        for w, d in product(g, dictionary):
            if string_similarity.single_wombocombo(w, d, dictionary) == 0:
                samples.append(dictionary.get(d))


        # 2nd case: more valid values in the dctionary
        print(group)
        print(samples)
        print("-----")
        for j, el in enumerate(group):
                for s in samples:
                    if string_similarity.single_wombocombo(s, el, dictionary) == 0:
                        clusters[i][j] = str(s)

        samples = []

    return clusters

