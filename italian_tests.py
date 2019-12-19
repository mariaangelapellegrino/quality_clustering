from src import manage_data, mycluster, string_similarity
import time as t
import numpy as np
import os
import datetime
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import AgglomerativeClustering as AC

import matplotlib.pyplot as plt
import matplotlib.cm as cm

dictionary_municipalities = manage_data.load_comuni()
array_municipalities = manage_data.load_comuni_as_array()

test_map = {
	"2017-Allievi-Ritirati" : "ComuneResidenza",
    "Agriturismi-Napoli" : "COMUNE AZIENDA",
    "Albo-Regionale-Cooperative-Sociali" : "ComuneSedeLegale",
    "Associazioni-Giovanili" : "Città",
    "Elenco-Comunita-Minori" : "Comune",
    "Elenco-Parafarmacie" : "DESCRIZIONECOMUNE",
    "Fattorie-didattiche" : "Sede operativa",
    "Immobili-demaniali" : "Comune",
    "dataset01" : "Città",
    "dataset03" : "Comune_ubicazione_studio_medico_struttura",
    "Indirizzi-Regione-Campania" : "Comune",
    "Istituti-Scolastici-Superiori" : "COMUNE",
    "Patrimonio-indisponibile" : "Comune",
    "Pediatri-Libera-Scelta" : "ComuneStudioMedico_Struttura",
    "Registro-Volontariato" : "Comune",
    "Territorio-Regione-Campania" : "Comune",
    "Agriturismi-Caserta" : "COMUNE AZIENDA",
    "Agriturismi-Avellino" : "COMUNE AZIENDA",
    "Agriturismi-Benevento" : "COMUNE AZIENDA",
    "Rifiuti_urbani_Benevento2017" : "Comune",
    "Rifiuti_urbani_Avellino2017" : "Comune",
    "Rifiuti_urbani_Caserta2017" : "Comune",
    "Rifiuti_urbani_Salerno2017" : "Comune",
    "Rifiuti_urbani_Napoli2017" : "Comune",
    "sportelli" : "Comune_anno_riferimento_dati",
    "Autovetture_Avellino2018" : "COMUNE",
    "Autovetture_Benevento2018" : "COMUNE",
    "Autovetture_Caserta2018" : "COMUNE",
    "Autovetture_Salerno2018" : "COMUNE",
    "Autovetture_Napoli2018" : "COMUNE",
    "2016-Enti-Accreditati" : "ComuneEnte",
    "Anagrafica-e-geolocalizzazione" : "Comune",
    "2017-Enti-Accreditati" : "ComuneEnte",
    "Elenco-Istituti-Tecnici-Gia-Costituiti" : "Comune",
    "Elenco-Istituti-Tecnici-Superiori-Nuova-Costituzione" : "Comune",
    "Strutture-Private-Accreditate" : "COMUNE_SEDE_OPERATIVA",
    "Elenco-Farmacie" : "DESCRIZIONE COMUNE",
    "Motocicli_Salerno2018" : "COMUNE",
    "Motocicli_Caserta2018" : "COMUNE",
    "Motocicli_Napoli2018" : "COMUNE",
    "Motocicli_Avellino2018" : "COMUNE",
    "Motocicli_Benevento2018" : "COMUNE",
    "2015-Enti-Accreditati" : "ComuneEnte",
    "2014-Enti-Accreditati" : "ComuneEnte",
    "APL-AgenziaPerIlLavoro" : "Comune",
    "Elenco-Negozi-e-Botteghe-a-Rilevanza-Storica" : "Comune",
    "Promozione-Culturale-1" : "Citta",
    "Promozione-Culturale-2" : "Citta",
    "Registro-delle-Imprese-Storiche-Ultracentenarie" : "Comune",
    "2018-Allievi-Ritirati" : "ComuneResidenza",
    "2016-Allievi-Ritirati" : "ComuneResidenza",
    "2015-Allievi-Ritirati" : "ComuneResidenza",
    "2014-Allievi-Ritirati" : "ComuneResidenza",
    "2018-Allievi-Partecipanti" : "ComuneResidenza",
    "2015-Allievi-Partecipanti" : "ComuneResidenza",
    "2014-Allievi-Partecipanti" : "ComuneResidenza",
    "Alberi-Monumentali-Della-Campania" : "PROVINCIA",
    "Registro-Fattorie-Sociali" : "Comune",
    "Istituti-Riuniti-Assistenza-e-Beneficienza" : "Comune",
    "Aziende-Pubbliche-Servizi-Alla-Persona" : "Sede",
    "Locazioni-passive" : "Comune",
    "2018-Enti-Accreditati" : "ComuneEnte",
    "Aziende-TPL" : "Comune",
    "centralinemonitoraggioqualitaaria" : "COMUNE",
    #"2016-Allievi-Partecipanti" : "ComuneResidenza", #PROBLEMATICO, non si arresta
}



RESULT_PATH = ""
    
def test():
    
    for file_name in test_map:
        RESULT_PATH = "results" 
        if not os.path.exists(RESULT_PATH):
            os.mkdir(RESULT_PATH)

        RESULT_PATH += "/" + file_name
        if not os.path.exists(RESULT_PATH):
            start = t.time()

            os.mkdir(RESULT_PATH)

            column_name = test_map[file_name]
            words = manage_data.load_csv(csv_name="datasets/"+file_name+".csv", column=column_name, encoding="UTF-8")

            test_corpus(RESULT_PATH, file_name, words, start)

        else:
            print(file_name + " already exists")

test_specifici_map = {
    "Elenco_enti_Spettacolo_e_Cinema": "SedeLegale",
    "Medici-Medicina-Generale" : "ComuneStudioMedico_Struttura",
}

def test_specifici():
    for file_name in test_specifici_map:
        RESULT_PATH = "results_Levensthein" 
        if not os.path.exists(RESULT_PATH):
            os.mkdir(RESULT_PATH)

        RESULT_PATH += "/" + file_name
        if not os.path.exists(RESULT_PATH):
            start = t.time()

            os.mkdir(RESULT_PATH)

            column_name = test_specifici_map[file_name]
            words = manage_data.load_csv(csv_name="datasets/"+file_name+".csv", column=column_name, encoding="UTF-8")

            for i, w in enumerate(words):
                words[i] = manage_data.rimuovi_provincia(w)

            test_corpus(RESULT_PATH, file_name, words, start)

        else:
            print(file_name + " already exists")

def test_corpus(RESULT_PATH, file_name, words, start):
    times = {}

    wrong_words = string_similarity.get_wrong_words(words, dictionary_municipalities)
    np.savetxt(RESULT_PATH+'/wrong_words.csv', [p for p in wrong_words], delimiter=',', fmt='%s')

    matrix, time_similarity_computation = string_similarity.wombo_combo(words, dictionary_municipalities)
    min_cluster, max_cluster = string_similarity.cluster_range(words, dictionary_municipalities)

    times['similarity_computation'] = datetime.timedelta(seconds=time_similarity_computation)

    current_cluster_num = int((min_cluster + max_cluster)/2)
    model, clusters, clustering_time = mycluster.agglomerative_propagation(matrix, current_cluster_num, words)
    change_cluster = mycluster.check_clusters(clusters, dictionary_municipalities)

    run = 0

    while change_cluster != 0 and current_cluster_num>=min_cluster and current_cluster_num<=max_cluster:
        if current_cluster_num==min_cluster and current_cluster_num<0:
            break
        if current_cluster_num==max_cluster and current_cluster_num>0:
            break

        current_cluster_num = current_cluster_num + change_cluster
        model, clusters, clustering_time = mycluster.agglomerative_propagation(matrix, current_cluster_num, words)
        change_cluster = mycluster.check_clusters(clusters, dictionary_municipalities)
        times['clustering_'+str(run)] = datetime.timedelta(seconds=clustering_time)
        run+=1

    corrected_clusters, corrections, correction_time = mycluster.propose_correction(clusters, dictionary_municipalities)
    times['correction'] = datetime.timedelta(seconds=correction_time)

    #print(corrections)
    np.savetxt(RESULT_PATH+'/corrections.csv', [p for p in zip(corrections.keys(),corrections.values())], delimiter=',', fmt='%s')

    #for word in words:
    #    if word in corrections:
    #        print(word + "," + corrections[word])

    end = t.time()
    total_execution_time = end - start
    print("Tempo totale " + file_name + ": ", total_execution_time)

    times['total_time'] = datetime.timedelta(seconds=total_execution_time)

    np.savetxt(RESULT_PATH+'/execution_time.csv', [p for p in zip(times.keys(),times.values())], delimiter=',', fmt='%s')
 

def test_Levensthein():
    for file_name in test_map:
        RESULT_PATH = "results_Levensthein" 
        if not os.path.exists(RESULT_PATH):
            os.mkdir(RESULT_PATH)

        RESULT_PATH += "/" + file_name
        if not os.path.exists(RESULT_PATH):
            start = t.time()

            os.mkdir(RESULT_PATH)

            column_name = test_map[file_name]
            words = manage_data.load_csv(csv_name="datasets/"+file_name+".csv", column=column_name, encoding="UTF-8")

            #for i, w in enumerate(words):
            #    words[i] = manage_data.rimuovi_provincia(w)

            times = {}

            wrong_words = string_similarity.get_wrong_words(words, dictionary_municipalities)
            np.savetxt(RESULT_PATH+'/wrong_words.csv', [p for p in wrong_words], delimiter=',', fmt='%s')

            matrix, time_similarity_computation = string_similarity.lev_distance_word_dictionary(wrong_words, array_municipalities)

            corrections = {}
            start_correction = t.time()
            for i in range(len(matrix)):
                column = matrix[i]
                similar_word = array_municipalities[np.argmin(column)]
                corrections[wrong_words[i]] = similar_word

            correction_time = t.time() - start_correction
            times['correction'] = datetime.timedelta(seconds=correction_time)

            #print(corrections)
            np.savetxt(RESULT_PATH+'/corrections.csv', [p for p in zip(corrections.keys(),corrections.values())], delimiter=',', fmt='%s')

            #for word in words:
            #    if word in corrections:
            #        print(word + "," + corrections[word])

            end = t.time()
            total_execution_time = end - start
            print("Tempo totale " + file_name + ": ", total_execution_time)

            times['total_time'] = datetime.timedelta(seconds=total_execution_time)

            np.savetxt(RESULT_PATH+'/execution_time.csv', [p for p in zip(times.keys(),times.values())], delimiter=',', fmt='%s')

        else:
            print(file_name + " already exists")


def test_Wombo_Combo():
    for file_name in test_map:
        RESULT_PATH = "results_L_FM" 
        if not os.path.exists(RESULT_PATH):
            os.mkdir(RESULT_PATH)

        RESULT_PATH += "/" + file_name
        if not os.path.exists(RESULT_PATH):
            start = t.time()

            os.mkdir(RESULT_PATH)

            column_name = test_map[file_name]
            words = manage_data.load_csv(csv_name="datasets/"+file_name+".csv", column=column_name, encoding="UTF-8")

            for i, w in enumerate(words):
                words[i] = manage_data.rimuovi_provincia(w)

            times = {}

            wrong_words = string_similarity.get_wrong_words(words, dictionary_municipalities)
            np.savetxt(RESULT_PATH+'/wrong_words.csv', [p for p in wrong_words], delimiter=',', fmt='%s')

            matrix, time_similarity_computation = string_similarity.wombo_combo_word_dictionary(wrong_words, array_municipalities)

            corrections = {}
            start_correction = t.time()
            for i in range(len(matrix)):
                column = matrix[i]
                similar_word = array_municipalities[np.argmin(column)]
                corrections[wrong_words[i]] = similar_word

            correction_time = t.time() - start_correction
            times['correction'] = datetime.timedelta(seconds=correction_time)

            #print(corrections)
            np.savetxt(RESULT_PATH+'/corrections.csv', [p for p in zip(corrections.keys(),corrections.values())], delimiter=',', fmt='%s')

            #for word in words:
            #    if word in corrections:
            #        print(word + "," + corrections[word])

            end = t.time()
            total_execution_time = end - start
            print("Tempo totale " + file_name + ": ", total_execution_time)

            times['total_time'] = datetime.timedelta(seconds=total_execution_time)

            np.savetxt(RESULT_PATH+'/execution_time.csv', [p for p in zip(times.keys(),times.values())], delimiter=',', fmt='%s')

        else:
            print(file_name + " already exists")

test_map={
    "2017-Allievi-Ritirati" : "ComuneResidenza",
    
}


def test_without_dictionary():
    
    for file_name in test_map:
        RESULT_PATH = "results_clustering_L_FM" 
        if not os.path.exists(RESULT_PATH):
            os.mkdir(RESULT_PATH)

        RESULT_PATH += "/" + file_name
        if not os.path.exists(RESULT_PATH):
            start = t.time()

            os.mkdir(RESULT_PATH)

            column_name = test_map[file_name]
            words = manage_data.load_csv(csv_name="datasets/"+file_name+".csv", column=column_name, encoding="UTF-8")

            test_corpus_without_dictionary(RESULT_PATH, file_name, words, start)

        else:
            print(file_name + " already exists")

def test_corpus_without_dictionary(RESULT_PATH, file_name, words, start):
    times = {}

    #wrong_words = string_similarity.get_wrong_words(words, dictionary_municipalities)
    #np.savetxt(RESULT_PATH+'/wrong_words.csv', [p for p in wrong_words], delimiter=',', fmt='%s')

    matrix, time_similarity_computation = string_similarity.wombo_combo_matrix(words, dictionary_municipalities)

    #min_cluster, max_cluster = string_similarity.cluster_range(words, dictionary_municipalities)
    max_cluster = string_similarity.max_clusters(words)

    times['similarity_computation'] = datetime.timedelta(seconds=time_similarity_computation)

    clusters_array = []
    clusters_silhoutte = []


    for i in range(5, max_cluster):
        model, clusters, clustering_time = mycluster.agglomerative_propagation(matrix, i, words)
        cluster_labels = model.fit_predict(matrix)
        print(cluster_labels)
        clusters_array.append(clusters)

        silhoutte_result = silhouette_score(matrix, cluster_labels, metric="precomputed")
        clusters_silhoutte.append(silhoutte_result)

        times['cluster_'+str(i)] = datetime.timedelta(seconds=clustering_time)

    best_clusters = clusters_array[np.argmax(clusters_silhoutte)]

    np.savetxt(RESULT_PATH+'/best_clusters.csv', best_clusters, delimiter=',', fmt='%s')

    end = t.time()
    total_execution_time = end - start
    print("Tempo totale " + file_name + ": ", total_execution_time)

    times['total_time'] = datetime.timedelta(seconds=total_execution_time)

    np.savetxt(RESULT_PATH+'/execution_time.csv', [p for p in zip(times.keys(),times.values())], delimiter=',', fmt='%s')

def silhouette_agglomerative():
    inf = 1
    sup = 4
    words = manage_data.load_csv(csv_name="datasets/2017-Allievi-Ritirati.csv", column="ComuneResidenza", encoding="UTF-8")
    X, time_similarity_computation = string_similarity.matrix_lev(words)
    sup += 1
    n_best = []
    scores = []
    for n_clusters in range(inf, sup):
        # Create a subplot with 1 row and 2 columns
        fig, (ax1) = plt.subplots(1)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = AC(affinity="precomputed", n_clusters=n_clusters, linkage="complete")
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels, metric="precomputed")
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        scores.append(silhouette_avg)


        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels, metric="precomputed")

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            plt.suptitle(("Silhouette Agglomerative "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')
    avg = statistics.mean(scores)
    for i, s in enumerate(scores):
        if s >= 0.90 or s >= avg:
            n_best.append(i+inf)
    plt.show()
    return n_best


def proviamo():

    words = manage_data.load_csv(csv_name="datasets/2017-Allievi-Ritirati.csv", column="ComuneResidenza", encoding="UTF-8")
    matrix, time_similarity_computation = string_similarity.wombo_combo_matrix(words, dictionary_municipalities)
    
    model = AC(affinity="precomputed", n_clusters=4, linkage="complete")
    cluster_labels = model.fit_predict(matrix)
    print(cluster_labels)

    silhoutte_result = silhouette_score(matrix, cluster_labels, metric="precomputed")

    print(silhoutte_result)

silhouette_agglomerative()


#test()
#test_specifici()
#test_Levensthein()
#test_Wombo_Combo()