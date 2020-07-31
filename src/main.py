import manage_data, mycluster, string_similarity
import time as t
import numpy as np
import os
import datetime
from sklearn.cluster import AgglomerativeClustering as AC

dictionary_municipalities = manage_data.load_municipalities()
array_municipalities = manage_data.load_municipalities_as_array()

test_map = {
	"2017-Allievi-Ritirati" : "ComuneResidenza",
    "Agriturismi-Napoli" : "COMUNE AZIENDA",
    "Albo-Regionale-Cooperative-Sociali" : "ComuneSedeLegale",
    "Associazioni-Giovanili" : "Citta",
    "Elenco-Comunita-Minori" : "Comune",
    "Elenco-Parafarmacie" : "DESCRIZIONECOMUNE",
    "Fattorie-didattiche" : "Sede operativa",
    "Immobili-demaniali" : "Comune",
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
}


RESULT_PATH = ""
    
def test_clustering():
    
    for file_name in test_map:
        RESULT_PATH = "../results_clustering_LevenstheinAndFuzzyMatching" 
        if not os.path.exists(RESULT_PATH):
            os.mkdir(RESULT_PATH)

        RESULT_PATH += "/" + file_name
        if not os.path.exists(RESULT_PATH):
            start = t.time()

            os.mkdir(RESULT_PATH)

            column_name = test_map[file_name]
            words = manage_data.load_csv(csv_name="../datasets/"+file_name+".csv", column=column_name, encoding="UTF-8")

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

            np.savetxt(RESULT_PATH+'/corrections.csv', [p for p in zip(corrections.keys(),corrections.values())], delimiter=',', fmt='%s')

            end = t.time()
            total_execution_time = end - start
            print("Total time " + file_name + ": ", total_execution_time)

            times['total_time'] = datetime.timedelta(seconds=total_execution_time)

            np.savetxt(RESULT_PATH+'/execution_time.csv', [p for p in zip(times.keys(),times.values())], delimiter=',', fmt='%s')

        else:
            print(file_name + " already exists")


def test_clustering_Levensthein():
    
    for file_name in test_map:
        RESULT_PATH = "../results_clustering_Levensthein" 
        if not os.path.exists(RESULT_PATH):
            os.mkdir(RESULT_PATH)

        RESULT_PATH += "/" + file_name
        if not os.path.exists(RESULT_PATH):
            start = t.time()

            os.mkdir(RESULT_PATH)

            column_name = test_map[file_name]
            words = manage_data.load_csv(csv_name="../datasets/"+file_name+".csv", column=column_name, encoding="UTF-8")

            times = {}

            wrong_words = string_similarity.get_wrong_words(words, dictionary_municipalities)
            np.savetxt(RESULT_PATH+'/wrong_words.csv', [p for p in wrong_words], delimiter=',', fmt='%s')

            matrix, time_similarity_computation = string_similarity.lev_distance_word_dictionary(words, dictionary_municipalities)
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

            np.savetxt(RESULT_PATH+'/corrections.csv', [p for p in zip(corrections.keys(),corrections.values())], delimiter=',', fmt='%s')

            end = t.time()
            total_execution_time = end - start
            print("Total time " + file_name + ": ", total_execution_time)

            times['total_time'] = datetime.timedelta(seconds=total_execution_time)

            np.savetxt(RESULT_PATH+'/execution_time.csv', [p for p in zip(times.keys(),times.values())], delimiter=',', fmt='%s')

        else:
            print(file_name + " already exists")

# dictionary-lookup + levensthein as similarity
def test_Levensthein():
    for file_name in test_map:
        RESULT_PATH = "../results_dictionaryLookup_Levensthein" 
        if not os.path.exists(RESULT_PATH):
            os.mkdir(RESULT_PATH)

        RESULT_PATH += "/" + file_name
        if not os.path.exists(RESULT_PATH):
            start = t.time()

            os.mkdir(RESULT_PATH)

            column_name = test_map[file_name]
            words = manage_data.load_csv(csv_name="../datasets/"+file_name+".csv", column=column_name, encoding="UTF-8")

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

            np.savetxt(RESULT_PATH+'/corrections.csv', [p for p in zip(corrections.keys(),corrections.values())], delimiter=',', fmt='%s')

            end = t.time()
            total_execution_time = end - start
            print("Total time " + file_name + ": ", total_execution_time)

            times['total_time'] = datetime.timedelta(seconds=total_execution_time)

            np.savetxt(RESULT_PATH+'/execution_time.csv', [p for p in zip(times.keys(),times.values())], delimiter=',', fmt='%s')

        else:
            print(file_name + " already exists")

# dictionary-lookup + Levensthin & fuzzy matching as similarity
def test_Wombo_Combo():
    for file_name in test_map:
        RESULT_PATH = "../results_dictionaryLookup_LevenstheinAndFuzzyMatching" 
        if not os.path.exists(RESULT_PATH):
            os.mkdir(RESULT_PATH)

        RESULT_PATH += "/" + file_name
        if not os.path.exists(RESULT_PATH):
            start = t.time()

            os.mkdir(RESULT_PATH)

            column_name = test_map[file_name]
            words = manage_data.load_csv(csv_name="../datasets/"+file_name+".csv", column=column_name, encoding="UTF-8")

            for i, w in enumerate(words):
                words[i] = manage_data.remove_province(w)

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

            np.savetxt(RESULT_PATH+'/corrections.csv', [p for p in zip(corrections.keys(),corrections.values())], delimiter=',', fmt='%s')

            end = t.time()
            total_execution_time = end - start
            print("Total time " + file_name + ": ", total_execution_time)

            times['total_time'] = datetime.timedelta(seconds=total_execution_time)

            np.savetxt(RESULT_PATH+'/execution_time.csv', [p for p in zip(times.keys(),times.values())], delimiter=',', fmt='%s')

        else:
            print(file_name + " already exists")

def run_tests():
    test_clustering()
    #test_clustering_Levensthein()
    #test_Levensthein()
    #test_Wombo_Combo()

run_tests()
