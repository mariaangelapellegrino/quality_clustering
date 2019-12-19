from src import manage_data, mycluster, string_similarity
import time as t
import numpy as np
import pandas as pd
import os
import datetime

test_map = {
	"contributi-francia-2017" : "inom",
	"treni francia" : "COMMUNE_COMPTEUR"
}

def test():
	data = pd.read_csv("dictionaries/french_municipalities.csv", sep=";", encoding="UTF-8")
	french_municipalities_array = np.unique(data["Municipalities"].to_numpy())
	french_municipalities_lowercase = np.array([x.lower() if isinstance(x, str) else x for x in french_municipalities_array])
	french_municipalities_dict = dict(zip(french_municipalities_lowercase, french_municipalities_array))

	for file_name in test_map:
		RESULT_PATH = "french_results" 
		if not os.path.exists(RESULT_PATH):
			os.mkdir(RESULT_PATH)

		RESULT_PATH += "/" + file_name
		if not os.path.exists(RESULT_PATH):
			times = {}
			start = t.time()

			os.mkdir(RESULT_PATH)

			column_name = test_map[file_name]
			words = manage_data.load_csv(csv_name="datasets/"+file_name+".csv", column=column_name, encoding="UTF-8", nrows=1000)

			wrong_words = string_similarity.get_wrong_words(words, french_municipalities_dict)
			np.savetxt(RESULT_PATH+'/wrong_words.csv', [p for p in wrong_words], delimiter=',', fmt='%s')

			matrix, time_similarity_computation = string_similarity.wombo_combo(words, french_municipalities_dict)
			min_cluster, max_cluster = string_similarity.cluster_range(words, french_municipalities_dict)

			times['similarity_computation'] = datetime.timedelta(seconds=time_similarity_computation)

			current_cluster_num = int((min_cluster + max_cluster)/2)
			model, clusters, clustering_time = mycluster.agglomerative_propagation(matrix, current_cluster_num, words)
			change_cluster = mycluster.check_clusters(clusters, french_municipalities_dict)

			run = 0

			while change_cluster != 0 and current_cluster_num>=min_cluster and current_cluster_num<=max_cluster:
				current_cluster_num = current_cluster_num + change_cluster
				model, clusters, clustering_time = mycluster.agglomerative_propagation(matrix, current_cluster_num, words)
				change_cluster = mycluster.check_clusters(clusters, french_municipalities_dict)
				times['clustering_'+str(run)] = datetime.timedelta(seconds=clustering_time)
				run+=1

			corrected_clusters, corrections, correction_time = mycluster.propose_correction(clusters, french_municipalities_dict)
			times['correction'] = datetime.timedelta(seconds=correction_time)

			np.savetxt(RESULT_PATH+'/corrections.csv', [p for p in zip(corrections.keys(),corrections.values())], delimiter=',', fmt='%s')

			end = t.time()
			total_execution_time = end - start
			print("Tempo totale " + file_name + ": ", total_execution_time)

			times['total_time'] = datetime.timedelta(seconds=total_execution_time)

			np.savetxt(RESULT_PATH+'/execution_time.csv', [p for p in zip(times.keys(),times.values())], delimiter=',', fmt='%s')

		else:
			print(file_name + " already exists")

test()