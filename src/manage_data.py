import pandas as pd
import tabulate
import numpy as np

DEFAULT_ENCODING = "windows-1252"

def show_clusters(cluster):
    cluster_set = {}
    for cl in cluster:

        if len(cl)==0:
            continue
        cl = np.unique(cl)
        # data[cl[0]] = pd.Series(cl[1:])
        cluster_set[cl[0]] = cl
    data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in cluster_set.items()]))
    data = data.replace(np.nan, '', regex=True)

    print("\ndimensioni: ", data.shape, "\n\n", tabulate.tabulate(data, headers="keys", tablefmt="orgtbl"))
    return data.shape

def load_matrix(filename):
    return np.load(filename)


def save_matrix(filename, matrix):
    np.save(filename, matrix)

def remove_province(province):
    i = province.find("(")
    if i != -1:
        province = province[0:i:]
    return province


def load_csv(csv_name, column, nrows=0, encoding=DEFAULT_ENCODING):

    if nrows == 0:
        data = pd.read_csv(csv_name, error_bad_lines=False, sep=";", encoding=encoding)
    else:
        data = pd.read_csv(csv_name, error_bad_lines=False, sep=";", nrows=nrows, encoding=encoding)

    words = data[column].to_numpy()

    for i, s in enumerate(words):
        if isinstance(s, float):
            words[i] = "Non specificato"
    return words


def load_regions(path="../dictionaries/elenco province.csv"):
    data = pd.read_csv(path, error_bad_lines=False, sep=";", encoding="ISO-8859-1")
    regions = np.unique(data["Regione"].to_numpy())
    key = np.array([x.lower() if isinstance(x, str) else x for x in regions])
    value =  np.array(regions)
    return dict(zip(key, value))


def load_province(path="../dictionaries/elenco province.csv"):
    data = pd.read_csv(path, error_bad_lines=False, sep=";", encoding="ISO-8859-1")
    province = np.unique(data["Provincia"].to_numpy())
    key = np.array([x.lower() if isinstance(x, str) else x for x in province])
    value = np.array(province)
    return dict(zip(key, value))


def load_municipalities(path="../dictionaries/italian_municipalities.csv"):
    data = pd.read_csv(path, error_bad_lines=False, sep=";", encoding="ISO-8859-1")
    comuni = np.unique(data["Comune"].to_numpy())
    key = np.array([x.lower() if isinstance(x, str) else x for x in comuni])
    value = np.array(comuni)
    return dict(zip(key, value))

def load_municipalities_as_array(path="../dictionaries/italian_municipalities.csv"):
    data = pd.read_csv(path, error_bad_lines=False, sep=";", encoding="ISO-8859-1")
    comuni = np.unique(data["Comune"].to_numpy())
    values = np.array(comuni)
    return values
