# Data Accuracy Issues In Textual Geographical Data by a Clustering-based approach

This repository behaves as a support material for the paper entitled "Detecting Data Accuracy Issues in Textual Geographical Data by a Clustering-based Approach" accepted as short paper at the CODS-COMAD 2021 conference.

By focusing on textual geographical data, we aim to detect inaccuracies and propose a correction by a clustering-based approach. Our method is mainly based on a dictionary of correct values, the Agglomerative clustering to group data in clusters, and Levenshtein and Fuzzy string searching for measuring the similarity among words.
We test our approach on real open datasets provided by our regional administration, heterogeneous in the topic, size, and type of errors by showing the positive results of using Levenshtein and Fuzzy Matching and exploiting Clustering methods in detecting and correcting quality issues in textual geographical data. The achieved results are useful both for data producers and consumers in any domain area.

## Repository structure

- **src** contains the main file that runs all the tests, i.e., the clustering-based approach by using Levenshtein and Fuzzy Matching and Levensthein in isolation and by testing both similarity metrics in a dictionary-lookup approach;
- **datasets** contains the evaluated datasets,
- **dictionaties** contains the dictionary of correct municipalities,
- the **results** folder contains the results for each approach and summary tables

## Requirements

The src/main.py file requires:
- Python 3
- pandas
- tabulate
- sklearn
- Fuzzywuzzy
- pyxdameraulevenshtein
- python-levensthein

## License

The project is released under the MIT license.



