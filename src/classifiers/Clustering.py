from nltk.probability import FreqDist
from collections import defaultdict


class Clustering:

    def __init__(self):
        pass

    @staticmethod
    def group_documents_by_clusters(assigned_clusters, documents):
        cluster_group = defaultdict()
        for idx, cls in enumerate(assigned_clusters):
            print(idx)
            print(cls)
            if cls not in cluster_group.keys():
                cluster_group[cls] = []
            cluster_group[cls].append(documents[idx])
        return cluster_group

    @staticmethod
    def top_terms_by_cluster(cluster_group):
        most_common = {}
        for clt in cluster_group.keys():
            fd = FreqDist([item for sublist in cluster_group[clt] for item in sublist])
            most_common[clt] = fd.most_common(10)
        return most_common
