# Hierarchy clustering utils
Agglomerative clustering is very flexible in terms of the clustering granularity. By default, it constructs the entire dendrogram, and you can choose any number of clusters in [1, num_samples]. Sometimes it could be interesting given the clustering result to explore children, parents or sibling of a particular cluster.

**Sometimes we want to check whether it make sense to merge a particular cluster with its sibling in dendrogram or to split a particular cluster into its children.** It makes sense when some meaningfull entities are clustered (e.g. words, movies, music tracks - entities for which we as humans can assess similarity). Given we can assess entities similarity, we can decide to merge or split a cluster.

By default, neither in scipy.cluster.hierarchy nor sklearn.cluster.AgglomerativeClustering have a convenient way to get cluster's children, parents or siblings. In this repo I implemented functions to do this and provided a small demo.