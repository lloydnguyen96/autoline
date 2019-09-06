import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


# Constants
n_features = 2 # number of features = dimension of feature space
n_clusters = 5 # number of clusters
n_samples_per_cluster = 500 # number of samples of each cluster
seed = 700
embiggen_factor = 70 # specify the 2D region from which we choose centroid's coordinates, this region is [-(embiggen_factor/2), embiggen_factor/2)
n_steps = 50


def create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed):
	np.random.seed(seed)
	slices = []
	centroids = []
	# create samples for each cluster
	for i in range(n_clusters):
		samples = tf.random_normal((n_samples_per_cluster, n_features),
			mean = 0.0, stddev = 5.0, dtype = tf.float32, seed = seed, name="cluster_{}".format(i))
		current_centroid = (np.random.random((1, n_features)) * embiggen_factor) - (embiggen_factor/2)
		centroids.append(current_centroid)
		samples += current_centroid
		slices.append(samples)
	# create a big "samples" dataset
	samples = (np.random.random((500, 2)) * embiggen_factor * 2.5/2) - (embiggen_factor * 2.5/4)
	slices.append(samples)
	samples = tf.concat(slices, 0, name='samples') # convert list type (list of 3 2D tensorflow matrices) to tensorflow type (one 2D matrix)
	centroids = tf.concat(centroids, 0, name='centroids') # convert list type (list of 3 2D matrices) to tensorflow type (one 2D matrix)
	return centroids, samples


def choose_random_centroids(samples, n_clusters):
	# Step: Initialisation: Select `n_clusters` number of random points
	n_samples = tf.shape(samples)[0]
	random_indices = tf.random_shuffle(tf.range(0, n_samples))
	begin = [0,]
	size = [n_clusters,]
	centroid_indices = tf.slice(random_indices, begin, size)
	initial_centroids = tf.gather(samples, centroid_indices) # coordinates of each centroid
	return initial_centroids


def assign_to_nearest(samples, centroids, n_clusters):
	# finds the nearest centroid for each sample
	# start from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
	expanded_vectors = tf.expand_dims(samples, 0)
	expanded_centroids = tf.expand_dims(centroids, 1)
	a = tf.subtract(expanded_vectors, expanded_centroids) # (x1 - x2, y1 - y2) of each couple (sample, centroid)
	b = tf.square(a) # ((x1 - x2)^2, (y1 - y2)^2) of each couple (sample, centroid)
	distances = tf.reduce_sum(b, 2) # distance = (x1 - x2)^2 + (y1 - y2)^2 of each couple (sample, centroid)
	nearest_indices = tf.argmin(distances, 0) # each sample has an index indicating a centroid's index that is nearest to it
	# end from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
	nearest_indices = tf.to_int32(nearest_indices)
	partitions = tf.dynamic_partition(samples, nearest_indices, n_clusters) # partitions = list of tensors, each tensor is a subset of samples (parameter)
	return partitions


def update_centroids(partitions):
	# updates the centroid to be the mean of all samples associated with it.
	new_centroids_list = []
	for partition in partitions:
		a = tf.reduce_mean(partition, 0) # a = (2,) contains coordinates of new centroids
		b = tf.expand_dims(a, 0) # b = modified a with appropriate dimension, convert (2,) -> (1, 2)
		new_centroids_list.append(b)
	new_centroids = tf.concat(new_centroids_list, 0)
	return new_centroids


def plot_clusters(centroids, partitions):
	# plot out the different clusters
	# choose a different colour for each cluster
	colour = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))
	for i, centroid in enumerate(centroids):
		# grab just the samples fpr the given cluster and plot them out with a new colour
		# also plot centroid
		plt.scatter(partitions[i][:,0], partitions[i][:,1], c = colour[i])
		plt.plot(centroid[0], centroid[1], markersize=8, marker="x", color='k', mew=6)
		plt.plot(centroid[0], centroid[1], markersize=4, marker="x", color='m', mew=4)


def main():
	data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
	initial_centroids = choose_random_centroids(samples, n_clusters)
	updated_centroids = initial_centroids
	for step in range(n_steps):
		partitions = assign_to_nearest(samples, updated_centroids, n_clusters)
		updated_centroids = update_centroids(partitions)
	partitions = assign_to_nearest(samples, updated_centroids, n_clusters)

	#model = tf.global_variables_initializer() # model = an Op that initializes global variables in the graph
	with tf.Session() as session:
		partitions_value = session.run(partitions)
		updated_centroid_value = session.run(updated_centroids)
		session.close()

	plt.figure(1)
	print(partitions_value[0])
	print(partitions_value[1])
	print(partitions_value[4])
	print(type(partitions_value))
	print(partitions)
	plot_clusters(updated_centroid_value, partitions_value)
	plot_boundaries(partitions_value)
	plt.show()
	return
# end main


def create_clusters(n_clusters, n_samples_per_cluster, embiggen_factor, n_steps):
	data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
	initial_centroids = choose_random_centroids(samples, n_clusters)
	updated_centroids = initial_centroids
	for step in range(n_steps):
		partitions = assign_to_nearest(samples, updated_centroids, n_clusters)
		updated_centroids = update_centroids(partitions)
	partitions = assign_to_nearest(samples, updated_centroids, n_clusters)

	#model = tf.global_variables_initializer() # model = an Op that initializes global variables in the graph
	with tf.Session() as session:
		partitions_value = session.run(partitions)
		updated_centroid_value = session.run(updated_centroids)
		session.close()
	return partitions_value
# end create_clusters


def plot_boundaries(partitions):
	for partition in partitions:
		edges = alpha_shape(partition, 500, True)
		for i, j in edges:
			plt.plot(partition[[i,j],0], partition[[i,j],1], 'k--')


def boundary_point(points, alpha):
	edges = alpha_shape(points, alpha, only_outer = True)
	boundaryPointIndexSetOne = {str(edge[0]) for edge in edges}
	boundaryPointIndexSetTwo = {str(edge[1]) for edge in edges}
	boundaryPointIndexSet = boundaryPointIndexSetOne.union(boundaryPointIndexSetTwo)
	boundaryPoint = [points[int(index)].tolist() for index in boundaryPointIndexSet]
	boundaryPointDict = dict(zip(boundaryPointIndexSet, boundaryPoint))
	return boundaryPointDict
# end boundary_point


def alpha_shape(points, alpha, only_outer=True):
	"""
	Compute the alpha shape (concave hull) of a set of points.
	:param points: np.array of shape (n,2) points.
	:param alpha: alpha value.
	:param only_outer: boolean value to specify if we keep only the outer border
	or also inner edges.
	:return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
	the indices in the points array.
	"""
	assert points.shape[0] > 3, "Need at least four points"

	def add_edge(edges, i, j):
		"""
		Add an undirected edge between the i-th and j-th points,
		if not in the list already
		"""
		if (i, j) in edges or (j, i) in edges:
			# already added
			assert (j, i) in edges, "Can't go twice over same directed edge right?"
			if only_outer:
				# if both neighboring triangles are in shape, it's not a boundary edge
				edges.remove((j, i))
			return
		edges.add((i, j))

	tri = Delaunay(points)
	edges = set()
	# Loop over triangles:
	# ia, ib, ic = indices of corner points of the triangle
	for ia, ib, ic in tri.vertices:
		pa = points[ia]
		pb = points[ib]
		pc = points[ic]
		# Computing radius of triangle circumcircle
		# www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
		a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
		b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
		c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
		s = (a + b + c) / 2.0
		area = np.sqrt(s * (s - a) * (s - b) * (s - c))
		circum_r = a * b * c / (4.0 * area)
		if circum_r < alpha:
			add_edge(edges, ia, ib)
			add_edge(edges, ib, ic)
			add_edge(edges, ic, ia)
	return edges
# end alpha_shape


if __name__ == "__main__":
	main()
# end if


'''
def main():
	data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
	initial_centroids = choose_random_centroids(samples, n_clusters)
	updated_centroids = initial_centroids
	for step in range(n_steps):
		nearest_indices = assign_to_nearest(samples, updated_centroids)
		updated_centroids = update_centroids(samples, nearest_indices, n_clusters)

	model = tf.global_variables_initializer() # model = an Op that initializes global variables in the graph
	with tf.Session() as session:
		updated_centroid_value = session.run(updated_centroids)
		all_samples = session.run(samples)

	plot_clusters(all_samples, updated_centroid_value, n_samples_per_cluster)
# end main




def assign_to_nearest(samples, centroids):
	# Finds the nearest centroid for each sample

	# START from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
	expanded_vectors = tf.expand_dims(samples, 0)
	expanded_centroids = tf.expand_dims(centroids, 1)
	distances = tf.reduce_sum( tf.square(
	           tf.subtract(expanded_vectors, expanded_centroids)), 2)
	mins = tf.argmin(distances, 0)
	# END from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
	nearest_indices = mins
	return nearest_indices


def update_centroids(samples, nearest_indices, n_clusters):
	# Updates the centroid to be the mean of all samples associated with it.
	nearest_indices = tf.to_int32(nearest_indices)
	partitions = tf.dynamic_partition(samples, nearest_indices, n_clusters)
	new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
	return new_centroids


def plot_clusters(all_samples, centroids, n_samples_per_cluster):
	import matplotlib.pyplot as plt
	#Plot out the different clusters
	#Choose a different colour for each cluster
	colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
	for i, centroid in enumerate(centroids):
		#Grab just the samples fpr the given cluster and plot them out with a new colour
		samples = all_samples[i*n_samples_per_cluster:(i+1)*n_samples_per_cluster]
		plt.scatter(samples[:,0], samples[:,1], c=colour[i])
		#Also plot centroid
		plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
		plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
	plt.show()
'''