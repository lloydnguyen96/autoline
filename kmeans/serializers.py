from rest_framework import serializers
from kmeans.models import KMeans, Cluster
from django.contrib.auth.models import User

class UserSerializer(serializers.HyperlinkedModelSerializer):
	kmeans = serializers.HyperlinkedRelatedField(many=True, view_name='kmeans-detail', read_only=True)

	class Meta:
		model = User
		fields = ('url', 'id', 'username', 'kmeans')

class KMeansSerializer(serializers.HyperlinkedModelSerializer):
	owner = serializers.ReadOnlyField(source='owner.username')
	clusters = serializers.HyperlinkedRelatedField(many=True, view_name='cluster-detail', read_only=True)

	class Meta:
		model = KMeans
		fields = ('url', 'id', 'text', 'n_clusters', 'n_samples_per_cluster', 
					'embiggen_factor', 'n_steps', 'date_added', 'owner', 'clusters')

class ClusterSerializer(serializers.HyperlinkedModelSerializer):
	owner = serializers.ReadOnlyField(source='owner.username')

	class Meta:
		model = Cluster
		fields = ('url', 'id', 'kMeans', 'boundaryPoints', 'owner')