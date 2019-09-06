from django.shortcuts import render
from rest_framework import viewsets
from kmeans.models import KMeans, Cluster
from django.contrib.auth.models import User
from kmeans.serializers import UserSerializer, KMeansSerializer, ClusterSerializer
from rest_framework import permissions
from kmeans.permissions import IsOwnerOrReadOnly

# Create your views here.
class UserViewSet(viewsets.ReadOnlyModelViewSet):
	"""
	This viewset automatically provides `list` and `detail` actions.
	"""
	queryset = User.objects.all()
	serializer_class = UserSerializer

class KMeansViewSet(viewsets.ModelViewSet):
	"""
	This viewset automatically provides 'list', 'create', 'retrieve',
	'update' and 'destroy' actions.

	Additionally we also provide an extra 'highlight' action.
	"""
	queryset = KMeans.objects.all()
	serializer_class = KMeansSerializer
	permission_classes = (permissions.IsAuthenticatedOrReadOnly,
						IsOwnerOrReadOnly,)

	def perform_create(self, serializer):
		serializer.save(owner=self.request.user)

class ClusterViewSet(viewsets.ModelViewSet):
	"""
	This viewset automatically provides 'list', 'create', 'retrieve',
	'update' and 'destroy' actions.

	Additionally we also provide an extra 'highlight' action.
	"""
	queryset = Cluster.objects.all()
	serializer_class = ClusterSerializer
	permission_classes = (permissions.IsAuthenticatedOrReadOnly,
						IsOwnerOrReadOnly,)