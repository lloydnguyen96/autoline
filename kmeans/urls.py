from django.urls import path, include
from rest_framework.routers import DefaultRouter
from kmeans.views import UserViewSet, KMeansViewSet, ClusterViewSet

router = DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'kmeans', KMeansViewSet)
router.register(r'cluster', ClusterViewSet)

# The API URLs are now determined automatically by the router.
urlpatterns = [
	path('', include(router.urls)),
]