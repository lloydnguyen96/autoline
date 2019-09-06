from django.db import models
from . import kMeans
from django.contrib.postgres.fields import JSONField

# Create your models here.
class KMeans(models.Model):
	text = models.TextField()
	n_clusters = models.IntegerField()
	n_samples_per_cluster = models.IntegerField()
	embiggen_factor = models.IntegerField()
	n_steps = models.IntegerField()
	date_added = models.DateTimeField(auto_now_add=True)
	owner = models.ForeignKey('auth.User', related_name='kmeans', on_delete=models.CASCADE)

	class Meta:
		ordering = ('date_added',)

	def save(self, *args, **kwargs):
		super(KMeans, self).save(*args, **kwargs)
		self.create_clusters()

	def create_clusters(self):
		partitions = kMeans.create_clusters(self.n_clusters, self.n_samples_per_cluster, self.embiggen_factor, self.n_steps)
		for i in range(self.n_clusters):
			boundaryPoints = kMeans.boundary_point(partitions[i], 500)
			cluster = Cluster(kMeans = self, boundaryPoints = boundaryPoints, owner = self.owner)
			cluster.save()

class Cluster(models.Model):
	kMeans = models.ForeignKey('KMeans', related_name='clusters', on_delete=models.CASCADE)
	boundaryPoints = JSONField()
	owner = models.ForeignKey('auth.User', related_name='clusters', on_delete=models.CASCADE)

	def save(self, *args, **kwargs):
		super(Cluster, self).save(*args, **kwargs)