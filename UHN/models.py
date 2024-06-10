from django.db import models


class TestFile(models.Model):
  name = models.CharField(max_length=255, primary_key = True)
  prediction_time = models.FloatField()
  dimension = models.CharField(max_length=1)
  uploaded_at = models.DateTimeField()
