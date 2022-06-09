from django.db import models

# Create your models here.
class KPKTweet(models.Model):
    tanggal         = models.TextField()
    user_name       = models.CharField(max_length=100)
    isi             = models.TextField()
    stop_removal    = models.TextField()
    label           = models.CharField(max_length=10, default="netral")
    polarity        = models.CharField(max_length=5, default="[0]")

class KPKTweetTemp(models.Model):
    tanggal         = models.TextField()
    user_name       = models.CharField(max_length=100)
    isi             = models.TextField()
    stop_removal    = models.TextField()
    label           = models.CharField(max_length=10, default="netral")
    polarity        = models.CharField(max_length=10, default="[0]")