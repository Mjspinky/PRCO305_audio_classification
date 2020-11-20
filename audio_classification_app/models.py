from django.db import models


class AudioClip(models.Model):
    filename = models.CharField(max_length=200)
    chroma_stft = models.DecimalField(default=0)
    rmse = models.DecimalField(default=0)
    spectral_centroid = models.DecimalField(default=0)
    spectral_bandwidth = models.DecimalField(default=0)
    rolloff = models.DecimalField(default=0)
    zero_crossing_rate = models.DecimalField(default=0)
    tempo = models.DecimalField(default=0)
    mfcc1 = models.DecimalField(default=0)
    mfcc2 = models.DecimalField(default=0)
    mfcc3 = models.DecimalField(default=0)
    mfcc4 = models.DecimalField(default=0)
    mfcc5 = models.DecimalField(default=0)
    mfcc6 = models.DecimalField(default=0)
    mfcc7 = models.DecimalField(default=0)
    mfcc8 = models.DecimalField(default=0)
    mfcc9 = models.DecimalField(default=0)
    mfcc10 = models.DecimalField(default=0)
    mfcc11 = models.DecimalField(default=0)
    mfcc12 = models.DecimalField(default=0)
    mfcc13 = models.DecimalField(default=0)
    mfcc14 = models.DecimalField(default=0)
    mfcc15 = models.DecimalField(default=0)
    mfcc16 = models.DecimalField(default=0)
    mfcc17 = models.DecimalField(default=0)
    mfcc18 = models.DecimalField(default=0)
    mfcc19 = models.DecimalField(default=0)
    mfcc20 = models.DecimalField(default=0)
    genre = models.CharField(max_length=200)

    def __str__(self):
        return self.filename
