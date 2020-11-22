from django.db import models


class AudioClip(models.Model):
    filename = models.CharField(max_length=200)
    chroma_stft = models.DecimalField(max_digits=25, decimal_places=20)
    rmse = models.DecimalField(max_digits=25, decimal_places=20)
    spectral_centroid = models.DecimalField(max_digits=25, decimal_places=20)
    spectral_bandwidth = models.DecimalField(max_digits=25, decimal_places=20)
    rolloff = models.DecimalField(max_digits=25, decimal_places=20)
    zero_crossing_rate = models.DecimalField(max_digits=25, decimal_places=20)
    tempo = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc1 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc2 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc3 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc4 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc5 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc6 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc7 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc8 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc9 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc10 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc11 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc12 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc13 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc14 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc15 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc16 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc17 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc18 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc19 = models.DecimalField(max_digits=25, decimal_places=20)
    mfcc20 = models.DecimalField(max_digits=25, decimal_places=20)
    genre = models.CharField(max_length=200)

    def __str__(self):
        return self.genre
