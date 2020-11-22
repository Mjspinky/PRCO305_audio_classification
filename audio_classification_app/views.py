from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render
from data_training import test_data_preparation
from audio_recording import audio_record
from .models import AudioClip


def index(request):
    return render(request, 'index.html')


def results(request, filename):
    response = "The dance style most preferred to this style of music is %s."
    genre = get_object_or_404(AudioClip, pk=1)
    return HttpResponse(response % genre)


def recording(request):
    audio_record()
    test_data_preparation()
    # context = test_data(request.POST['filename'])
    response = "Please wait for processing"
    return HttpResponse(response)

# def test_data(filename):
