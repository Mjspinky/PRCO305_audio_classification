from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render, redirect
from utils.recorded_data_processing import recorded_data_preparation
from utils.audio_recording import audio_record
from .models import AudioClip


def index(request):
    return render(request, 'index.html')


def results(request, filename):
    response = "The dance style most preferred to this style of music is %s."
    genre = get_object_or_404(AudioClip, pk=1)
    return HttpResponse(response % genre)


def recording(request):
    audio_record()
    return redirect("processing_request")


def processing_request(request):
    return render(request, 'process.html')


def processing(request):
    prediction = recorded_data_preparation()
    context = {'prediction': prediction}
    return render(request, 'result.html', context)

# def test_data(filename):
