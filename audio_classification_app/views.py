from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader


def index(request):
    return render(request, 'index.html')


def results(request, genre):
    response = "The dance style most preferred to this style of music is %s."
    return HttpResponse(response % genre)
