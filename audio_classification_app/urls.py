from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('recording', views.recording, name='recording'),
    path('processing_request', views.processing_request, name='processing_request'),
    path('processing', views.processing, name='processing_request'),

]
