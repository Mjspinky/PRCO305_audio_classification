from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('<string:filename>/results/', views.results, name='results'),
]
