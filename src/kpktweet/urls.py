from django.urls import path
from django.contrib.auth.decorators import login_required
from django.views.generic.edit import DeleteView

from kpktweet.views import (TweetListView, InsertNewView, GraphView, ClassiView)

app_name = 'kpk'


urlpatterns = [
    path('',                TweetListView.as_view(),        name='index'),
    path('tabel',           TweetListView.as_view(),        name='index'),
    path('insert',          InsertNewView.as_view(),        name='post'),
    path('grafik',          GraphView.as_view(),            name='graph'),
    path('klasifikasi',     ClassiView.as_view(),           name='classi') 
]