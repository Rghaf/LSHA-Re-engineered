from django.urls import path
from . import views

urlpatterns = [
    path('run/', views.run_algorithm, name='run_algorithm'),
]