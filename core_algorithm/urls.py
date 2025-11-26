from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import trigger_learning_view
# EventListViewSet, EventsViewSet



urlpatterns = [
    path('start/', trigger_learning_view, name='start_learning'),
    # path('api/', include(router.urls)),
]