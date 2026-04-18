from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import CaseStudyViewSet, CsvFileViewSet


router = DefaultRouter()
router.register(r'case-study', CaseStudyViewSet, basename='case-study')
router.register(r'csv-files', CsvFileViewSet, basename='csv-files')
# router.register(r'events', EventsViewSet, basename='events')

urlpatterns = [
    path('', include(router.urls)),
]

