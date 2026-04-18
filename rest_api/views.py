from rest_framework import viewsets
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from .models import CaseStudy, CsvFile
from .serializers import CaseStudySerializer, CsvFileSerializer


class CaseStudyViewSet(viewsets.ModelViewSet):
    queryset = CaseStudy.objects.all()
    serializer_class = CaseStudySerializer
    permission_classes = [AllowAny]

class CsvFileViewSet(viewsets.ModelViewSet):
    queryset = CsvFile.objects.all()
    serializer_class = CsvFileSerializer
    permission_classes = [AllowAny]