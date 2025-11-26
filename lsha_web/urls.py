from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('core_algorithm.urls')),
    path('api/', include('rest_api.urls')),
    # Include your app's URLs
]