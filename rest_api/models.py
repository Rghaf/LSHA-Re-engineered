from django.db import models

# Create your models here.
class CaseStudy(models.Model):
    id = models.AutoField(primary_key=True)
    # User is optional, just for restore case studies in the future.
    user = models.CharField(max_length=100, null=True, blank=True)
    name = models.CharField(max_length=200)

    def __str__(self):
        # If name exists, return it. If not, return a placeholder with the ID.
        if self.name:
            return self.name
        return f"Unnamed List (ID: {self.id})"

class Event(models.Model):
    id = models.AutoField(primary_key=True)
    case_study = models.ForeignKey(CaseStudy, on_delete=models.CASCADE, related_name='events', null=True, blank=True)
    condition_status = models.BooleanField(default=False)
    condition = models.CharField(max_length=200, blank=True, null=True)
    threshold = models.FloatField(null=True, blank=True)
    channel = models.CharField(max_length=100)
    symbol = models.CharField(max_length=100)

    def __str__(self):
        return self.channel
    