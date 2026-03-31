from django.db import models

# Create your models here.
class CaseStudy(models.Model):
    id = models.AutoField(primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=200)
    email = models.CharField(max_length=100, null=True, blank=True)
    resample_strategy = models.CharField(max_length=100, default='UPPAAL')
    uppaal_model_file = models.FileField(upload_to='uploads/uppaal/models', null=True, blank=True)
    uppaal_query_file = models.FileField(upload_to='uploads/uppaal/queries', null=True, blank=True)
    csv_file = models.FileField(upload_to='uploads/csv/files', null=True, blank=True)
    driver_signal = models.CharField(max_length=100, null=True, blank=True)
    uppaal_query = models.JSONField(null=True, blank=True, default=dict)
    main_variable = models.CharField(max_length=100, null=True, blank=True, default='')
    context_variables = models.JSONField(null=True, blank=True, default=list)
    user_json = models.JSONField(null=True, blank=True)
    noise = models.FloatField(default=0.0)
    n_min = models.IntegerField(null = True, blank = True, default = 10)
    p_value = models.FloatField(default=0.05)
    mi_query = models.BooleanField(default=False)
    plot_ddtw = models.BooleanField(default=False)
    ht_query = models.BooleanField(default=False)
    ht_query_type = models.CharField(max_length=100, null=True, blank=True)
    eq_condition = models.CharField(max_length=100, null=True, blank=True)
    is_aggregation = models.BooleanField(null = True, blank = True, default = False)
    final_result_txt = models.FileField(upload_to='results/final_results', null=True, blank=True)
    final_result_pdf = models.FileField(upload_to='results/final_results', null=True, blank=True)

    def __str__(self):
        # If name exists, return it. If not, return a placeholder with the ID.
        if self.name:
            return self.name
        return f"Unnamed List (ID: {self.id})"

