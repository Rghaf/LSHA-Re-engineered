from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse

from .tasks import run_lsha_learning_task

def trigger_learning_view(request):
    if request.method == 'POST':
        # Get parameters from POST request (e.g., from a form)
        case_study = request.POST.get('case_study', 'THERMO') # Default to THERMO
        pov = request.POST.get('pov', 'default_pov') # Get POV
        start_dt = request.POST.get('start_date', '2024-01-01')
        end_dt = request.POST.get('end_date', '2024-12-31')

        # Launch the Celery task
        task = run_lsha_learning_task.delay(case_study, pov, start_dt, end_dt)
        print(settings.BASE_DIR)
        print(settings.LSHA_ROOT)
        # print(settings.BASE_DIR)
        return HttpResponse(f"Learning task started with ID: {task.id}. Check logs or status endpoint.")
    else:
        # Simple form for triggering (replace with a proper Django form later)
        return render(request, 'core_algorithm/trigger_form.html')