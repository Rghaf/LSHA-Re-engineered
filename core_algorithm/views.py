from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# Import the model from your OTHER app
from rest_api.models import CaseStudy 
from .tasks import run_lsha_learning_task

@api_view(['POST'])
def run_algorithm(request):
    """
    API Endpoint to trigger the L*SHA algorithm for an EXISTING Case Study.
    
    Expects JSON payload:
    {
        "case_study_id": 28
    }
    """
    try:
        # 1. Get the ID from the request
        case_study_id = request.data.get('case_study_id')
        
        if not case_study_id:
            return Response(
                {"error": "case_study_id is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        # 2. Verify the Case Study exists
        try:
            case_study = CaseStudy.objects.get(id=case_study_id)
        except CaseStudy.DoesNotExist:
            return Response(
                {"error": f"Case Study with ID {case_study_id} not found."},
                status=status.HTTP_404_NOT_FOUND
            )

        # 3. Trigger the Celery Task
        # We pass the ID so the worker can fetch the fresh data from DB
        task = run_lsha_learning_task.delay(case_study_id)

        return Response({
            "message": "Algorithm started successfully.",
            "task_id": task.id,
            "case_study_name": case_study.name
        }, status=status.HTTP_202_ACCEPTED)

    except Exception as e:
        return Response(
            {"error": str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )