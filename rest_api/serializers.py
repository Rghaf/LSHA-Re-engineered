from rest_framework import serializers

from .models import CaseStudy

class CaseStudySerializer(serializers.ModelSerializer):
    class Meta:
        model = CaseStudy
        fields = '__all__'
        read_only_fields = ['id']

    def to_internal_value(self, data):
        if data.get('context_variables') == '':
            data = data.copy()
            data['context_variables'] = []
        return super().to_internal_value(data)

# class EventsSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Event
#         fields = '__all__'
#         read_only_fields = ['id']