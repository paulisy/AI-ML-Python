"""
Planting calendar models
"""
from django.db import models
from users.models import User

class PlantingCalendar(models.Model):
    """Store generated planting calendars"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='calendars', null=True, blank=True)
    crop_name = models.CharField(max_length=50)
    location = models.CharField(max_length=100)
    planting_date = models.DateField()
    harvest_date = models.DateField()
    growing_days = models.IntegerField()
    total_gdd_required = models.IntegerField()
    confidence_score = models.FloatField()
    
    # Calendar data (stored as JSON)
    calendar_events = models.JSONField()
    recommendations = models.JSONField()
    risks = models.JSONField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.crop_name} - {self.planting_date}"