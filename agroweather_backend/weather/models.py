"""
Weather data models
"""
from django.db import models

class WeatherForecast(models.Model):
    """Store weather forecasts"""
    location = models.CharField(max_length=100)
    latitude = models.FloatField()
    longitude = models.FloatField()
    forecast_date = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Weather parameters
    temp_max = models.FloatField()
    temp_min = models.FloatField()
    temp_avg = models.FloatField()
    rainfall = models.FloatField()
    humidity = models.FloatField()
    wind_speed = models.FloatField()
    cloud_cover = models.FloatField()
    
    # Prediction metadata
    confidence_score = models.FloatField(null=True, blank=True)
    model_version = models.CharField(max_length=50, default='v1.0')
    
    class Meta:
        unique_together = ['location', 'forecast_date', 'created_at']
        ordering = ['forecast_date']
    
    def __str__(self):
        return f"{self.location} - {self.forecast_date}"


class WeatherAlert(models.Model):
    """Weather alerts for farmers"""
    SEVERITY_CHOICES = [
        ('low', 'Low'),
        ('moderate', 'Moderate'),
        ('high', 'High'),
        ('extreme', 'Extreme'),
    ]
    
    location = models.CharField(max_length=100)
    alert_type = models.CharField(max_length=50)  # e.g., 'heavy_rain', 'drought'
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES)
    message = models.TextField()
    start_date = models.DateTimeField()
    end_date = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.alert_type} - {self.location} ({self.severity})"
