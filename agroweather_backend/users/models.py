"""
User models for AgroWeather AI
"""
from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    """Extended user model for farmers"""
    phone_number = models.CharField(max_length=20, unique=True)
    location = models.CharField(max_length=100, blank=True)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    farm_size = models.CharField(max_length=50, blank=True)  # e.g., "1-5 hectares"
    preferred_language = models.CharField(max_length=10, default='en')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.username} - {self.phone_number}"


class UserCrop(models.Model):
    """Crops grown by user"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='crops')
    crop_name = models.CharField(max_length=50)
    planting_date = models.DateField(null=True, blank=True)
    
    class Meta:
        unique_together = ['user', 'crop_name']
    
    def __str__(self):
        return f"{self.user.username} - {self.crop_name}"