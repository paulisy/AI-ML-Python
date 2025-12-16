"""
API URL Configuration
"""
from django.urls import path
from . import views

urlpatterns = [
    # Health check
    path('health/', views.health_check, name='health-check'),
    
    # Crops
    path('crops/', views.list_crops, name='list-crops'),
    path('crops/<str:crop_name>/', views.crop_detail, name='crop-detail'),
    
    # Weather
    path('weather/forecast/', views.generate_forecast, name='generate-forecast'),
    path('alerts/', views.list_alerts, name='list-alerts'),
    
    # Planting
    path('planting/calendar/', views.generate_planting_calendar, name='generate-calendar'),
    
    # Users
    path('users/register/', views.user_registration, name='user-registration'),
    path('users/profile/', views.user_profile, name='user-profile'),
    path('users/calendars/', views.user_calendars, name='user-calendars'),
]

