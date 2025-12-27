"""
API Views for AgroWeather AI
"""
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from datetime import datetime, timedelta
import sys
import os

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .serializers import (
    WeatherForecastSerializer,
    WeatherAlertSerializer,
    PlantingCalendarSerializer,
    UserRegistrationSerializer,
    UserProfileSerializer,
    ForecastRequestSerializer,
    PlantingCalendarRequestSerializer
)
from weather.models import WeatherForecast, WeatherAlert
from planting.models import PlantingCalendar
from users.models import User

# Import business logic
from backend.planting_calendar import PlantingCalendarGenerator, CropDatabase
from weather.services import get_weather_service



@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """
    Health check endpoint
    GET /api/health/
    """
    return Response({
        'status': 'healthy',
        'service': 'AgroWeather AI API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })


@api_view(['GET'])
@permission_classes([AllowAny])
def list_crops(request):
    """
    List all available crops
    GET /api/crops/
    """
    crops_db = CropDatabase()
    crops = []
    
    for crop_name in crops_db.list_crop():
        crop = crops_db.get_crop(crop_name)
        crops.append({
            'name': crop.name,
            'growing_days': crop.growing_days,
            'water_requirement': crop.water_requirement,
            'description': crop.description,
            'planting_months': f"{crop.planting_start_month}-{crop.planting_end_month}",
            'total_gdd_required': crop.total_gdd_required
        })
    
    return Response({
        'count': len(crops),
        'crops': crops
    })


@api_view(['GET'])
@permission_classes([AllowAny])
def crop_detail(request, crop_name):
    """
    Get details for a specific crop
    GET /api/crops/{crop_name}/
    """
    crops_db = CropDatabase()
    crop = crops_db.get_crop(crop_name)
    
    if not crop:
        return Response({
            'error': f"Crop '{crop_name}' not found"
        }, status=status.HTTP_404_NOT_FOUND)
    
    return Response({
        'name': crop.name,
        'min_temp': crop.min_temp,
        'max_temp': crop.max_temp,
        'optimal_rainfall': crop.optimal_rainfall,
        'min_rainfall': crop.min_rainfall,
        'max_rainfall': crop.max_rainfall,
        'growing_days': crop.growing_days,
        'planting_start_month': crop.planting_start_month,
        'planting_end_month': crop.planting_end_month,
        'base_temp_gdd': crop.base_temp_gdd,
        'total_gdd_required': crop.total_gdd_required,
        'soil_types': crop.soil_types,
        'water_requirement': crop.water_requirement,
        'description': crop.description
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def generate_forecast(request):
    """
    Generate weather forecast
    POST /api/weather/forecast/
    
    Request body:
    {
        "latitude": 5.1156,
        "longitude": 7.3636,
        "days": 7
    }
    """
    serializer = ForecastRequestSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    # TODO: Integrate LSTM model for actual predictions
    # For now, return mock data
    
    latitude = serializer.validated_data['latitude']
    longitude = serializer.validated_data['longitude']
    days = serializer.validated_data['days']
    
    forecasts = []
    base_date = datetime.now().date()
    
    for i in range(days):
        forecast_date = base_date + timedelta(days=i)
        forecasts.append({
            'date': forecast_date.isoformat(),
            'temp_max': 32.0 + i * 0.5,
            'temp_min': 24.0,
            'temp_avg': 28.0,
            'rainfall': 5.0 if i % 2 == 0 else 0.0,
            'humidity': 80.0,
            'wind_speed': 12.0,
            'cloud_cover': 60.0,
            'confidence_score': 0.85
        })
    
    return Response({
        'location': {
            'latitude': latitude,
            'longitude': longitude,
            'name': 'Aba'
        },
        'forecast_days': days,
        'forecasts': forecasts,
        'generated_at': datetime.now().isoformat()
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def generate_planting_calendar(request):
    """
    Generate planting calendar
    POST /api/planting/calendar/
    
    Request body:
    {
        "crop_name": "maize",
        "planting_date": "2025-05-05",
        "latitude": 5.1156,
        "longitude": 7.3636,
        "location": "Aba"
    }
    """
    serializer = PlantingCalendarRequestSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    # Extract validated data
    crop_name = serializer.validated_data['crop_name']
    planting_date_str = serializer.validated_data['planting_date']
    planting_date = datetime.strptime(str(planting_date_str), '%Y-%m-%d')
    location = serializer.validated_data.get('location', 'Aba')
    
    # Generate calendar using business logic
    try:
        generator = PlantingCalendarGenerator()
        calendar_data = generator.generate_calendar(
            crop_name=crop_name,
            planting_date=planting_date,
            location=location
        )
        
        # Save to database (optional - for logged-in users)
        if request.user.is_authenticated:
            calendar_obj = PlantingCalendar.objects.create(
                user=request.user,
                crop_name=calendar_data['crop'],
                location=calendar_data['location'],
                planting_date=calendar_data['planting_date'],
                harvest_date=calendar_data['harvest_date'],
                growing_days=calendar_data['growing_days'],
                total_gdd_required=calendar_data['total_gdd_required'],
                confidence_score=calendar_data['confidence_score'],
                calendar_events=calendar_data['calendar_events'],
                recommendations=calendar_data['recommendations'],
                risks=calendar_data['risks']
            )
        
        return Response(calendar_data, status=status.HTTP_201_CREATED)
        
    except ValueError as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({
            'error': f'Failed to generate calendar: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([AllowAny])
def list_alerts(request):
    """
    List active weather alerts
    GET /api/alerts/
    """
    alerts = WeatherAlert.objects.filter(is_active=True)
    serializer = WeatherAlertSerializer(alerts, many=True)
    
    return Response({
        'count': alerts.count(),
        'alerts': serializer.data
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def user_registration(request):
    """
    Register new user
    POST /api/users/register/
    
    Request body:
    {
        "username": "farmer1",
        "email": "farmer@example.com",
        "password": "securepass123",
        "phone_number": "+234XXXXXXXXXX",
        "location": "Aba",
        "latitude": 5.1156,
        "longitude": 7.3636,
        "farm_size": "1-5 hectares",
        "crops": ["maize", "rice"]
    }
    """
    serializer = UserRegistrationSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    user = serializer.save()
    
    return Response({
        'message': 'User registered successfully',
        'user': {
            'id': user.id,
            'username': user.username,
            'phone_number': user.phone_number,
            'location': user.location
        }
    }, status=status.HTTP_201_CREATED)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_profile(request):
    """
    Get user profile
    GET /api/users/profile/
    
    Requires authentication
    """
    serializer = UserProfileSerializer(request.user)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_calendars(request):
    """
    Get user's planting calendars
    GET /api/users/calendars/
    """
    calendars = PlantingCalendar.objects.filter(user=request.user)
    serializer = PlantingCalendarSerializer(calendars, many=True)
    
    return Response({
        'count': calendars.count(),
        'calendars': serializer.data
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def generate_forecast(request):
    """
    Generate weather forecast using LSTM model (with fallback to mock data)
    POST /api/weather/forecast/
    
    Request body:
    {
        "latitude": 5.1156,
        "longitude": 7.3636,
        "days": 7
    }
    """
    serializer = ForecastRequestSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    latitude = serializer.validated_data['latitude']
    longitude = serializer.validated_data['longitude']
    days = serializer.validated_data['days']
    
    try:
        # Try to get weather service (loads LSTM model)
        weather_service = get_weather_service()
        forecast = weather_service.generate_forecast(latitude, longitude, days)
        return Response(forecast, status=status.HTTP_200_OK)
        
    except FileNotFoundError:
        # Fallback to mock forecast when model files are missing
        return _generate_mock_forecast(latitude, longitude, days)
        
    except Exception as e:
        # Fallback to mock forecast for any other errors
        return _generate_mock_forecast(latitude, longitude, days)


def _generate_mock_forecast(latitude, longitude, days):
    """Generate mock weather forecast for demo purposes"""
    import numpy as np
    from datetime import datetime, timedelta
    
    forecasts = []
    base_date = datetime.now().date()
    
    # Simulate realistic weather patterns for Aba, Nigeria
    for i in range(days):
        pred_date = base_date + timedelta(days=i+1)
        
        # Seasonal rainfall patterns (higher in rainy season: April-October)
        month = pred_date.month
        if 4 <= month <= 10:  # Rainy season
            rainfall_base = 8.0
            rain_probability = 0.6
        else:  # Dry season
            rainfall_base = 1.0
            rain_probability = 0.2
        
        # Generate rainfall with some randomness
        if np.random.random() < rain_probability:
            rainfall = max(0, np.random.exponential(rainfall_base))
        else:
            rainfall = 0.0
        
        # Temperature varies slightly
        temp_max = 32.0 + np.random.normal(0, 2)
        temp_min = 24.0 + np.random.normal(0, 1.5)
        temp_avg = (temp_max + temp_min) / 2
        
        forecasts.append({
            'date': pred_date.isoformat(),
            'temp_max': round(temp_max, 1),
            'temp_min': round(temp_min, 1),
            'temp_avg': round(temp_avg, 1),
            'rainfall': round(rainfall, 1),
            'humidity': round(80.0 + np.random.normal(0, 5), 1),
            'wind_speed': round(12.0 + np.random.normal(0, 3), 1),
            'cloud_cover': round(60.0 + np.random.normal(0, 15), 1),
            'confidence_score': 0.75  # Lower confidence for mock data
        })
    
    return Response({
        'location': {
            'latitude': latitude,
            'longitude': longitude,
            'name': 'Aba'
        },
        'forecast_days': days,
        'forecasts': forecasts,
        'generated_at': datetime.now().isoformat(),
        'model_info': {
            'version': 'v1.0-mock',
            'note': 'Using mock data - train LSTM model for real predictions',
            'accuracy': 'Demo only'
        }
    }, status=status.HTTP_200_OK)
