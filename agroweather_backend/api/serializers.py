"""
Serializers for API endpoints
Convert Python objects to JSON and vice versa
"""
from rest_framework import serializers
from weather.models import WeatherForecast, WeatherAlert
from planting.models import PlantingCalendar
from users.models import User, UserCrop


class WeatherForecastSerializer(serializers.ModelSerializer):
    """Serialize weather forecast data"""
    
    class Meta:
        model = WeatherForecast
        fields = [
            'id', 'location', 'latitude', 'longitude', 'forecast_date',
            'temp_max', 'temp_min', 'temp_avg', 'rainfall', 'humidity',
            'wind_speed', 'cloud_cover', 'confidence_score', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class WeatherAlertSerializer(serializers.ModelSerializer):
    """Serialize weather alerts"""
    
    class Meta:
        model = WeatherAlert
        fields = [
            'id', 'location', 'alert_type', 'severity', 'message',
            'start_date', 'end_date', 'is_active', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class PlantingCalendarSerializer(serializers.ModelSerializer):
    """Serialize planting calendar"""
    
    class Meta:
        model = PlantingCalendar
        fields = [
            'id', 'crop_name', 'location', 'planting_date', 'harvest_date',
            'growing_days', 'total_gdd_required', 'confidence_score',
            'calendar_events', 'recommendations', 'risks', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class UserCropSerializer(serializers.ModelSerializer):
    """Serialize user crops"""
    
    class Meta:
        model = UserCrop
        fields = ['id', 'crop_name', 'planting_date']


class UserRegistrationSerializer(serializers.ModelSerializer):
    """Serialize user registration"""
    crops = serializers.ListField(
        child=serializers.CharField(),
        write_only=True,
        required=False
    )
    
    class Meta:
        model = User
        fields = [
            'username', 'email', 'password', 'phone_number',
            'location', 'latitude', 'longitude', 'farm_size',
            'preferred_language', 'crops'
        ]
        extra_kwargs = {
            'password': {'write_only': True}
        }
    
    def create(self, validated_data):
        crops = validated_data.pop('crops', [])
        user = User.objects.create_user(**validated_data)
        
        # Create user crops
        for crop_name in crops:
            UserCrop.objects.create(user=user, crop_name=crop_name)
        
        return user


class UserProfileSerializer(serializers.ModelSerializer):
    """Serialize user profile"""
    crops = UserCropSerializer(many=True, read_only=True)
    
    class Meta:
        model = User
        fields = [
            'id', 'username', 'email', 'phone_number', 'location',
            'latitude', 'longitude', 'farm_size', 'preferred_language',
            'crops', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class ForecastRequestSerializer(serializers.Serializer):
    """Serialize forecast request"""
    latitude = serializers.FloatField(required=True)
    longitude = serializers.FloatField(required=True)
    days = serializers.IntegerField(default=7, min_value=1, max_value=14)


class PlantingCalendarRequestSerializer(serializers.Serializer):
    """Serialize planting calendar request"""
    crop_name = serializers.CharField(required=True)
    planting_date = serializers.DateField(required=True)
    latitude = serializers.FloatField(required=True)
    longitude = serializers.FloatField(required=True)
    location = serializers.CharField(default='Aba')
