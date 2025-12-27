# AgroWeather AI - Setup Instructions

## 🚀 Quick Start Guide

Your AgroWeather AI project is now fully set up and ready to use! Here's how to get started:

### 1. Prerequisites ✅
- Python 3.8+ (already installed)
- Django and dependencies (already installed)
- Visual Crossing API key (already configured)

### 2. Start the Django Server 🌐

```bash
cd agroweather_backend
python manage.py runserver
```

The server will start at: **http://127.0.0.1:8000/**

### 3. Test the API 🧪

#### Option A: Use the Web Interface
Open your browser and go to:
```
http://127.0.0.1:8000/test_api.html
```

This provides a user-friendly interface to test all API endpoints.

#### Option B: Use Command Line
```bash
# Health check
curl http://127.0.0.1:8000/api/health/

# Get available crops
curl http://127.0.0.1:8000/api/crops/

# Generate planting calendar
curl -X POST http://127.0.0.1:8000/api/planting/calendar/ \
  -H "Content-Type: application/json" \
  -d '{
    "crop_name": "maize",
    "planting_date": "2025-05-05",
    "latitude": 5.1156,
    "longitude": 7.3636
  }'
```

## 📋 Available API Endpoints

### Core Endpoints
- `GET /api/health/` - Health check
- `GET /api/crops/` - List all available crops
- `GET /api/crops/{crop_name}/` - Get specific crop details
- `POST /api/planting/calendar/` - Generate planting calendar
- `POST /api/weather/forecast/` - Generate weather forecast
- `GET /api/alerts/` - Get weather alerts

### User Management (Optional)
- `POST /api/users/register/` - Register new user
- `GET /api/users/profile/` - Get user profile (requires auth)
- `GET /api/users/calendars/` - Get user's calendars (requires auth)

## 🌾 Supported Crops

1. **Maize** - 90 days, moderate water requirement
2. **Rice** - 120 days, high water requirement  
3. **Cassava** - 300 days, low water requirement
4. **Yam** - 240 days, moderate water requirement
5. **Cowpea** - 70 days, low water requirement

## 🔧 Configuration

### Environment Variables
The project uses environment variables in `agroweather_backend/.env`:

```env
VISUAL_CROSSING_API_KEY=X2VWBMCZP2DXYLARG2RHG4Z2P
```

### Django Settings
Key settings in `agroweather_backend/agroweather/settings.py`:
- Database: SQLite (for development)
- CORS: Enabled for frontend integration
- REST Framework: Configured for API responses

## 📊 Example API Responses

### Planting Calendar Response
```json
{
  "crop": "Maize",
  "location": "Aba",
  "planting_date": "2025-05-05",
  "harvest_date": "2025-08-03",
  "growing_days": 90,
  "confidence_score": 90.0,
  "calendar_events": [
    {
      "date": "2025-05-05",
      "event": "Planting Day",
      "action": "Prepare soil, plant seeds at proper depth and spacing"
    }
  ],
  "recommendations": [
    "✅ Good timing! Currently in optimal planting window for Maize",
    "🌾 Monitor weather closely near harvest time"
  ]
}
```

### Weather Forecast Response
```json
{
  "location": {
    "latitude": 5.1156,
    "longitude": 7.3636,
    "name": "Aba"
  },
  "forecast_days": 7,
  "forecasts": [
    {
      "date": "2025-12-17",
      "rainfall": 5.0,
      "confidence_score": 0.85
    }
  ]
}
```

## 🛠️ Development

### Project Structure
```
agroweather_backend/
├── agroweather/          # Django project settings
├── api/                  # REST API endpoints
├── users/                # User management
├── weather/              # Weather models and services
├── planting/             # Planting calendar logic
├── backend/              # Business logic
├── models/               # ML models
└── manage.py             # Django management
```

### Adding New Features
1. **New Crop**: Add to `CropDatabase.CROPS` in `backend/planting_calendar.py`
2. **New API Endpoint**: Add to `api/views.py` and `api/urls.py`
3. **New Model**: Create in appropriate app's `models.py`

## 🚨 Troubleshooting

### Common Issues

1. **Server won't start**
   ```bash
   cd agroweather_backend
   python manage.py migrate
   python manage.py runserver
   ```

2. **API returns 404**
   - Check that server is running on port 8000
   - Verify URL paths match the endpoints

3. **CORS errors in browser**
   - Already configured for localhost:3000 and localhost:5173
   - Add your frontend URL to `CORS_ALLOWED_ORIGINS` in settings.py

### Getting Help
- Check Django logs in the terminal where you ran `runserver`
- Use the test interface at `/test_api.html` to debug API calls
- Verify API key is set in `.env` file

## 🎯 Next Steps

1. **Frontend Integration**: Connect your React/Vue/Angular app to these APIs
2. **Production Deployment**: Configure for production with proper database and security
3. **ML Model Training**: Train custom models with your local weather data
4. **Mobile App**: Use the REST API to build mobile applications

## 📞 Support

The AgroWeather AI system is now fully functional with:
- ✅ Django REST API server running
- ✅ Database migrations applied
- ✅ All endpoints tested and working
- ✅ Planting calendar generation
- ✅ Weather forecast integration
- ✅ Crop database with 5 Nigerian crops
- ✅ Web interface for testing

You can now successfully test all URLs and integrate with frontend applications!