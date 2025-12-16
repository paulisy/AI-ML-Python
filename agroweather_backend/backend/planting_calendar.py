"""
AgroWeather - Planting Calendar Algorithm
GDD-based crop maturity prediction and planting recommendations.
"""

from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import pickle

@dataclass
class Crop:
    """
    Crop information and requirements.
    """
    name: str
    min_temp: float
    max_temp: float
    optimal_rainfall: float  # mm per season
    min_rainfall: float
    max_rainfall: float
    growing_days: int
    planting_start_month: int
    planting_end_month: int
    base_temp_gdd: float
    soil_types: List[str]
    water_requirement: str
    description: str
    total_gdd_required: int = None  # Will calculate if not provided

    def __post_init__(self):
        """
        Calculate total GDD required based on optimal rainfall if not provided.
        """
        if self.total_gdd_required is None:
            self.total_gdd_required = int(17.5 * self.growing_days)


class CropDatabase:
    """
    Database of crop information.
    """
    CROPS = {
        'maize': Crop(
            name='Maize',
            min_temp=18,
            max_temp=35,
            optimal_rainfall=600,
            min_rainfall=500,
            max_rainfall=800,
            growing_days=90,
            planting_start_month=4,
            planting_end_month=7,
            base_temp_gdd=10,
            soil_types=['loamy', 'sandy-loam'],
            water_requirement='moderate',
            description='Staple cereal crop, drought-tolerant',
            total_gdd_required=1500
        ),
        'rice': Crop(
            name='Rice',
            min_temp=20,
            max_temp=38,
            optimal_rainfall=1200,
            min_rainfall=1000,
            max_rainfall=1800,
            growing_days=120,
            planting_start_month=5,
            planting_end_month=8,
            base_temp_gdd=10,
            soil_types=['clay', 'loamy'],
            water_requirement='high',
            description='Water-intensive staple crop',
            total_gdd_required=2000
        ),
        'cassava': Crop(
            name='Cassava',
            min_temp=20,
            max_temp=35,
            optimal_rainfall=1000,
            min_rainfall=800,
            max_rainfall=1500,
            growing_days=300,
            planting_start_month=3,
            planting_end_month=7,
            base_temp_gdd=15,
            soil_types=['loamy', 'sandy', 'clay'],
            water_requirement='low',
            description='Drought-resistant root crop',
            total_gdd_required=5250
        ),
        'yam': Crop(
            name='Yam',
            min_temp=25,
            max_temp=35,
            optimal_rainfall=1200,
            min_rainfall=1000,
            max_rainfall=1500,
            growing_days=240,
            planting_start_month=3,
            planting_end_month=5,
            base_temp_gdd=18,
            soil_types=['loamy', 'sandy-loam'],
            water_requirement='moderate',
            description='Traditional staple root crop',
            total_gdd_required=4200
        ),
        'cowpea': Crop(
            name='Cowpea',
            min_temp=20,
            max_temp=35,
            optimal_rainfall=500,
            min_rainfall=400,
            max_rainfall=700,
            growing_days=70,
            planting_start_month=5,
            planting_end_month=8,
            base_temp_gdd=8,
            soil_types=['sandy', 'loamy'],
            water_requirement='low',
            description='Nitrogen-fixing legume, drought-tolerant',
            total_gdd_required=1225
        )
    }

    @classmethod
    def get_crop(cls, crop_name:str) -> Optional[Crop]:
        """
        Get crop information by name.
        """
        return cls.CROPS.get(crop_name.lower())
    
    @classmethod
    def list_crop(cls) -> List[str]:
        """
        List all available crops.
        """
        return list(cls.CROPS.keys())


class GDDCalculator:
    """
    Calculate Growing Degree Days (GDD).
    """
    @staticmethod
    def calculate_gdd(temp_max: float, temp_min: float, base_temp: float) -> float:
        """
        Calculate GDD for a single day.

        Formula: GDD = (Tmax + Tmin) / 2 - Tbase
        GDD cannot be negative (if avg temp < base temp, GDD = 0)
        
        Args:
            temp_max: Maximum temperature (°C)
            temp_min: Minimum temperature (°C)
            base_temp: Base temperature for crop (°C)
        
        Returns:
            GDD for the day
        """
        avg_temp = (temp_max + temp_min) / 2
        return max(0, avg_temp - base_temp_gdd)

    @staticmethod
    def calculate_days_to_maturity(total_gdd_required: float, avg_gdd_per_day: float = 17.5) -> int:
        """
        Calculate days needed for crop to mature
        
        Args:
            total_gdd_required: Total GDD crop needs
            avg_gdd_per_day: Average GDD per day (from historical data)
        
        Returns:
            Estimated days to maturity
        """
        return int(np.ceil(total_gdd_required / avg_gdd_per_day))
    
    @staticmethod
    def accumulate_gdd(daily_temps: List[Tuple[float, float]], 
                      base_temp: float) -> List[float]:
        """
        Calculate cumulative GDD over multiple days
        
        Args:
            daily_temps: List of (temp_max, temp_min) tuples
            base_temp: Base temperature
        
        Returns:
            List of cumulative GDD values
        """
        gdd_cumulative = []
        total = 0
        
        for temp_max, temp_min in daily_temps:
            daily_gdd = GDDCalculator.calculate_gdd(temp_max, temp_min, base_temp)
            total += daily_gdd
            gdd_cumulative.append(total)
        
        return gdd_cumulative


class PlantingWindowAnalyzer:
    """Analyze optimal planting windows"""
    
    @staticmethod
    def is_in_planting_season(current_date: datetime, crop: Crop) -> bool:
        """Check if current date is in crop's planting season"""
        current_month = current_date.month
        
        start_month = crop.planting_start_month
        end_month = crop.planting_end_month
        
        if start_month <= end_month:
            # Normal case: April-July
            return start_month <= current_month <= end_month
        else:
            # Wraps around year: Nov-Feb
            return current_month >= start_month or current_month <= end_month
    
    @staticmethod
    def get_season_name(month: int) -> str:
        """Get season name for Nigerian climate"""
        if month in [11, 12, 1, 2, 3]:
            return "dry_season"
        else:
            return "rainy_season"
    
    @staticmethod
    def calculate_planting_window(current_date: datetime, 
                                  crop: Crop) -> Dict:
        """
        Calculate optimal planting window
        
        Returns:
            Dictionary with window start, end, and status
        """
        current_month = current_date.month
        current_year = current_date.year
        
        # Calculate window start date
        if current_month < crop.planting_start_month:
            # Before planting season
            window_start = datetime(current_year, crop.planting_start_month, 1)
            window_end = datetime(current_year, crop.planting_end_month, 28)
            status = "upcoming"
        elif current_month > crop.planting_end_month:
            # After planting season - suggest next year
            window_start = datetime(current_year + 1, crop.planting_start_month, 1)
            window_end = datetime(current_year + 1, crop.planting_end_month, 28)
            status = "missed"
        else:
            # During planting season
            window_start = datetime(current_year, crop.planting_start_month, 1)
            window_end = datetime(current_year, crop.planting_end_month, 28)
            status = "current"
        
        return {
            'window_start': window_start,
            'window_end': window_end,
            'status': status,
            'days_until_start': (window_start - current_date).days,
            'days_until_end': (window_end - current_date).days
        }


class RiskAssessor:
    """Assess planting risks"""
    
    @staticmethod
    def assess_harvest_timing_risk(harvest_date: datetime) -> Dict:
        """
        Assess risk of harvest timing
        Peak rain in Aba: September
        """
        harvest_month = harvest_date.month
        
        if harvest_month == 9:
            return {
                'risk_level': 'high',
                'reason': 'Harvest during peak rain season (September)',
                'recommendation': 'Consider planting 2 weeks earlier or using early-maturing variety'
            }
        elif harvest_month in [8, 10]:
            return {
                'risk_level': 'moderate',
                'reason': f'Harvest in {"August" if harvest_month == 8 else "October"} (high rain period)',
                'recommendation': 'Monitor weather closely near harvest time'
            }
        elif harvest_month in [1, 2, 12]:
            return {
                'risk_level': 'low',
                'reason': 'Harvest during dry season',
                'recommendation': 'Optimal harvest conditions'
            }
        else:
            return {
                'risk_level': 'low',
                'reason': 'Acceptable harvest timing',
                'recommendation': 'Standard precautions'
            }
    
    @staticmethod
    def assess_rainfall_adequacy(forecast_rainfall: float, crop: Crop) -> Dict:
        """Assess if forecasted rainfall is adequate"""
        
        if forecast_rainfall < crop.min_rainfall:
            return {
                'adequate': False,
                'status': 'too_low',
                'message': f'Insufficient rainfall expected ({forecast_rainfall:.0f}mm < {crop.min_rainfall}mm required)',
                'recommendation': 'Consider irrigation or wait for better conditions'
            }
        elif forecast_rainfall > crop.max_rainfall:
            return {
                'adequate': False,
                'status': 'too_high',
                'message': f'Excessive rainfall risk ({forecast_rainfall:.0f}mm > {crop.max_rainfall}mm)',
                'recommendation': 'Ensure proper drainage or consider delaying'
            }
        else:
            return {
                'adequate': True,
                'status': 'optimal',
                'message': f'Rainfall conditions favorable ({forecast_rainfall:.0f}mm)',
                'recommendation': 'Good conditions for planting'
            }


class PlantingCalendarGenerator:
    """Main class for generating planting calendars"""
    
    def __init__(self):
        self.crop_db = CropDatabase()
        self.gdd_calc = GDDCalculator()
        self.window_analyzer = PlantingWindowAnalyzer()
        self.risk_assessor = RiskAssessor()
    
    def generate_calendar(self, 
                         crop_name: str,
                         planting_date: datetime,
                         location: str = "Aba",
                         forecast_data: Optional[Dict] = None) -> Dict:
        """
        Generate complete planting calendar
        
        Args:
            crop_name: Name of crop
            planting_date: Proposed planting date
            location: Location (default: Aba)
            forecast_data: Optional weather forecast data
        
        Returns:
            Complete calendar with recommendations
        """
        # Get crop info
        crop = self.crop_db.get_crop(crop_name)
        if not crop:
            raise ValueError(f"Crop '{crop_name}' not found in database")
        
        # Calculate harvest date (using GDD)
        # For simplicity, using average GDD (17.5/day for Aba)
        days_to_maturity = crop.growing_days
        harvest_date = planting_date + timedelta(days=days_to_maturity)
        
        # Calculate GDD accumulation
        total_gdd = crop.total_gdd_required
        avg_gdd_per_day = 17.5  # From EDA
        
        # Check planting window
        window_info = self.window_analyzer.calculate_planting_window(
            planting_date, crop
        )
        
        # Assess risks
        harvest_risk = self.risk_assessor.assess_harvest_timing_risk(harvest_date)
        
        # Generate week-by-week calendar
        calendar_events = self._generate_calendar_events(
            planting_date, harvest_date, crop
        )
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            planting_date, crop, window_info, harvest_risk
        )
        
        return {
            'crop': crop.name,
            'location': location,
            'planting_date': planting_date.strftime('%Y-%m-%d'),
            'harvest_date': harvest_date.strftime('%Y-%m-%d'),
            'growing_days': days_to_maturity,
            'total_gdd_required': total_gdd,
            'planting_window': {
                'start': window_info['window_start'].strftime('%Y-%m-%d'),
                'end': window_info['window_end'].strftime('%Y-%m-%d'),
                'status': window_info['status']
            },
            'confidence_score': confidence,
            'risks': harvest_risk,
            'calendar_events': calendar_events,
            'recommendations': self._generate_recommendations(
                crop, planting_date, window_info, harvest_risk
            )
        }
    
    def _generate_calendar_events(self, 
                                  planting_date: datetime,
                                  harvest_date: datetime,
                                  crop: Crop) -> List[Dict]:
        """Generate week-by-week calendar events"""
        events = []
        
        # Planting
        events.append({
            'date': planting_date.strftime('%Y-%m-%d'),
            'day': 0,
            'event': 'Planting Day',
            'description': f'Plant {crop.name} seeds',
            'action': 'Prepare soil, plant seeds at proper depth and spacing'
        })
        
        # Germination (typically 7-10 days)
        germination_date = planting_date + timedelta(days=8)
        events.append({
            'date': germination_date.strftime('%Y-%m-%d'),
            'day': 8,
            'event': 'Germination Expected',
            'description': 'Seeds should sprout',
            'action': 'Monitor soil moisture, thin seedlings if necessary'
        })
        
        # First fertilizer application (2-3 weeks)
        fert1_date = planting_date + timedelta(days=21)
        events.append({
            'date': fert1_date.strftime('%Y-%m-%d'),
            'day': 21,
            'event': 'First Fertilizer Application',
            'description': 'Apply base fertilizer',
            'action': 'Apply NPK fertilizer, ensure adequate moisture'
        })
        
        # Mid-season (flowering for many crops)
        midseason = crop.growing_days // 2
        midseason_date = planting_date + timedelta(days=midseason)
        events.append({
            'date': midseason_date.strftime('%Y-%m-%d'),
            'day': midseason,
            'event': 'Mid-Season / Flowering',
            'description': 'Critical growth stage',
            'action': 'Apply second fertilizer, monitor for pests'
        })
        
        # Pre-harvest (1 week before)
        preharvest_date = harvest_date - timedelta(days=7)
        events.append({
            'date': preharvest_date.strftime('%Y-%m-%d'),
            'day': crop.growing_days - 7,
            'event': 'Pre-Harvest Preparation',
            'description': 'Crop maturity approaching',
            'action': 'Check crop maturity, prepare harvesting equipment'
        })
        
        # Harvest
        events.append({
            'date': harvest_date.strftime('%Y-%m-%d'),
            'day': crop.growing_days,
            'event': 'Expected Harvest Date',
            'description': f'{crop.name} should be mature',
            'action': 'Harvest crop, proper drying and storage'
        })
        
        return events
    
    def _calculate_confidence(self, 
                             planting_date: datetime,
                             crop: Crop,
                             window_info: Dict,
                             harvest_risk: Dict) -> float:
        """
        Calculate confidence score (0-100)
        """
        confidence = 100.0
        
        # Penalty for planting outside optimal window
        if window_info['status'] == 'missed':
            confidence -= 30
        elif window_info['status'] == 'upcoming':
            if window_info['days_until_start'] > 30:
                confidence -= 20
        
        # Penalty for harvest risk
        if harvest_risk['risk_level'] == 'high':
            confidence -= 25
        elif harvest_risk['risk_level'] == 'moderate':
            confidence -= 10
        
        # Check season appropriateness
        season = self.window_analyzer.get_season_name(planting_date.month)
        if season == 'dry_season' and crop.water_requirement == 'high':
            confidence -= 20
        
        return max(0, min(100, confidence))
    
    def _generate_recommendations(self,
                                 crop: Crop,
                                 planting_date: datetime,
                                 window_info: Dict,
                                 harvest_risk: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Timing recommendations
        if window_info['status'] == 'current':
            recommendations.append(f"✅ Good timing! Currently in optimal planting window for {crop.name}")
        elif window_info['status'] == 'upcoming':
            days_until = window_info['days_until_start']
            recommendations.append(f"⏰ Planting season starts in {days_until} days. Prepare soil now.")
        else:
            recommendations.append(f"⚠️ Outside optimal planting window. Consider waiting until {window_info['window_start'].strftime('%B')}")
        
        # Harvest risk recommendations
        recommendations.append(f"🌾 {harvest_risk['recommendation']}")
        
        # Water requirement recommendations
        if crop.water_requirement == 'high':
            recommendations.append(f"💧 {crop.name} requires high water. Ensure irrigation is available during dry spells")
        elif crop.water_requirement == 'low':
            recommendations.append(f"🌵 {crop.name} is drought-tolerant. Good choice for areas with uncertain rainfall")
        
        # Soil recommendations
        if crop.soil_types:
            soil_list = ', '.join(crop.soil_types)
            recommendations.append(f"🌱 Best soil types: {soil_list}")
        
        return recommendations


def test_planting_calendar():
    """Test the planting calendar generator"""
    print("🌾 Testing Planting Calendar Generator")
    print("="*70)
    
    generator = PlantingCalendarGenerator()
    
    # Test case: Maize planting in May
    test_date = datetime(2025, 5, 5)
    
    print(f"\n📅 Generating calendar for Maize planted on {test_date.strftime('%B %d, %Y')}")
    print("-"*70)
    
    calendar = generator.generate_calendar(
        crop_name='maize',
        planting_date=test_date
    )
    
    print(f"\n🌽 Crop: {calendar['crop']}")
    print(f"📍 Location: {calendar['location']}")
    print(f"🌱 Planting Date: {calendar['planting_date']}")
    print(f"🌾 Harvest Date: {calendar['harvest_date']}")
    print(f"📊 Growing Days: {calendar['growing_days']}")
    print(f"🌡️  Total GDD Required: {calendar['total_gdd_required']}")
    print(f"⭐ Confidence Score: {calendar['confidence_score']:.0f}%")
    
    print(f"\n⚠️  Risk Assessment:")
    print(f"   Level: {calendar['risks']['risk_level'].upper()}")
    print(f"   Reason: {calendar['risks']['reason']}")
    print(f"   Recommendation: {calendar['risks']['recommendation']}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(calendar['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    print(f"\n📅 Calendar Events:")
    for event in calendar['calendar_events']:
        print(f"\n   Day {event['day']:3d} - {event['date']}")
        print(f"   {event['event']}")
        print(f"   → {event['action']}")
    
    print("\n" + "="*70)
    print("✅ Test complete!")


if __name__ == "__main__":
    # Create backend directory if it doesn't exist
    import os
    os.makedirs('backend', exist_ok=True)
    
    # Run test
    test_planting_calendar()
