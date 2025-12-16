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