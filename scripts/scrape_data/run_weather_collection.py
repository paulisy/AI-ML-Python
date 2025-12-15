#!/usr/bin/env python3
"""
Helper script to run weather collection in chunks to respect API limits.
This makes it easy to collect data incrementally without overwriting existing data.
"""

from collect_weather import main, get_date_ranges_for_api_limits

def run_all_chunks():
    """Run all predefined date chunks automatically."""
    print("🚀 Running all date chunks automatically...")
    main(use_chunks=True)

def run_specific_chunk(chunk_number):
    """Run a specific chunk by number (1-based)."""
    ranges = get_date_ranges_for_api_limits()
    
    if chunk_number < 1 or chunk_number > len(ranges):
        print(f"❌ Invalid chunk number. Choose between 1 and {len(ranges)}")
        return
    
    start_date, end_date = ranges[chunk_number - 1]
    print(f"🚀 Running chunk {chunk_number}: {start_date} to {end_date}")
    main(start_date, end_date)

def show_available_chunks():
    """Show all available date chunks."""
    ranges = get_date_ranges_for_api_limits()
    print("📅 Available date chunks:")
    print("=" * 50)
    
    for i, (start_date, end_date) in enumerate(ranges, 1):
        print(f"Chunk {i}: {start_date} to {end_date}")

def run_custom_range(start_date, end_date):
    """Run a custom date range."""
    print(f"🚀 Running custom range: {start_date} to {end_date}")
    main(start_date, end_date)

if __name__ == "__main__":
    print("🌦️  Weather Collection Helper")
    print("=" * 50)
    
    # UNCOMMENT ONE OF THESE OPTIONS:
    
    # Option 1: Show available chunks
    show_available_chunks()
    
    # Option 2: Run all chunks automatically (be careful with API limits!)
    # run_all_chunks()
    
    # Option 3: Run a specific chunk (change the number)
    # run_specific_chunk(1)  # Change this number (1-6)
    run_specific_chunk(7)
    
    # Option 4: Run custom date range
    # run_custom_range('2024-01-01', '2024-01-31')
    
    print("\n💡 Edit this file to uncomment the option you want to use!")