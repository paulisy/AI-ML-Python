#!/usr/bin/env python3
"""
Migration script to clean up old project structure
This script helps transition from the old structure to the new organized structure
"""
import os
import shutil
import sys


def safe_remove(path):
    """Safely remove file or directory"""
    try:
        if os.path.isfile(path):
            os.remove(path)
            print(f"   ✅ Removed file: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"   ✅ Removed directory: {path}")
        else:
            print(f"   ⚠️  Path not found: {path}")
    except Exception as e:
        print(f"   ❌ Error removing {path}: {e}")


def main():
    """Main migration function"""
    print("🔄 AgroWeather AI - Structure Migration")
    print("=" * 50)
    print("This script will clean up the old project structure")
    print("and remove redundant files after refactoring.")
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed with cleanup? (y/N): ")
    if response.lower() != 'y':
        print("❌ Migration cancelled.")
        return
    
    print("\n🧹 Starting cleanup...")
    
    # Old directories to remove
    old_dirs = [
        "scripts/clean_data_and_feature_extraction",
        "scripts/scrape_data", 
        "scripts/models/__pycache__",
        "scripts/__pycache__",
        "scripts/models/models"  # Old nested models directory
    ]
    
    # Old files to remove
    old_files = [
        "scripts/models/test_import.py",
        "scripts/models/__init__.py"
    ]
    
    print("\n📂 Removing old directories:")
    for directory in old_dirs:
        safe_remove(directory)
    
    print("\n📄 Removing old files:")
    for file in old_files:
        safe_remove(file)
    
    # Keep the original LSTM model as backup
    if os.path.exists("scripts/models/lstm_model.py"):
        backup_path = "scripts/models/lstm_model_backup.py"
        try:
            shutil.copy2("scripts/models/lstm_model.py", backup_path)
            print(f"   ✅ Created backup: {backup_path}")
        except Exception as e:
            print(f"   ⚠️  Could not create backup: {e}")
    
    # Keep the original prepare_ml_data as backup
    if os.path.exists("scripts/models/prepare_ml_data.py"):
        backup_path = "scripts/models/prepare_ml_data_backup.py"
        try:
            shutil.copy2("scripts/models/prepare_ml_data.py", backup_path)
            print(f"   ✅ Created backup: {backup_path}")
        except Exception as e:
            print(f"   ⚠️  Could not create backup: {e}")
    
    print("\n📋 Migration Summary:")
    print("-" * 30)
    print("✅ Old structure cleaned up")
    print("✅ Backups created for important files")
    print("✅ New structure is ready to use")
    
    print("\n🎯 Next Steps:")
    print("1. Test the new structure: python scripts/project_info.py")
    print("2. Run tests: python -m pytest tests/")
    print("3. Try the pipeline: python scripts/run_full_pipeline.py --help")
    
    print("\n" + "=" * 50)
    print("✅ Migration completed successfully!")


if __name__ == "__main__":
    main()