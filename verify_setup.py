"""
Setup Verification Script for India Olympics Project
Run this to verify all data files and dependencies are correctly installed.
"""

import sys
import os

def check_dependencies():
    """Check if all required Python packages are installed."""
    print("=" * 60)
    print("1. CHECKING PYTHON PACKAGES")
    print("=" * 60)
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'streamlit': 'streamlit',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib'
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name:20} - Installed")
        except ImportError:
            print(f"✗ {package_name:20} - NOT FOUND")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages are installed!")
        return True

def check_data_files():
    """Check if data files exist and are valid."""
    print("\n" + "=" * 60)
    print("2. CHECKING DATA FILES")
    print("=" * 60)
    
    data_dir = 'data'
    required_files = ['athlete_events.csv', 'noc_regions.csv']
    
    if not os.path.exists(data_dir):
        print(f"✗ Data directory '{data_dir}' not found!")
        return False
    
    all_good = True
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✓ {filename:25} - Found ({size_mb:.1f} MB)")
        else:
            print(f"✗ {filename:25} - NOT FOUND")
            all_good = False
    
    return all_good

def verify_data_content():
    """Verify data can be loaded and contains India data."""
    print("\n" + "=" * 60)
    print("3. VERIFYING DATA CONTENT")
    print("=" * 60)
    
    try:
        import pandas as pd
        
        # Load athlete_events.csv
        print("Loading athlete_events.csv...")
        df = pd.read_csv('data/athlete_events.csv')
        print(f"✓ Total records: {len(df):,}")
        print(f"✓ Years covered: {df['Year'].min()} - {df['Year'].max()}")
        print(f"✓ Columns: {len(df.columns)} ({', '.join(df.columns[:5])}, ...)")
        
        # Filter for India
        print("\nFiltering for India (Summer Olympics, 1948+)...")
        india_df = df[
            (df['Team'] == 'India') & 
            (df['Season'] == 'Summer') & 
            (df['Year'] >= 1948)
        ]
        
        if len(india_df) == 0:
            print("✗ No data found for India! Check Team name in dataset.")
            return False
        
        print(f"✓ India records: {len(india_df):,}")
        print(f"✓ Year range: {india_df['Year'].min()} - {india_df['Year'].max()}")
        print(f"✓ Unique Olympics: {india_df['Year'].nunique()}")
        print(f"✓ Unique athletes: {india_df['Name'].nunique()}")
        print(f"✓ Sports participated: {india_df['Sport'].nunique()}")
        
        # Medal count - IMPORTANT: Show both records and actual medals
        medals_won = india_df[india_df['Medal'].notna()]
        print(f"✓ Medal records (individual athletes): {len(medals_won)}")
        
        # Calculate ACTUAL unique medals
        unique_medals = medals_won.groupby(['Year', 'Sport', 'Event', 'Medal']).size().reset_index()
        print(f"✓ Actual unique medals won: {len(unique_medals)}")
        print(f"   (Note: Team sports count each member separately in records)")
        
        if len(unique_medals) > 0:
            print(f"\nActual medal breakdown:")
            for medal_type, count in unique_medals['Medal'].value_counts().items():
                print(f"  - {medal_type}: {count}")
            print(f"\n  TOTAL: {len(unique_medals)} medals (1948-2016)")
            print(f"  NOTE: Dataset ends at 2016. Missing 2020 & 2024 Olympics.")
        
        # Load noc_regions.csv
        print("\nLoading noc_regions.csv...")
        noc_df = pd.read_csv('data/noc_regions.csv')
        print(f"✓ Total NOC regions: {len(noc_df)}")
        
        india_noc = noc_df[noc_df['region'] == 'India']
        if not india_noc.empty:
            print(f"✓ India NOC code: {india_noc['NOC'].values[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False

def check_project_structure():
    """Verify project folder structure."""
    print("\n" + "=" * 60)
    print("4. CHECKING PROJECT STRUCTURE")
    print("=" * 60)
    
    required_structure = {
        'data': 'folder',
        'notebooks': 'folder',
        'app': 'folder',
        'notebooks/olympic_analysis.ipynb': 'file',
        'app/streamlit_app.py': 'file',
        'requirements.txt': 'file',
        'README.md': 'file'
    }
    
    all_good = True
    for path, item_type in required_structure.items():
        exists = os.path.isdir(path) if item_type == 'folder' else os.path.isfile(path)
        status = "✓" if exists else "✗"
        print(f"{status} {path}")
        if not exists:
            all_good = False
    
    return all_good

def main():
    """Run all verification checks."""
    print("\n" + "🏅" * 30)
    print("INDIA OLYMPICS PROJECT - SETUP VERIFICATION")
    print("🏅" * 30 + "\n")
    
    checks = [
        ("Python packages", check_dependencies),
        ("Data files", check_data_files),
        ("Data content", verify_data_content),
        ("Project structure", check_project_structure)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n⚠ Error during {check_name} check: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{check_name:25} - {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED! Your setup is ready.")
        print("\nNext steps:")
        print("1. Run: jupyter notebook")
        print("2. Open: notebooks/olympic_analysis.ipynb")
        print("3. Run all cells to process data and train models")
        print("4. Then run: cd app && streamlit run streamlit_app.py")
    else:
        print("⚠ SOME CHECKS FAILED. Please fix the issues above.")
        print("\nCommon solutions:")
        print("- Install packages: pip install -r requirements.txt")
        print("- Verify data files are in the data/ folder")
        print("- Check file paths and permissions")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

