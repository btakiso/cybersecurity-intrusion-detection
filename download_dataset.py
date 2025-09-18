"""
Dataset Download Script for Cybersecurity Intrusion Detection Project
Downloads the dataset from Kaggle using kagglehub and places it in the data/ directory

Author: Bereket Takiso
"""

import kagglehub
import shutil
import os

def download_cybersecurity_dataset():
    """Download the cybersecurity intrusion detection dataset from Kaggle"""
    
    print("🔽 Downloading Cybersecurity Intrusion Detection Dataset from Kaggle...")
    print("Dataset: dnkumars/cybersecurity-intrusion-detection-dataset")
    print("="*60)
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("dnkumars/cybersecurity-intrusion-detection-dataset")
        print(f"✅ Dataset downloaded to: {path}")
        
        # Find the CSV file in the downloaded path
        csv_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if csv_files:
            source_file = csv_files[0]  # Take the first CSV file found
            
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Copy the file to our data directory
            destination = 'data/cybersecurity_intrusion_data.csv'
            shutil.copy2(source_file, destination)
            
            print(f"✅ Dataset copied to: {destination}")
            print(f"📊 File size: {os.path.getsize(destination)} bytes")
            
            # Verify the file
            import pandas as pd
            df = pd.read_csv(destination)
            print(f"📈 Dataset shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            
            print("\n🎉 Dataset ready for analysis!")
            print("You can now run the Jupyter notebook or Python scripts.")
            
        else:
            print("❌ No CSV files found in the downloaded dataset")
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have kagglehub installed: pip install kagglehub")
        print("2. Ensure you're authenticated with Kaggle API")
        print("3. Check your internet connection")
        return False
    
    return True

if __name__ == "__main__":
    # Run the download
    success = download_cybersecurity_dataset()
    
    if success:
        print(f"\n📁 Project structure after download:")
        print("cybersecurity-intrusion-detection/")
        print("├── data/")
        print("│   └── cybersecurity_intrusion_data.csv  ✅")
        print("├── notebooks/")
        print("├── src/")
        print("└── ...")
    else:
        print("\n💡 Alternative: Download manually from:")
        print("https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset")
        print("And place the CSV file in the data/ directory")
