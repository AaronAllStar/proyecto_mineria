import dinosaur as dino
import os

# Create a demo script
dataset_path = "full_dataset_thesis - 1.csv"

if os.path.exists(dataset_path):
    print(f"Testing Dinosaur with {dataset_path}...")
    
    # Using the factory function
    explorer = dino.Dinosaur(dataset_path)
    
    # Show report
    explorer.report()
    
    # Get summary dictionary
    summary = explorer.summary()
    print("\nSummary dictionary keys:", summary.keys())
    
    print("\n[Dinosaur] Is ready for action!")
else:
    print(f"❌ Dataset {dataset_path} not found for testing.")
