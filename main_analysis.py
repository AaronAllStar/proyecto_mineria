import dinosaur as dino

# File path
FILE_PATH = "full_dataset_thesis - 1.csv"

def run_analysis():
    print("--- DINOSAUR ANALYSIS ---")
    
    # Initialize Dinosaur with the dataset
    explorer = dino.Dinosaur(FILE_PATH)
    
    # 1. Run the report
    explorer.report()
    
    # 2. Get the head of the dataframe to see what's inside
    if explorer.df is not None:
        print("\nPrueba de los primeros 5 registros:")
        print(explorer.df.head())
        
        # 3. Example of individual visualization
        # explorer.visualize(mode='distributions') # This would open windows
    
if __name__ == "__main__":
    run_analysis()
