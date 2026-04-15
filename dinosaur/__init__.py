from .reader import Reader
from .analyzer import Analyzer
from .visualizer import Visualizer
from .preprocessor import DinoCleaner

__version__ = "0.1.0"

class Dinosaur:
    """
    Main interface for the Dinosaur library.
    """
    def __init__(self, data=None):
        self.df = None
        if isinstance(data, str):
            self.df = Reader.read(data)
        elif hasattr(data, 'head'): # Check if it's already a dataframe-like object
            self.df = data

    def load(self, file_path, **kwargs):
        self.df = Reader.read(file_path, **kwargs)
        print(f"[Dinosaur] Dataset loaded successfully: {self.df.shape}")
        return self.df

    def report(self):
        if self.df is None:
            print("❌ No data loaded yet. Use load() first.")
            return
        Analyzer.quality_report(self.df)

    def summary(self):
        if self.df is None:
            return None
        return Analyzer.quick_summary(self.df)

    def visualize(self, mode='all'):
        if self.df is None:
            print("❌ No data loaded yet.")
            return
        
        if mode in ['all', 'missing']:
            Visualizer.plot_missing(self.df)
        if mode in ['all', 'distributions']:
            Visualizer.plot_distributions(self.df)
        if mode in ['all', 'correlation']:
            Visualizer.plot_correlation(self.df)

# Factory function for quick start
def explore(file_path):
    dino = Dinosaur(file_path)
    dino.report()
    dino.visualize()
    return dino
