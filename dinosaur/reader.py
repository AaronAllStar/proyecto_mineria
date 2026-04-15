import pandas as pd
import os

class Reader:
    """
    Expert dataset reader for the Dinosaur EDA library.
    Capable of reading CSV, Excel, JSON, and more with smart detection.
    """
    
    @staticmethod
    def read(file_path, **kwargs):
        """
        Reads a dataset from a file path with automatic type detection.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {file_path} no existe.")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.csv':
                # Try common delimiters if comma fails (basic smart detection)
                return pd.read_csv(file_path, **kwargs)
            elif ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_path, **kwargs)
            elif ext == '.json':
                return pd.read_json(file_path, **kwargs)
            elif ext == '.parquet':
                return pd.read_parquet(file_path, **kwargs)
            elif ext == '.feather':
                return pd.read_feather(file_path, **kwargs)
            else:
                raise ValueError(f"Extensión de archivo '{ext}' no soportada por Dinosaur.")
        except Exception as e:
            print(f"Error al leer el archivo {file_path}: {e}")
            raise e
