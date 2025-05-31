import pandas as pd

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        df = pd.read_csv(self.filepath)
        if 'date' in df.columns:
            # Let pandas infer the format and coerce errors, do NOT force UTC
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df