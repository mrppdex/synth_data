from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd

class SubjectLevelSynthesizer:
    """
    Wrapper for CTGAN to generate subject-level data (e.g., DM domain).
    """
    def __init__(self, epochs=300):
        self.metadata = SingleTableMetadata()
        self.model = None
        self.epochs = epochs

    def train(self, df: pd.DataFrame):
        """
        Train the CTGAN model.
        """
        self.metadata.detect_from_dataframe(df)
        self.model = CTGANSynthesizer(self.metadata, epochs=self.epochs)
        self.model.fit(df)

    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Generate synthetic data.
        """
        return self.model.sample(num_rows=n_samples)

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = CTGANSynthesizer.load(path)
