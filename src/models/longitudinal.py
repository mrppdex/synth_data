from src.models.transformer import CustomLongitudinalTransformer
import pandas as pd

class LongitudinalSynthesizer:
    """
    Wrapper for CustomLongitudinalTransformer to generate longitudinal data (e.g., LB, AE domains).
    Uses a custom PyTorch Transformer Seq2Seq model.
    """
    def __init__(self, sequence_index: str, entity_columns: list[str], epochs=100, constraints=None):
        self.sequence_index = sequence_index
        self.entity_columns = entity_columns
        self.epochs = epochs
        self.constraints = constraints
        self.model = None

    def train(self, df: pd.DataFrame, progress_callback=None):
        """
        Train the custom transformer model.
        """
        self.model = CustomLongitudinalTransformer(
            sequence_index=self.sequence_index,
            entity_columns=self.entity_columns,
            epochs=self.epochs,
            constraints=self.constraints
        )
        self.model.train(df, progress_callback=progress_callback)

    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Generate synthetic data.
        """
        return self.model.generate(n_samples)

    def save(self, path: str):
        # TODO: Implement save logic for PyTorch model
        pass

    def load(self, path: str):
        # TODO: Implement load logic for PyTorch model
        pass
