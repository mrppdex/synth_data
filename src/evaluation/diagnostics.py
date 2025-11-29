from sdv.evaluation.single_table import evaluate_quality, get_column_plot
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def evaluate_data_quality(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata):
    """
    Runs SDV quality report.
    """
    report = evaluate_quality(
        real_data,
        synthetic_data,
        metadata
    )
    return report

def plot_column_distribution(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, column: str, metadata=None):
    """
    Plots distribution comparison for a single column.
    """
    fig = get_column_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        column_name=column,
        metadata=metadata
    )
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap"):
    """
    Plots correlation heatmap for numerical columns.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        return None
        
    corr = numeric_df.corr()
    
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title=title,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1
    )
    return fig
