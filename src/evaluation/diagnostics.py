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

import plotly.express as px
import plotly.graph_objects as go

def plot_column_distribution(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, column: str, metadata=None):
    """
    Plots distribution comparison for a single column with custom aesthetics.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Check if column exists
    if column not in real_data.columns or column not in synthetic_data.columns:
        return go.Figure()

    # Determine type
    is_categorical = real_data[column].dtype == 'object' or real_data[column].dtype.name == 'category'
    
    fig = go.Figure()
    
    if is_categorical:
        # Categorical: Bar Chart
        real_counts = real_data[column].value_counts(normalize=True).sort_index()
        synth_counts = synthetic_data[column].value_counts(normalize=True).sort_index()
        
        # Align indices
        all_cats = sorted(list(set(real_counts.index) | set(synth_counts.index)))
        real_vals = [real_counts.get(c, 0) for c in all_cats]
        synth_vals = [synth_counts.get(c, 0) for c in all_cats]
        
        fig.add_trace(go.Bar(
            x=all_cats, 
            y=real_vals, 
            name='Real',
            marker_color='#1f77b4', # Blue
            marker_pattern_shape='\\' 
        ))
        fig.add_trace(go.Bar(
            x=all_cats, 
            y=synth_vals, 
            name='Synthetic',
            marker_color='#ff7f0e', # Orange
            marker_pattern_shape='.'
        ))
        fig.update_layout(barmode='group', title=f"Distribution of {column}")
        
    else:
        # Numerical: Histogram/KDE
        fig.add_trace(go.Histogram(
            x=real_data[column],
            name='Real',
            marker_color='#1f77b4',
            opacity=0.7,
            histnorm='probability density'
        ))
        fig.add_trace(go.Histogram(
            x=synthetic_data[column],
            name='Synthetic',
            marker_color='#ff7f0e',
            opacity=0.7,
            histnorm='probability density'
        ))
        fig.update_layout(barmode='overlay', title=f"Distribution of {column}")
        
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap"):
    """
    Plots correlation heatmap for numerical columns.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Drop columns that are all NA
    numeric_df = numeric_df.dropna(axis=1, how='all')
    
    # Drop columns with 0 variance (constant values) as they produce NaN correlation
    numeric_df = numeric_df.loc[:, numeric_df.std() > 0]
    
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
