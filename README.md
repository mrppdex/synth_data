# Synthetic CDISC SDTM Data Generation

This project aims to generate synthetic CDISC SDTM data using machine learning techniques.

## Features
- **Subject-Level Data**: Uses CTGAN for domains like DM.
- **Longitudinal Data**: Uses Sequential models (PAR) for domains like LB, AE.
- **Preprocessing**: Handles 1:1 column relationships and missing data.
- **GUI**: Streamlit-based interface for easy interaction.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   streamlit run src/gui/app.py
   ```
