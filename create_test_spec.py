import pandas as pd
import os

def create_dummy_spec():
    if not os.path.exists('specs'):
        os.makedirs('specs')
        
    # ADAE
    df_adae = pd.DataFrame({
        'Variable': ['USUBJID', 'ASTDT', 'AENDT', 'AESEV', 'AETERM', 'AEDECOD'],
        'Label': ['Unique Subject Identifier', 'Analysis Start Date', 'Analysis End Date', 'Severity', 'Reported Term', 'Dictionary-Derived Term'],
        'Type': ['Char', 'Num', 'Num', 'Char', 'Char', 'Char'],
        'Length': [25, 8, 8, 10, 200, 200],
        'Codelist': [None, None, None, 'MILD, MODERATE, SEVERE', None, None],
        'Derivation': ['From DM', 'Derived from AESTDTC', 'Derived from AEENDTC', 'Direct Map', 'Direct Map', 'MedDRA Coding']
    })
    df_adae.to_excel('specs/ADAE.xlsx', index=False)
    
    # ADSL
    df_adsl = pd.DataFrame({
        'Variable': ['USUBJID', 'AGE', 'SEX', 'RACE', 'ARM', 'TRTSDT', 'TRTEDT'],
        'Label': ['Unique Subject Identifier', 'Age', 'Sex', 'Race', 'Treatment Arm', 'Treatment Start Date', 'Treatment End Date'],
        'Type': ['Char', 'Num', 'Char', 'Char', 'Char', 'Num', 'Num'],
        'Length': [25, 8, 1, 50, 50, 8, 8],
        'Codelist': [None, None, 'M, F', 'WHITE, BLACK, ASIAN, OTHER', None, None, None],
        'Derivation': ['From DM', 'From DM', 'From DM', 'From DM', 'From DM', 'From EX', 'From EX']
    })
    df_adsl.to_excel('specs/ADSL.xlsx', index=False)
    
    # ADLB
    df_adlb = pd.DataFrame({
        'Variable': ['USUBJID', 'ADT', 'PARAM', 'AVAL', 'ANRHI', 'ANRLO'],
        'Label': ['Unique Subject Identifier', 'Analysis Date', 'Parameter', 'Analysis Value', 'High Limit', 'Low Limit'],
        'Type': ['Char', 'Num', 'Char', 'Num', 'Num', 'Num'],
        'Length': [25, 8, 50, 8, 8, 8],
        'Codelist': [None, None, 'ALT, AST, BILI', None, None, None],
        'Derivation': ['From LB', 'From LB', 'From LB', 'From LB', 'From LB', 'From LB']
    })
    df_adlb.to_excel('specs/ADLB.xlsx', index=False)
    
    print("Created specs/ADAE.xlsx, ADSL.xlsx, ADLB.xlsx")

if __name__ == "__main__":
    create_dummy_spec()
