"""
PROTECTIVE FACTOR EXTRACTION
=============================
Drop this file into your project folder alongside extract_data.py

This module extracts protective factors that reduce readmission risk,
helping to reduce systematic false positives.

Based on failure analysis: 56% of FPs are systematic (fixable).
"""

import pandas as pd
import numpy as np


def extract_protective_factors(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Extract protective factors from clinical data.
    
    Args:
        df: DataFrame with admission data
        verbose: Print extraction statistics
        
    Returns:
        DataFrame with protective factor columns
    """
    if verbose:
        print(f"  [PROTECTIVE] Extracting protective factors...")
    
    features = pd.DataFrame(index=df.index)
    n = len(df)
    
    # Get clinical text
    if 'clinical_text' in df.columns:
        text = df['clinical_text'].fillna('').str.lower()
    else:
        text = pd.Series([''] * n, index=df.index)
    
    # 1. SUPERVISED DISCHARGE
    features['pf_supervised_discharge'] = 0
    for col in ['dc_to_snf', 'dc_to_rehab', 'supervised_discharge', 'dc_to_home_health']:
        if col in df.columns:
            features['pf_supervised_discharge'] = np.maximum(
                features['pf_supervised_discharge'],
                df[col].fillna(0).astype(int)
            )
    
    # 2. FOLLOW-UP SCHEDULED
    features['pf_followup_scheduled'] = (
        text.str.contains(r'follow.?up.*(appt|appointment|scheduled|in \d)', regex=True) |
        text.str.contains(r'see.*(pcp|dr\.|clinic).*(week|days)', regex=True) |
        text.str.contains(r'return.*(clinic|office)', regex=True)
    ).astype(int)
    
    # 3. CARE COORDINATION
    features['pf_care_coordination'] = (
        text.str.contains(r'case manag|social work|discharge plan', regex=True) |
        text.str.contains(r'home health|vna|visiting nurse', regex=True) |
        text.str.contains(r'care coordinat', regex=True)
    ).astype(int)
    
    # 4. FAMILY SUPPORT
    support_positive = (
        text.str.contains(r'family.*(bedside|present|support)', regex=True) |
        text.str.contains(r'(spouse|wife|husband|daughter|son).*(at|will|help)', regex=True) |
        text.str.contains(r'lives with|support at home', regex=True)
    )
    support_negative = (
        text.str.contains(r'lives alone|no family|isolated|homeless', regex=True) |
        text.str.contains(r'no support|limited support', regex=True)
    )
    features['pf_family_support'] = (support_positive & ~support_negative).astype(int)
    
    # 5. CLINICALLY STABLE
    features['pf_clinically_stable'] = (
        text.str.contains(r'stable|at baseline|doing well|improved', regex=True) &
        ~text.str.contains(r'unstable|decompensate|worsen', regex=True)
    ).astype(int)
    
    # 6. PATIENT ENGAGED
    features['pf_patient_engaged'] = (
        text.str.contains(r'patient (understand|agree|engaged)', regex=True) |
        text.str.contains(r'educated.*patient|verbalized understanding', regex=True) |
        text.str.contains(r'compliant|adherent', regex=True)
    ).astype(int)
    
    # 7. NO AMA
    features['pf_no_ama'] = 1
    if 'dc_ama' in df.columns:
        features['pf_no_ama'] = (1 - df['dc_ama'].fillna(0)).astype(int)
    
    # 8. AGGREGATE SCORE
    core_cols = [
        'pf_supervised_discharge', 'pf_followup_scheduled', 'pf_care_coordination',
        'pf_family_support', 'pf_clinically_stable', 'pf_patient_engaged'
    ]
    existing = [c for c in core_cols if c in features.columns]
    features['protective_score'] = features[existing].sum(axis=1)
    features['high_protection'] = (features['protective_score'] >= 3).astype(int)
    features['low_protection'] = (features['protective_score'] <= 1).astype(int)
    
    if verbose:
        print(f"    Extracted {len(features.columns)} protective features")
        print(f"    High protection: {features['high_protection'].sum():,} ({features['high_protection'].mean():.1%})")
        print(f"    Low protection: {features['low_protection'].sum():,} ({features['low_protection'].mean():.1%})")
    
    return features