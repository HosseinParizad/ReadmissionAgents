"""
CLINICAL AI FAILURE ANALYSIS SYSTEM
====================================

Comprehensive failure analysis for multi-agent readmission prediction models.
Analyzes False Positives and False Negatives to identify systematic errors
and generate actionable insights for model improvement.

USAGE:
------
    # Run analysis on all architectures
    python analyze_failures.py
    
    # Analyze specific architecture
    python analyze_failures.py --arch 0
    
    # Focus on specific error type
    python analyze_failures.py --error-type fp  # or fn
    
    # Generate detailed case reports
    python analyze_failures.py --detailed --top-k 50

OUTPUT:
-------
    ./model_outputs/failure_analysis/
    ├── arch{N}_failure_report.txt       # Detailed failure analysis
    ├── arch{N}_fp_cases.csv             # False positive cases
    ├── arch{N}_fn_cases.csv             # False negative cases
    ├── arch{N}_error_patterns.csv       # Identified error patterns
    └── comparison_failure_analysis.csv  # Cross-architecture comparison
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import project modules
try:
    from config import DATA_PATH, TARGET, RANDOM_STATE
    from compare_all_architectures import (
        load_shared_data, get_specialist_preds, 
        ARCHITECTURE_NAMES, SHARED_DIR
    )
except ImportError as e:
    print(f"⚠️ Warning: Could not import project modules: {e}")
    print("   Make sure config.py and compare_all_architectures.py are in the same directory")


# =============================================================================
# CONFIGURATION
# =============================================================================
ANALYSIS_DIR = './model_outputs/failure_analysis/'
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Clinical thresholds
RISK_THRESHOLDS = {
    'low': 0.35,
    'moderate': 0.50,
    'high': 0.65,
}

SPECIALIST_THRESHOLDS = {
    'low': 0.45,
    'moderate': 0.55,
    'high': 0.65,
}

# Error pattern categories
ERROR_PATTERNS = {
    'fp_supervised_discharge': {
        'name': 'Supervised Discharge Protection',
        'description': 'Patient discharged to SNF/rehab but model predicted readmission',
        'condition': lambda row: row['supervised_discharge'] == 1 and row['y_pred'] == 1 and row['y_true'] == 0
    },
    'fp_protective_factors': {
        'name': 'Strong Protective Factors Missed',
        'description': 'Patient had strong protective factors but model predicted readmission',
        'condition': lambda row: row.get('protective_score', 0) >= 2 and row['y_pred'] == 1 and row['y_true'] == 0
    },
    'fp_overweight_history': {
        'name': 'Prior Readmission Over-Weighting',
        'description': 'Model over-weighted prior readmission history',
        'condition': lambda row: row.get('prev_was_readmitted', 0) == 1 and row['y_pred'] == 1 and row['y_true'] == 0
    },
    'fp_high_medication': {
        'name': 'Medication Count Over-Weighting',
        'description': 'High medication count may have inflated risk',
        'condition': lambda row: row.get('n_medications', 0) >= 40 and row['y_pred'] == 1 and row['y_true'] == 0
    },
    'fn_unpredictable': {
        'name': 'Unpredictable Events',
        'description': 'Readmission due to trauma/accident (unavoidable)',
        'condition': lambda row: row.get('is_unpredictable_service', 0) == 1 and row['y_pred'] == 0 and row['y_true'] == 1
    },
    'fn_low_data': {
        'name': 'Insufficient Clinical Data',
        'description': 'Missing key clinical information',
        'condition': lambda row: (row.get('n_diagnoses', 0) < 5 or row.get('n_medications', 0) < 10) and row['y_pred'] == 0 and row['y_true'] == 1
    },
    'fn_specialist_disagreement': {
        'name': 'Specialist Disagreement',
        'description': 'High variance in specialist opinions',
        'condition': lambda row: row.get('op_std', 0) > 0.25 and row['y_pred'] == 0 and row['y_true'] == 1
    },
    'fn_specialist_disagreement': {
        'name': 'Specialist Disagreement',
        'description': 'High variance in specialist opinions',
        'condition': lambda row: row.get('op_std', 0) > 0.25 and row['y_pred'] == 0 and row['y_true'] == 1
    },
}

# =============================================================================
# EXPLAINABILITY EXTRACTOR
# =============================================================================
class ExplainabilityExtractor:
    """Extracts interpretable features from raw data for failure analysis."""
    
    def __init__(self):
        import re
        self.re = re
        
        # Expanded Keywords
        self.keywords = {
            'ACUTE_EVENT': [
                'unstable', 'critical', 'sepsis', 'septic', 'shock', 'intubated', 
                'respiratory failure', 'cardiac arrest', 'code blue', 'hypotension',
                'hypoxia', 'altered mental status', 'unresponsive', 'coma', 'bleed',
                'hemorrhage', 'stemi', 'nstem', 'infarction', 'stroke', 'cva'
            ],
            'PRIOR_UTILIZATION': [
                'readmission', 'frequent flyer', 'multiple admission', 'recent discharge',
                'return to ed', 'bounce back', 're-presentation', 'non-compliant'
            ],
            'SOCIAL_RISK': [
                'homeless', 'undomiciled', 'shelter', 'substance', 'alcohol', 'etoh',
                'cocaine', 'heroin', 'overdose', 'withdrawal', 'social work',
                'lives alone', 'no support', 'financial', 'insurance', 'lack of transportation'
            ],
            'COMPLEXITY': [
                'multisystem', 'transplant', 'dialysis', 'esrd', 'metastatic', 'cancer',
                'immunosuppressed', 'hospice', 'palliative', 'dnr/dni', 'comfort care',
                'total care', 'bedbound', 'gastrostomy', 'tracheostomy'
            ],
            'PROTECTIVE': [
                'stable', 'improved', 'improving', 'tolerating', 'weaned',
                'ambulatory', 'independent', 'self-care', 'alert and oriented',
                'participating', 'family at bedside', 'involved family', 
                'lives with spouse', 'lives with family', 'home health', 'vna',
                'follow-up arranged', 'appointment scheduled', 'rehabilitation', 
                'snf', 'skilled nursing', 'compliant', 'good understanding'
            ]
        }
        
    def analyze_note(self, text: str) -> Dict[str, any]:
        """Analyze clinical note text for keywords."""
        if not isinstance(text, str) or pd.isna(text):
            return {}
            
        text_lower = text.lower()
        results = {}
        
        # Count keywords
        for category, words in self.keywords.items():
            found = [w for w in words if w in text_lower]
            results[f'kw_{category}_count'] = len(found)
            results[f'kw_{category}_found'] = found[:3] 
            
        return results
        
    def generate_note_explanation(self, score: float, analysis: Dict, case: pd.Series) -> str:
        """Generate text explanation for note score using text and metadata."""
        explanation = []
        
        if score > 0.65:
            # Explain HIGH score
            # 1. Check Keywords (most important - actual text content)
            keyword_explanations = []
            
            if analysis.get('kw_ACUTE_EVENT_count', 0) > 0:
                words = ", ".join(analysis['kw_ACUTE_EVENT_found'][:3])
                keyword_explanations.append(f"acute instability indicators ({words})")
            
            if analysis.get('kw_SOCIAL_RISK_count', 0) > 0:
                words = ", ".join(analysis['kw_SOCIAL_RISK_found'][:3])
                keyword_explanations.append(f"social/behavioral risk factors ({words})")
                
            if analysis.get('kw_PRIOR_UTILIZATION_count', 0) > 0:
                words = ", ".join(analysis['kw_PRIOR_UTILIZATION_found'][:3])
                keyword_explanations.append(f"prior utilization patterns ({words})")

            if analysis.get('kw_COMPLEXITY_count', 0) > 0:
                words = ", ".join(analysis['kw_COMPLEXITY_found'][:3])
                keyword_explanations.append(f"multisystem complexity ({words})")
            
            # If we found keywords, that's the primary explanation
            if keyword_explanations:
                explanation.append(f"Clinical notes document: {'; '.join(keyword_explanations)}")
            else:
                # 2. Fallback: Explain via metadata + what ClinicalBERT likely detected
                context_signals = []
                
                if case.get('n_diagnoses', 0) > 25:
                    context_signals.append(f"{int(case['n_diagnoses'])} diagnoses suggesting complex multimorbidity")
                if case.get('n_medications', 0) > 20:
                    context_signals.append(f"{int(case['n_medications'])} medications indicating polypharmacy")
                if case.get('los_days', 0) > 10:
                    context_signals.append(f"{case['los_days']:.1f}-day stay suggesting complicated course")
                if case.get('charlson_score', 0) > 6:
                    context_signals.append(f"Charlson score {int(case['charlson_score'])} (severe comorbidity burden)")
                if case.get('emergency_admission', 0) == 1:
                    context_signals.append("emergency admission")
                    
                if context_signals:
                    # More specific explanation of what the model detected
                    explanation.append(
                        f"ClinicalBERT embeddings + manual features detected high-risk patterns. "
                        f"Clinical context: {'; '.join(context_signals[:2])}. "
                        f"Likely captured: treatment intensity, care transitions, or latent clinical deterioration markers in note language"
                    )
                else:
                    explanation.append(
                        "ClinicalBERT detected high-risk language patterns in notes "
                        "(e.g., urgent tone, treatment escalation terminology, or coded references to instability) "
                        "not captured by explicit keywords"
                    )
                    
        else:
            # Explain LOW score
            protective_explanations = []
            
            if analysis.get('kw_PROTECTIVE_count', 0) > 0:
                words = ", ".join(analysis['kw_PROTECTIVE_found'][:3])
                protective_explanations.append(f"stability indicators ({words})")
            
            if protective_explanations:
                explanation.append(f"Clinical notes emphasize: {'; '.join(protective_explanations)}")
            else:
                # Explain why low
                if case.get('n_diagnoses', 0) < 10 and case.get('los_days', 0) < 3:
                    explanation.append(
                        f"Note language consistent with uncomplicated course "
                        f"({int(case.get('n_diagnoses', 0))} diagnoses, {case.get('los_days', 0):.1f}-day stay). "
                        f"ClinicalBERT likely detected: routine discharge terminology, absence of complication markers"
                    )
                else:
                    explanation.append(
                        "Despite moderate clinical complexity, note language emphasizes stability. "
                        "ClinicalBERT likely detected: positive progress descriptors, successful treatment markers, "
                        "or planned/controlled discharge language rather than urgent/unplanned terminology"
                    )
                
        return "; ".join(explanation)





class FailureAnalyzer:
    """Comprehensive failure analysis for readmission prediction models."""
    
    def __init__(self, arch_num: int, arch_name: str):
        """Initialize analyzer for specific architecture.
        
        Args:
            arch_num: Architecture number (0-6)
            arch_name: Architecture name (e.g., 'IMSE')
        """
        self.arch_num = arch_num
        self.arch_name = arch_name
        self.output_dir = os.path.join(ANALYSIS_DIR, f'arch{arch_num}_{arch_name}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.df_full = None
        self.fp_cases = None
        self.fn_cases = None
        self.error_patterns = {}
    
    def load_predictions(self, predictions_file: str) -> pd.DataFrame:
        """Load predictions and merge with original text data.
        
        Args:
            predictions_file: Path to predictions CSV
            
        Returns:
            DataFrame with predictions, features, and text
        """
        if not os.path.exists(predictions_file):
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
            
        # 1. Load predictions
        df = pd.read_csv(predictions_file)
        
        # Ensure required columns exist
        required = ['y_true', 'y_pred', 'y_prob']
        if not all(col in df.columns for col in required):
            raise ValueError(f"Missing required columns: {required}")
            
        # 2. Load original data for text
        try:
            print("   Loading original data for text analysis...")
            orig_df = pd.read_csv(DATA_PATH)
            
            # Check for identifiers
            if 'hadm_id' in df.columns and 'hadm_id' in orig_df.columns:
                # Merge on hadm_id
                cols_to_merge = ['hadm_id', 'clinical_text']
                if 'subject_id' in orig_df.columns and 'subject_id' not in df.columns:
                    cols_to_merge.append('subject_id')
                
                print("   Merging predictions with clinical text...")
                df = df.merge(orig_df[cols_to_merge], on='hadm_id', how='left')
                
            else:
                print("   ⚠️ Identifiers (hadm_id) not found in predictions. Skipping text merge.")
                df['clinical_text'] = ""
                
        except Exception as e:
            print(f"   ⚠️ Failed to load original data: {e}")
            df['clinical_text'] = ""
            
        return df
    
    def generate_case_report(self, case: pd.Series, error_type: str) -> str:
        """Generate detailed human-readable report for a single case."""
        report = []
        extractor = ExplainabilityExtractor()
        
        # Analyze text if available
        text_analysis = extractor.analyze_note(case.get('clinical_text', ''))
        
        report.append(f"CASE REPORT: {error_type}")
        report.append("-" * 40)
        report.append(f"Subject ID:  {case.get('subject_id', 'N/A')}")
        report.append(f"HADM ID:     {case.get('hadm_id', 'N/A')}")
        report.append(f"Indices:     Row {case.name}")
        report.append("")
        
        report.append("[MODEL PREDICTION]")
        risk_level = self.classify_risk_level(case['y_prob'])
        report.append(f"   Prediction:  {'READMIT' if case['y_pred']==1 else 'NO READMIT'}")
        report.append(f"   Probability: {case['y_prob']:.2%}  [{risk_level}]")
        report.append(f"   Actual:      {'READMIT' if case['y_true']==1 else 'NO READMIT'}")
        report.append("")
        
        if 'uncertainty_threshold' in case.index:
            lower = case.get('y_lower', case['y_prob'])
            upper = case.get('y_upper', case['y_prob'])
            report.append(f"   Uncertainty: {lower:.2%} - {upper:.2%}")
            report.append("")
            
        report.append("[BRAIN] SPECIALIST OPINIONS:")
        specialists = [
            ('Lab Specialist', case.get('op_lab', 0), 'lab'),
            ('Note Specialist', case.get('op_note', 0), 'note'),
            ('Pharmacy Specialist', case.get('op_pharm', 0), 'pharm'),
            ('History Specialist', case.get('op_hist', 0), 'hist'),
            ('Psychosocial Specialist', case.get('op_psych', 0), 'psych'),
        ]
        
        for name, prob, spec_type in specialists:
            level = self.classify_specialist_opinion(prob)
            explanation = ""
            
            # Add specific explanations
            if spec_type == 'note':
                explanation = extractor.generate_note_explanation(prob, text_analysis)
                if explanation:
                    explanation = f"\n      ↳ WHY: {explanation}"
            
            report.append(f"   {name:25s} {prob:.2%}  [{level}]{explanation}")
        
        # Psychosocial sub-scores
        if 'op_psych_mental' in case.index:
            report.append("")
            report.append("   Psychosocial Sub-Scores:")
            report.append(f"      Mental Health:    {case.get('op_psych_mental', 0):.2%}")
            
            # Explain Social Support
            social_analysis = ""
            if text_analysis.get('social_negative'):
                social_analysis = f" (neg: {', '.join(text_analysis['social_negative'][:2])})"
            elif text_analysis.get('social_positive'):
                social_analysis = f" (pos: {', '.join(text_analysis['social_positive'][:2])})"
                
            report.append(f"      Social Support:   {case.get('op_psych_social', 0):.2%}{social_analysis}")
            report.append(f"      Care Support:     {case.get('op_psych_care', 0):.2%}")

        return "\n".join(report)
    
    def classify_risk_level(self, prob: float) -> str:
        """Classify probability into risk level.
        
        Args:
            prob: Predicted probability
            
        Returns:
            Risk level string
        """
        if prob < RISK_THRESHOLDS['low']:
            return 'LOW'
        elif prob < RISK_THRESHOLDS['moderate']:
            return 'MODERATE'
        elif prob < RISK_THRESHOLDS['high']:
            return 'HIGH'
        else:
            return 'VERY HIGH'
    
    def classify_specialist_opinion(self, prob: float) -> str:
        """Classify specialist opinion.
        
        Args:
            prob: Specialist probability
            
        Returns:
            Opinion level string
        """
        if prob < SPECIALIST_THRESHOLDS['low']:
            return 'LOW'
        elif prob < SPECIALIST_THRESHOLDS['moderate']:
            return 'MODERATE'
        elif prob < SPECIALIST_THRESHOLDS['high']:
            return 'HIGH'
        else:
            return 'VERY HIGH'
    
    def identify_error_patterns(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Identify systematic error patterns.
        
        Args:
            df: DataFrame with predictions and features
            
        Returns:
            Dictionary of error pattern DataFrames
        """
        patterns = {}
        
        for pattern_id, pattern_info in ERROR_PATTERNS.items():
            # Apply condition
            try:
                mask = df.apply(pattern_info['condition'], axis=1)
                pattern_df = df[mask].copy()
                
                if len(pattern_df) > 0:
                    patterns[pattern_id] = {
                        'name': pattern_info['name'],
                        'description': pattern_info['description'],
                        'count': len(pattern_df),
                        'cases': pattern_df
                    }
            except Exception as e:
                print(f"  ⚠️ Error checking pattern {pattern_id}: {e}")
                continue
        
        return patterns
    
    def analyze_fp_cases(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Analyze False Positive cases.
        
        Args:
            df: DataFrame with predictions
            
        Returns:
            Tuple of (fp_dataframe, fp_statistics)
        """
        fp = df[(df['y_pred'] == 1) & (df['y_true'] == 0)].copy()
        
        if len(fp) == 0:
            return pd.DataFrame(), {}
        
        # Add risk classification
        fp['risk_level'] = fp['y_prob'].apply(self.classify_risk_level)
        
        # Compute statistics
        stats = {
            'total': len(fp),
            'rate': len(fp) / len(df),
            'avg_prob': fp['y_prob'].mean(),
            'risk_distribution': fp['risk_level'].value_counts().to_dict(),
        }
        
        # Common characteristics
        if 'supervised_discharge' in fp.columns:
            stats['pct_supervised_discharge'] = fp['supervised_discharge'].mean()
        if 'prev_was_readmitted' in fp.columns:
            stats['pct_prev_readmitted'] = fp['prev_was_readmitted'].mean()
        if 'n_medications' in fp.columns:
            stats['avg_medications'] = fp['n_medications'].mean()
        if 'protective_score' in fp.columns:
            stats['avg_protective_score'] = fp['protective_score'].mean()
        
        return fp, stats
    
    def analyze_fn_cases(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Analyze False Negative cases.
        
        Args:
            df: DataFrame with predictions
            
        Returns:
            Tuple of (fn_dataframe, fn_statistics)
        """
        fn = df[(df['y_pred'] == 0) & (df['y_true'] == 1)].copy()
        
        if len(fn) == 0:
            return pd.DataFrame(), {}
        
        # Add risk classification
        fn['risk_level'] = fn['y_prob'].apply(self.classify_risk_level)
        
        # Compute statistics
        stats = {
            'total': len(fn),
            'rate': len(fn) / len(df),
            'avg_prob': fn['y_prob'].mean(),
            'risk_distribution': fn['risk_level'].value_counts().to_dict(),
        }
        
        # Common characteristics
        if 'is_unpredictable_service' in fn.columns:
            stats['pct_unpredictable'] = fn['is_unpredictable_service'].mean()
        if 'op_std' in fn.columns:
            stats['avg_specialist_disagreement'] = fn['op_std'].mean()
        if 'n_diagnoses' in fn.columns:
            stats['avg_diagnoses'] = fn['n_diagnoses'].mean()
        
        return fn, stats
    
    def generate_case_report(self, case: pd.Series, case_num: int, error_type: str) -> str:
        """Generate detailed report for a single case.
        
        Args:
            case: Series with case data
            case_num: Case number
            error_type: 'FP' or 'FN'
            
        Returns:
            Formatted case report string
        """
        report = []
        extractor = ExplainabilityExtractor()
        
        # Analyze text if available
        text_analysis = extractor.analyze_note(case.get('clinical_text', ''))
        
        report.append("=" * 80)
        report.append(f"CASE #{case_num} - HADM_ID: {case.get('hadm_id', 'UNKNOWN')}")
        report.append(f"ERROR TYPE: {error_type}")
        report.append("=" * 80)
        report.append("")
        
        # Prediction summary
        report.append("[PREDICT] PREDICTION SUMMARY:")
        report.append(f"   Risk Level:        {self.classify_risk_level(case['y_prob'])}")
        report.append(f"   Predicted Prob:    {case['y_prob']:.2%}")
        report.append(f"   Predicted Label:   {'READMITTED' if case['y_pred'] == 1 else 'NOT READMITTED'}")
        report.append(f"   Actual Outcome:    {'READMITTED' if case['y_true'] == 1 else 'NOT READMITTED'}")
        report.append(f"   Prediction:        {'✓ CORRECT' if case['y_pred'] == case['y_true'] else '✗ INCORRECT'}")
        if 'uncertainty_threshold' in case.index:
            lower = case.get('y_lower', case['y_prob'])
            upper = case.get('y_upper', case['y_prob'])
            report.append(f"   Uncertainty Range: {lower:.2%} - {upper:.2%}")
        report.append("")
        
        # Patient context
        report.append("[USER] PATIENT CONTEXT:")
        report.append(f"   Age:                 {case.get('anchor_age', 'N/A')} years")
        report.append(f"   Gender:              {case.get('gender', 'N/A')}")
        report.append(f"   Prior Visits:        {case.get('prior_visits_count', 'N/A')}")
        report.append(f"   Prior Readmissions:  {case.get('n_prior_readmissions', 'N/A')}")
        report.append(f"   ED Visits (6mo):     {case.get('ed_visits_6mo', 'N/A')}")
        report.append(f"   Length of Stay:      {case.get('los_days', 'N/A'):.1f} days")
        report.append(f"   ICU Days:            {case.get('icu_days', 'N/A'):.1f} days")
        report.append(f"   Service:             {case.get('curr_service', 'N/A')}")
        report.append(f"   Charlson Score:      {case.get('charlson_score', 'N/A')}")
        report.append(f"   Emergency Admit:     {'Yes' if case.get('emergency_admit', 0) == 1 else 'No'}")
        report.append(f"   Supervised Disch:    {'Yes' if case.get('supervised_discharge', 0) == 1 else 'No'}")
        report.append("")
        
        # Clinical complexity
        report.append("[MEDICAL] CLINICAL COMPLEXITY:")
        report.append(f"   Medications:         {case.get('n_medications', 'N/A')}")
        report.append(f"   Diagnoses:           {case.get('n_diagnoses', 'N/A')}")
        report.append(f"   Procedures:          {case.get('n_procedures', 'N/A')}")
        if 'charlson_score' in case and 'los_days' in case:
            lace = min(case.get('los_days', 0), 14) + case.get('charlson_score', 0)
            report.append(f"   LACE Score:          {lace:.1f}")
        report.append("")
        
        # Specialist opinions
        if 'op_lab' in case.index:
            report.append("[BRAIN] SPECIALIST OPINIONS:")
            specialists = [
                ('Lab Specialist', case.get('op_lab', 0), 'lab'),
                ('Note Specialist', case.get('op_note', 0), 'note'),
                ('Pharmacy Specialist', case.get('op_pharm', 0), 'pharm'),
                ('History Specialist', case.get('op_hist', 0), 'hist'),
                ('Psychosocial Specialist', case.get('op_psych', 0), 'psych'),
            ]
            
            for name, prob, spec_type in specialists:
                level = self.classify_specialist_opinion(prob)
                explanation = ""
                
                # Add specific explanations
                if spec_type == 'note':
                    explanation = extractor.generate_note_explanation(prob, text_analysis, case)
                    if explanation:
                        explanation = f"\n      ↳ WHY: {explanation}"
                        
                report.append(f"   {name:25s} {prob:.2%}  [{level}]{explanation}")
            
            # Psychosocial sub-scores
            if 'op_psych_mental' in case.index:
                report.append("")
                report.append("   Psychosocial Sub-Scores:")
                report.append(f"      Mental Health:    {case.get('op_psych_mental', 0):.2%}")
                
                # Explain Social Support
                social_analysis = ""
                if text_analysis.get('social_negative'):
                    social_analysis = f" (neg: {', '.join(text_analysis['social_negative'][:2])})"
                elif text_analysis.get('social_positive'):
                    social_analysis = f" (pos: {', '.join(text_analysis['social_positive'][:2])})"
                    
                report.append(f"      Social Support:   {case.get('op_psych_social', 0):.2%}{social_analysis}")
                
                # Add Care Support
                report.append(f"      Care Support:     {case.get('op_psych_care', 0):.2%}")
            
            # Agreement
            if 'op_std' in case.index:
                agreement = "HIGH" if case['op_std'] < 0.15 else "MODERATE" if case['op_std'] < 0.25 else "LOW"
                report.append("")
                report.append(f"   Agent Agreement:     {agreement}")
            
            report.append("")
        
        # Error analysis
        report.append("[WARNING] ERROR ANALYSIS:")
        if error_type == 'FALSE POSITIVE':
            report.append("   FALSE POSITIVE: Model predicted readmission but patient was not readmitted")
            report.append("   Possible reasons:")
            
            # Check protective factors
            protective = []
            if case.get('protective_score', 0) >= 2:
                protective.append("high protective score")
            if case.get('supervised_discharge', 0) == 1:
                protective.append("supervised discharge")
            if case.get('pf_followup_scheduled', 0) == 1:
                protective.append("scheduled followup")
            if case.get('pf_family_support', 0) == 1:
                protective.append("family support")
            
            # Add text-based protective explanations
            if text_analysis.get('kw_PROTECTIVE_count', 0) > 2:
                protective.append("strong stability keywords in notes")
            
            if protective:
                report.append(f"   • Protective factors present: {', '.join(protective)}")
            
            # Check over-weighting
            if case.get('n_medications', 0) >= 40:
                report.append("   • High medication count may have over-weighted risk")
            if case.get('prev_was_readmitted', 0) == 1:
                report.append("   • Model may have over-weighted prior readmission history")
            if case.get('op_std', 0) < 0.1:
                report.append("   • High specialist agreement may have inflated confidence")
            if case.get('charlson_score', 0) >= 5:
                report.append("   • High comorbidity score may have over-estimated risk")
            
        else:  # FALSE NEGATIVE
            report.append("   FALSE NEGATIVE: Model predicted NO readmission but patient WAS readmitted")
            report.append("   Possible reasons:")
            
            # Check unpredictable
            if case.get('is_unpredictable_service', 0) == 1:
                report.append("   • Unpredictable event (trauma/accident) - clinically unavoidable")
            
            # Check data quality
            if case.get('n_diagnoses', 0) < 5:
                report.append("   • Low diagnosis count - insufficient clinical data")
            if case.get('n_medications', 0) < 10:
                report.append("   • Low medication count - may indicate incomplete record")
            
            # Check disagreement
            if case.get('op_std', 0) > 0.25:
                report.append("   • High specialist disagreement - conflicting signals")
            
            # Check low risk indicators
            if case.get('y_prob', 1.0) < 0.15:
                report.append("   • Very low predicted probability - may be edge case")
                
            # Check text indicators for missing risk
            if text_analysis.get('kw_HIGH_RISK_count', 0) > 0:
                report.append(f"   • Missed high-risk keywords in notes: {', '.join(text_analysis['kw_HIGH_RISK_found'])}")
        
        report.append("")
        report.append("=" * 80)
        report.append("")
        
        return "\n".join(report)
    
    def run_analysis(self, df: pd.DataFrame, top_k: int = 25, detailed: bool = False) -> Dict:
        """Run complete failure analysis.
        
        Args:
            df: DataFrame with predictions and features
            top_k: Number of top cases to report
            detailed: Whether to generate detailed case reports
            
        Returns:
            Dictionary with analysis results
        """
        print(f"\n{'='*80}")
        print(f"ANALYZING ARCHITECTURE {self.arch_num}: {self.arch_name}")
        print(f"{'='*80}")
        
        self.df_full = df
        
        # Overall statistics
        total_cases = len(df)
        total_errors = ((df['y_pred'] != df['y_true']).sum())
        
        print(f"\n   Total Cases:    {total_cases:,}")
        print(f"   Total Errors:   {total_errors:,} ({total_errors/total_cases:.1%})")
        
        # Analyze FP and FN
        print(f"\n   Analyzing False Positives...")
        self.fp_cases, fp_stats = self.analyze_fp_cases(df)
        
        print(f"   Analyzing False Negatives...")
        self.fn_cases, fn_stats = self.analyze_fn_cases(df)
        
        print(f"\n   False Positives: {fp_stats.get('total', 0):,} ({fp_stats.get('rate', 0):.1%})")
        print(f"   False Negatives: {fn_stats.get('total', 0):,} ({fn_stats.get('rate', 0):.1%})")
        
        # Identify patterns
        print(f"\n   Identifying error patterns...")
        self.error_patterns = self.identify_error_patterns(df)
        
        for pattern_id, pattern_info in self.error_patterns.items():
            print(f"      {pattern_info['name']}: {pattern_info['count']:,} cases")
        
        # Generate reports
        results = {
            'arch_num': self.arch_num,
            'arch_name': self.arch_name,
            'total_cases': total_cases,
            'total_errors': total_errors,
            'error_rate': total_errors / total_cases,
            'fp_stats': fp_stats,
            'fn_stats': fn_stats,
            'error_patterns': {k: v['count'] for k, v in self.error_patterns.items()},
        }
        
        # Save CSV files
        if len(self.fp_cases) > 0:
            fp_file = os.path.join(self.output_dir, 'fp_cases.csv')
            self.fp_cases.to_csv(fp_file, index=False)
            print(f"\n   ✅ Saved FP cases to: {fp_file}")
        
        if len(self.fn_cases) > 0:
            fn_file = os.path.join(self.output_dir, 'fn_cases.csv')
            self.fn_cases.to_csv(fn_file, index=False)
            print(f"   ✅ Saved FN cases to: {fn_file}")
        
        # Generate detailed report
        if detailed:
            print(f"\n   Generating detailed failure report...")
            report_file = os.path.join(self.output_dir, 'failure_report.txt')
            self._generate_detailed_report(report_file, top_k)
            print(f"   ✅ Saved report to: {report_file}")
        
        return results
    
    def _generate_detailed_report(self, output_file: str, top_k: int):
        """Generate detailed failure report with case studies."""
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*80 + "\n")
            f.write("CLINICAL AI FAILURE ANALYSIS REPORT\n")
            f.write(f"Architecture: {self.arch_name} (#{self.arch_num})\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Summary
            f.write(f"Total Cases: {len(self.df_full):,}\n")
            f.write(f"False Positives: {len(self.fp_cases):,} | False Negatives: {len(self.fn_cases):,}\n")
            f.write("="*80 + "\n\n")

            # Methodology
            f.write("METHODOLOGY NOTE:\n")
            f.write("-" * 80 + "\n")
            f.write("Specialist scores are derived from multi-modal analysis:\n")
            f.write("   • Note Specialist:  Ensemble of ClinicalBERT Embeddings + LDA Topics + Manual Features (Structure, Risk Patterns)\n")
            f.write("                        processed by HistGradientBoostingClassifier.\n")
            f.write("   • Lab Specialist:   Dynamic Lab Trajectories (RNN/Transformer)\n")
            f.write("   • High 'Why' explanations are heuristic interpretations of the black-box score.\n")
            f.write("=" * 80 + "\n\n")
            
            # Error Patterns
            f.write("IDENTIFIED ERROR PATTERNS:\n")
            f.write("-"*80 + "\n")
            for pattern_id, pattern_info in self.error_patterns.items():
                f.write(f"\n{pattern_info['name']} ({pattern_info['count']} cases)\n")
                f.write(f"   {pattern_info['description']}\n")
            f.write("\n" + "="*80 + "\n\n")
            
            # Top FP cases
            if len(self.fp_cases) > 0:
                f.write("="*80 + "\n")
                f.write("TOP FALSE POSITIVE CASES\n")
                f.write("="*80 + "\n\n")
                
                # Sort by probability (highest false alarms)
                top_fp = self.fp_cases.nlargest(top_k, 'y_prob')
                
                for idx, (_, case) in enumerate(top_fp.iterrows(), 1):
                    report = self.generate_case_report(case, idx, 'FALSE POSITIVE')
                    f.write(report)
            
            # Top FN cases
            if len(self.fn_cases) > 0:
                f.write("\n" + "="*80 + "\n")
                f.write("TOP FALSE NEGATIVE CASES\n")
                f.write("="*80 + "\n\n")
                
                # Sort by probability (lowest misses)
                top_fn = self.fn_cases.nsmallest(top_k, 'y_prob')
                
                for idx, (_, case) in enumerate(top_fn.iterrows(), 1):
                    report = self.generate_case_report(case, idx, 'FALSE NEGATIVE')
                    f.write(report)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Analyze failures in multi-agent readmission prediction'
    )
    parser.add_argument(
        '--arch', type=int, default=None,
        help='Architecture number to analyze (default: all)'
    )
    parser.add_argument(
        '--error-type', type=str, choices=['fp', 'fn', 'both'], default='both',
        help='Error type to analyze'
    )
    parser.add_argument(
        '--detailed', action='store_true',
        help='Generate detailed case reports'
    )
    parser.add_argument(
        '--top-k', type=int, default=25,
        help='Number of top cases to report'
    )
    parser.add_argument(
        '--predictions-dir', type=str, default='./model_outputs/',
        help='Directory containing architecture predictions'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CLINICAL AI FAILURE ANALYSIS SYSTEM")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine which architectures to analyze
    if args.arch is not None:
        arch_nums = [args.arch]
    else:
        # Find all available architecture predictions
        arch_nums = []
        for arch_num in range(7):  # 0-6
            arch_name = ARCHITECTURE_NAMES.get(arch_num, {}).get('abbrev', f'arch{arch_num}')
            pred_file = os.path.join(ANALYSIS_DIR, f'arch{arch_num}_{arch_name}', 'predictions_with_features.csv')
            if os.path.exists(pred_file):
                arch_nums.append(arch_num)
        
        if not arch_nums:
            print("\n⚠️  No prediction files found!")
            print(f"   Looking in: {ANALYSIS_DIR}")
            print("\n   Run compare_all_architectures.py first to generate predictions.")
            print("   Example: python compare_all_architectures.py --arch 0 --skip-specialist --debug")
            print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            return
    
    print(f"\nAnalyzing architectures: {arch_nums}")
    
    # Run analysis for each architecture
    all_analysis_results = []
    
    for arch_num in arch_nums:
        arch_name = ARCHITECTURE_NAMES.get(arch_num, {}).get('abbrev', f'arch{arch_num}')
        arch_full = ARCHITECTURE_NAMES.get(arch_num, {}).get('full', f'Architecture {arch_num}')
        
        # Load predictions
        pred_file = os.path.join(ANALYSIS_DIR, f'arch{arch_num}_{arch_name}', 'predictions_with_features.csv')
        
        if not os.path.exists(pred_file):
            print(f"\n⚠️  Skipping Architecture {arch_num}: predictions file not found")
            print(f"   Expected: {pred_file}")
            continue
        
        # Create analyzer and run analysis
        analyzer = FailureAnalyzer(arch_num, arch_name)
        
        try:
            df = analyzer.load_predictions(pred_file)
            results = analyzer.run_analysis(df, top_k=args.top_k, detailed=args.detailed)
            all_analysis_results.append(results)
            
        except Exception as e:
            print(f"\n❌ Error analyzing Architecture {arch_num}: {e}")
            if args.detailed:
                import traceback
                traceback.print_exc()
            continue
    
    # Generate comparison report if multiple architectures
    if len(all_analysis_results) > 1:
        print(f"\n{'='*80}")
        print("CROSS-ARCHITECTURE FAILURE COMPARISON")
        print(f"{'='*80}\n")
        
        comparison_data = []
        for res in all_analysis_results:
            comparison_data.append({
                'Architecture': f"{res['arch_name']} (#{res['arch_num']})",
                'Total Cases': res['total_cases'],
                'Total Errors': res['total_errors'],
                'Error Rate': f"{res['error_rate']:.1%}",
                'FP Count': res['fp_stats'].get('total', 0),
                'FP Rate': f"{res['fp_stats'].get('rate', 0):.1%}",
                'FN Count': res['fn_stats'].get('total', 0),
                'FN Rate': f"{res['fn_stats'].get('rate', 0):.1%}",
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Save comparison
        comparison_file = os.path.join(ANALYSIS_DIR, 'comparison_failure_analysis.csv')
        df_comparison.to_csv(comparison_file, index=False)
        print(f"\n✅ Comparison saved to: {comparison_file}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()