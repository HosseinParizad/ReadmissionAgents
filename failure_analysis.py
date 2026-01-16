"""
CLINICAL AI FAILURE ANALYSIS SYSTEM - ENHANCED WITH FULL EXPLAINABILITY
========================================================================

Comprehensive failure analysis for multi-agent readmission prediction models.
Now includes detailed "WHY" explanations for ALL specialists.

USAGE:
------
    python analyze_failures.py --detailed --top-k 50
    python analyze_failures.py --arch 0 --detailed
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from config import DATA_PATH, TARGET, RANDOM_STATE
    from compare_all_architectures import ARCHITECTURE_NAMES, SHARED_DIR
except ImportError as e:
    print(f"⚠️ Warning: Could not import project modules: {e}")

ANALYSIS_DIR = './model_outputs/failure_analysis/'
os.makedirs(ANALYSIS_DIR, exist_ok=True)

RISK_THRESHOLDS = {'low': 0.35, 'moderate': 0.50, 'high': 0.65}
SPECIALIST_THRESHOLDS = {'low': 0.45, 'moderate': 0.55, 'high': 0.65}


# =============================================================================
# SPECIALIST EXPLAINER - ALL SPECIALISTS
# =============================================================================
class SpecialistExplainer:
    """Generates human-readable explanations for ALL specialist predictions."""
    
    def __init__(self):
        import re
        self.re = re
        
        self.note_keywords = {
            'ACUTE_EVENT': ['unstable', 'critical', 'sepsis', 'shock', 'intubated', 
                'respiratory failure', 'cardiac arrest', 'hypotension', 'hypoxia', 
                'altered mental status', 'hemorrhage', 'stroke'],
            'SOCIAL_RISK': ['homeless', 'shelter', 'substance', 'alcohol', 'withdrawal', 
                'lives alone', 'no support', 'financial'],
            'COMPLEXITY': ['multisystem', 'transplant', 'dialysis', 'esrd', 'metastatic', 
                'cancer', 'immunosuppressed', 'palliative'],
            'PROTECTIVE': ['stable', 'improved', 'improving', 'tolerating', 'ambulatory', 
                'family support', 'home health', 'snf', 'skilled nursing', 'compliant']
        }
    
    def analyze_note(self, text: str) -> Dict:
        """Analyze clinical note for keywords."""
        if not isinstance(text, str) or pd.isna(text):
            return {}
        
        text_lower = text.lower()
        results = {}
        
        for category, words in self.note_keywords.items():
            found = [w for w in words if w in text_lower]
            results[f'kw_{category}_count'] = len(found)
            results[f'kw_{category}_found'] = found[:3]
        
        return results
    
    def explain_lab_specialist(self, score: float, case: pd.Series) -> str:
        """Explain Lab Specialist prediction."""
        explanations = []
        
        if score > 0.65:  # HIGH RISK
            lab_issues = []
            
            creat = case.get('creatinine_mean', 0)
            if creat > 3.0:
                lab_issues.append(f"severe kidney dysfunction (Cr {creat:.1f} mg/dL, normal <1.2)")
            elif creat > 1.5:
                lab_issues.append(f"kidney impairment (Cr {creat:.1f} mg/dL)")
            
            hgb = case.get('hemoglobin_mean', 0)
            if 0 < hgb < 7:
                lab_issues.append(f"severe anemia (Hgb {hgb:.1f} g/dL, normal >12)")
            elif 0 < hgb < 10:
                lab_issues.append(f"anemia (Hgb {hgb:.1f} g/dL)")
            
            lactate = case.get('lactate_mean', 0)
            if lactate > 4.0:
                lab_issues.append(f"severe lactic acidosis (lactate {lactate:.1f}, suggests sepsis/shock)")
            elif lactate > 2.0:
                lab_issues.append(f"elevated lactate ({lactate:.1f}, tissue hypoperfusion)")
            
            wbc = case.get('wbc_mean', 0)
            if wbc > 20:
                lab_issues.append(f"marked leukocytosis (WBC {wbc:.1f}K, suggests infection)")
            elif 0 < wbc < 4:
                lab_issues.append(f"leukopenia (WBC {wbc:.1f}K, immunosuppression)")
            
            bili = case.get('bilirubin_mean', 0)
            if bili > 3.0:
                lab_issues.append(f"severe hyperbilirubinemia (T.Bili {bili:.1f} mg/dL)")
            
            inr = case.get('inr_mean', 0)
            if inr > 2.0:
                lab_issues.append(f"coagulopathy (INR {inr:.1f}, bleeding risk)")
            
            if lab_issues:
                explanations.append(f"Critical lab abnormalities: {'; '.join(lab_issues[:3])}")
            else:
                explanations.append("Borderline abnormal labs with concerning trends")
        else:  # LOW RISK
            normal_labs = []
            creat = case.get('creatinine_mean', 0)
            if 0 < creat <= 1.2:
                normal_labs.append("normal kidney function")
            hgb = case.get('hemoglobin_mean', 0)
            if hgb >= 12:
                normal_labs.append("normal hemoglobin")
            
            if normal_labs:
                explanations.append(f"Labs reassuring: {', '.join(normal_labs)}")
            else:
                explanations.append("No critical lab abnormalities")
        
        return "; ".join(explanations) if explanations else "Lab patterns analyzed"
    
    def explain_note_specialist(self, score: float, text_analysis: Dict, case: pd.Series) -> str:
        """Explain Note Specialist prediction."""
        explanations = []
        
        if score > 0.65:  # HIGH RISK
            kw_exp = []
            
            if text_analysis.get('kw_ACUTE_EVENT_count', 0) > 0:
                words = ", ".join(text_analysis['kw_ACUTE_EVENT_found'][:3])
                kw_exp.append(f"acute instability ({words})")
            
            if text_analysis.get('kw_SOCIAL_RISK_count', 0) > 0:
                words = ", ".join(text_analysis['kw_SOCIAL_RISK_found'][:3])
                kw_exp.append(f"social barriers ({words})")
            
            if text_analysis.get('kw_COMPLEXITY_count', 0) > 0:
                words = ", ".join(text_analysis['kw_COMPLEXITY_found'][:3])
                kw_exp.append(f"complex care ({words})")
            
            if kw_exp:
                explanations.append(f"Notes document: {'; '.join(kw_exp)}")
            else:
                context = []
                if case.get('n_diagnoses', 0) > 25:
                    context.append(f"{int(case['n_diagnoses'])} diagnoses")
                if case.get('n_medications', 0) > 20:
                    context.append(f"{int(case['n_medications'])} medications")
                if case.get('los_days', 0) > 10:
                    context.append(f"{case['los_days']:.0f}-day stay")
                
                if context:
                    explanations.append(
                        f"ClinicalBERT detected high-risk patterns. Context: {', '.join(context[:2])}. "
                        f"Likely: treatment escalation, instability markers"
                    )
                else:
                    explanations.append("High-risk language patterns (urgent tone, complications)")
        else:  # LOW RISK
            if text_analysis.get('kw_PROTECTIVE_count', 0) > 0:
                words = ", ".join(text_analysis['kw_PROTECTIVE_found'][:3])
                explanations.append(f"Stability emphasized: {words}")
            else:
                if case.get('los_days', 0) < 3:
                    explanations.append(f"Uncomplicated {case.get('los_days', 0):.0f}-day stay")
                else:
                    explanations.append("Stable course, positive progress descriptors")
        
        return "; ".join(explanations)
    
    def explain_pharmacy_specialist(self, score: float, med_text: str, case: pd.Series) -> str:
        """Explain Pharmacy Specialist prediction."""
        explanations = []
        
        if not isinstance(med_text, str):
            med_text = ""
        
        med_lower = med_text.lower()
        n_meds = case.get('n_medications', len(med_text.split(',')))
        
        if score > 0.65:  # HIGH RISK
            risk_meds = []
            
            high_risk = {
                'anticoagulants': ['warfarin', 'heparin', 'enoxaparin', 'rivaroxaban'],
                'insulin': ['insulin', 'lantus', 'humalog'],
                'opioids': ['morphine', 'oxycodone', 'fentanyl', 'dilaudid'],
                'cardiac': ['digoxin', 'amiodarone']
            }
            
            for category, meds in high_risk.items():
                found = [m for m in meds if m in med_lower]
                if found:
                    risk_meds.append(f"{category} ({', '.join(found[:2])})")
            
            if risk_meds:
                explanations.append(f"High-risk medications: {'; '.join(risk_meds[:3])}")
            
            if n_meds >= 20:
                explanations.append(f"Extreme polypharmacy ({n_meds} meds, interaction/adherence risk)")
            elif n_meds >= 15:
                explanations.append(f"Polypharmacy ({n_meds} medications)")
            
            if not explanations:
                explanations.append("Complex medication regimen with multiple risk factors")
        else:  # LOW RISK
            if n_meds < 5:
                explanations.append(f"Simple regimen ({n_meds} medications)")
            elif n_meds < 10:
                explanations.append(f"Manageable burden ({n_meds} meds, no high-risk agents)")
            else:
                explanations.append("Moderate med count, no high-risk combinations")
        
        return "; ".join(explanations) if explanations else "Medication profile analyzed"
    
    def explain_history_specialist(self, score: float, history_text: str, case: pd.Series) -> str:
        """Explain History Specialist prediction."""
        explanations = []
        
        if not isinstance(history_text, str):
            history_text = ""
        
        history_lower = history_text.lower()
        
        if score > 0.65:  # HIGH RISK
            conditions = []
            
            if 'heart failure' in history_lower or 'chf' in history_lower:
                conditions.append("heart failure")
            if 'copd' in history_lower:
                conditions.append("COPD")
            if 'diabetes' in history_lower:
                conditions.append("diabetes")
            if 'dialysis' in history_lower or 'esrd' in history_lower:
                conditions.append("end-stage renal disease")
            if 'cirrhosis' in history_lower:
                conditions.append("liver disease")
            if 'cancer' in history_lower or 'metastatic' in history_lower:
                conditions.append("malignancy")
            
            if conditions:
                explanations.append(
                    f"High-risk chronic conditions ({len(conditions)}): {', '.join(conditions[:3])}"
                )
            
            if 'readmit' in history_lower or 'prior admission' in history_lower:
                explanations.append("Prior admissions/readmissions documented")
            
            if not explanations:
                charlson = case.get('charlson_score', 0)
                if charlson >= 5:
                    explanations.append(f"Heavy comorbidity burden (Charlson {charlson})")
                else:
                    explanations.append("Complex medical history")
        else:  # LOW RISK
            if len(history_text) < 100:
                explanations.append("Limited significant medical history")
            else:
                charlson = case.get('charlson_score', 0)
                if charlson <= 2:
                    explanations.append(f"Low comorbidity burden (Charlson {charlson})")
                else:
                    explanations.append("Stable chronic conditions, well-controlled")
        
        return "; ".join(explanations) if explanations else "Medical history assessed"
    
    def explain_psychosocial_specialist(self, score: float, text: str, case: pd.Series) -> str:
        """Explain Psychosocial Specialist prediction."""
        explanations = []
        
        if not isinstance(text, str):
            text = ""
        
        text_lower = text.lower()
        
        if score > 0.65:  # HIGH RISK
            risks = []
            
            if any(w in text_lower for w in ['depression', 'anxiety', 'psychiatric', 'suicid']):
                risks.append("mental health concerns")
            
            if any(w in text_lower for w in ['substance', 'alcohol', 'drug abuse', 'withdrawal']):
                risks.append("substance use disorder")
            
            if 'homeless' in text_lower or 'shelter' in text_lower:
                risks.append("housing instability")
            elif 'lives alone' in text_lower or 'no support' in text_lower:
                risks.append("limited social support")
            
            if 'non-compliant' in text_lower or 'non-adherent' in text_lower:
                risks.append("medication non-adherence")
            
            if risks:
                explanations.append(f"Psychosocial barriers: {', '.join(risks[:3])}")
            else:
                explanations.append("Psychosocial risk factors (care access, support concerns)")
        else:  # LOW RISK
            protective = []
            
            if 'family support' in text_lower or 'caregiver' in text_lower:
                protective.append("strong family support")
            if 'follow-up scheduled' in text_lower:
                protective.append("care continuity arranged")
            if 'snf' in text_lower or 'rehab' in text_lower or 'home health' in text_lower:
                protective.append("transitional care services")
            
            if protective:
                explanations.append(f"Protective factors: {', '.join(protective)}")
            else:
                explanations.append("No major psychosocial barriers")
        
        return "; ".join(explanations) if explanations else "Psychosocial factors reviewed"


# =============================================================================
# FAILURE ANALYZER
# =============================================================================
class FailureAnalyzer:
    """Comprehensive failure analysis with specialist explanations."""
    
    def __init__(self, arch_num: int, arch_name: str):
        self.arch_num = arch_num
        self.arch_name = arch_name
        self.output_dir = os.path.join(ANALYSIS_DIR, f'arch{arch_num}_{arch_name}')
        os.makedirs(self.output_dir, exist_ok=True)
        self.explainer = SpecialistExplainer()
    
    def load_predictions(self, predictions_file: str) -> pd.DataFrame:
        """Load predictions and merge with text data."""
        df = pd.read_csv(predictions_file)
        
        try:
            print("   Loading original data for text analysis...")
            orig_df = pd.read_csv(DATA_PATH)
            
            if 'hadm_id' in df.columns and 'hadm_id' in orig_df.columns:
                text_cols = ['hadm_id', 'clinical_text']
                if 'med_list_text' in orig_df.columns:
                    text_cols.append('med_list_text')
                if 'full_history_text' in orig_df.columns:
                    text_cols.append('full_history_text')
                
                df = df.merge(orig_df[text_cols], on='hadm_id', how='left')
            else:
                df['clinical_text'] = ""
                df['med_list_text'] = ""
                df['full_history_text'] = ""
        except Exception as e:
            print(f"   ⚠️ Failed to load text: {e}")
            df['clinical_text'] = ""
            df['med_list_text'] = ""
            df['full_history_text'] = ""
        
        return df
    
    def classify_risk_level(self, prob: float) -> str:
        if prob < RISK_THRESHOLDS['low']:
            return 'LOW'
        elif prob < RISK_THRESHOLDS['moderate']:
            return 'MODERATE'
        elif prob < RISK_THRESHOLDS['high']:
            return 'HIGH'
        else:
            return 'VERY HIGH'
    
    def classify_specialist_opinion(self, prob: float) -> str:
        if prob < SPECIALIST_THRESHOLDS['low']:
            return 'LOW'
        elif prob < SPECIALIST_THRESHOLDS['moderate']:
            return 'MODERATE'
        elif prob < SPECIALIST_THRESHOLDS['high']:
            return 'HIGH'
        else:
            return 'VERY HIGH'
    
    def generate_case_report(self, case: pd.Series, case_num: int, error_type: str) -> str:
        """Generate detailed case report with specialist explanations."""
        report = []
        
        clinical_text = case.get('clinical_text', '')
        med_text = case.get('med_list_text', '')
        history_text = case.get('full_history_text', '')
        text_analysis = self.explainer.analyze_note(clinical_text)
        
        report.append("=" * 80)
        report.append(f"CASE #{case_num} - HADM_ID: {case.get('hadm_id', 'UNKNOWN')}")
        report.append(f"ERROR TYPE: {error_type}")
        report.append("=" * 80)
        report.append("")
        
        # Prediction
        report.append("[PREDICTION SUMMARY]")
        report.append(f"   Risk Level:        {self.classify_risk_level(case['y_prob'])}")
        report.append(f"   Predicted Prob:    {case['y_prob']:.2%}")
        report.append(f"   Predicted:         {'READMITTED' if case['y_pred'] == 1 else 'NOT READMITTED'}")
        report.append(f"   Actual:            {'READMITTED' if case['y_true'] == 1 else 'NOT READMITTED'}")
        report.append(f"   Result:            {'✗ INCORRECT' if case['y_pred'] != case['y_true'] else '✓ CORRECT'}")
        report.append("")
        
        # Patient context
        report.append("[PATIENT CONTEXT]")
        report.append(f"   Age:               {case.get('anchor_age', 'N/A')} years")
        report.append(f"   Gender:            {case.get('gender', 'N/A')}")
        report.append(f"   Length of Stay:    {case.get('los_days', 'N/A'):.1f} days")
        report.append(f"   ICU Days:          {case.get('icu_days', 0):.1f} days")
        report.append(f"   Charlson Score:    {case.get('charlson_score', 'N/A')}")
        report.append(f"   Medications:       {case.get('n_medications', 'N/A')}")
        report.append(f"   Diagnoses:         {case.get('n_diagnoses', 'N/A')}")
        report.append("")
        
        # Specialists WITH explanations
        report.append("[SPECIALIST OPINIONS - WITH EXPLANATIONS]")
        report.append("")
        
        specialists = [
            ('Lab Specialist', 'op_lab', 'lab'),
            ('Note Specialist', 'op_note', 'note'),
            ('Pharmacy Specialist', 'op_pharm', 'pharm'),
            ('History Specialist', 'op_hist', 'hist'),
            ('Psychosocial Specialist', 'op_psych', 'psych'),
        ]
        
        for name, col, spec_type in specialists:
            if col not in case.index:
                continue
            
            prob = case[col]
            level = self.classify_specialist_opinion(prob)
            report.append(f"   {name:30s} {prob:.2%}  [{level}]")
            
            # Get explanation
            explanation = ""
            if spec_type == 'lab':
                explanation = self.explainer.explain_lab_specialist(prob, case)
            elif spec_type == 'note':
                explanation = self.explainer.explain_note_specialist(prob, text_analysis, case)
            elif spec_type == 'pharm':
                explanation = self.explainer.explain_pharmacy_specialist(prob, med_text, case)
            elif spec_type == 'hist':
                explanation = self.explainer.explain_history_specialist(prob, history_text, case)
            elif spec_type == 'psych':
                explanation = self.explainer.explain_psychosocial_specialist(prob, clinical_text, case)
            
            if explanation:
                # Word wrap at 76 chars
                words = explanation.split()
                lines = []
                current = "      ↳ WHY: "
                
                for word in words:
                    if len(current) + len(word) + 1 > 76:
                        lines.append(current)
                        current = "              " + word
                    else:
                        current += (" " + word) if not current.endswith("WHY: ") else word
                
                if current.strip():
                    lines.append(current)
                
                report.extend(lines)
            
            report.append("")
        
        # Psychosocial sub-scores
        if 'op_psych_mental' in case.index:
            report.append("   Psychosocial Sub-Scores:")
            report.append(f"      Mental Health:    {case.get('op_psych_mental', 0):.2%}")
            report.append(f"      Social Support:   {case.get('op_psych_social', 0):.2%}")
            report.append(f"      Care Access:      {case.get('op_psych_care', 0):.2%}")
            report.append("")
        
        # Agreement
        if 'op_std' in case.index:
            std = case['op_std']
            agreement = "HIGH" if std < 0.15 else "MODERATE" if std < 0.25 else "LOW"
            report.append(f"   Specialist Agreement:  {agreement} (std={std:.3f})")
            report.append("")
        
        # Error analysis
        report.append("[ERROR ANALYSIS]")
        if error_type == 'FALSE POSITIVE':
            report.append("   Model predicted readmission but patient was NOT readmitted")
            report.append("")
            report.append("   Likely Contributing Factors:")
            
            if case.get('supervised_discharge', 0) == 1:
                report.append("   • Patient discharged to SNF/rehab (protective)")
            
            if text_analysis.get('kw_PROTECTIVE_count', 0) > 2:
                words = ', '.join(text_analysis['kw_PROTECTIVE_found'][:3])
                report.append(f"   • Stability indicators in notes: {words}")
            
            if case.get('n_medications', 0) >= 30:
                report.append(f"   • High medication count ({case['n_medications']}) may have inflated risk")
            
            if case.get('charlson_score', 0) >= 6:
                report.append(f"   • High Charlson ({case['charlson_score']}) but stable conditions")
        
        else:  # FALSE NEGATIVE
            report.append("   Model predicted NO readmission but patient WAS readmitted")
            report.append("")
            report.append("   Likely Contributing Factors:")
            
            if case.get('n_diagnoses', 0) < 5:
                report.append("   • Low diagnosis count - insufficient clinical data")
            
            if case.get('op_std', 0) > 0.25:
                report.append("   • High specialist disagreement - conflicting signals")
            
            if case.get('y_prob', 1.0) < 0.15:
                report.append("   • Very low predicted probability - edge case")
        
        report.append("")
        report.append("=" * 80)
        report.append("")
        
        return "\n".join(report)
    
    def analyze_fp_cases(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Analyze False Positive cases."""
        fp = df[(df['y_pred'] == 1) & (df['y_true'] == 0)].copy()
        if len(fp) == 0:
            return pd.DataFrame(), {}
        
        fp['risk_level'] = fp['y_prob'].apply(self.classify_risk_level)
        stats = {
            'total': len(fp),
            'rate': len(fp) / len(df),
            'avg_prob': fp['y_prob'].mean(),
        }
        return fp, stats
    
    def analyze_fn_cases(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Analyze False Negative cases."""
        fn = df[(df['y_pred'] == 0) & (df['y_true'] == 1)].copy()
        if len(fn) == 0:
            return pd.DataFrame(), {}
        
        fn['risk_level'] = fn['y_prob'].apply(self.classify_risk_level)
        stats = {
            'total': len(fn),
            'rate': len(fn) / len(df),
            'avg_prob': fn['y_prob'].mean(),
        }
        return fn, stats
    
    def run_analysis(self, df: pd.DataFrame, top_k: int = 25, detailed: bool = False) -> Dict:
        """Run complete failure analysis."""
        print(f"\n{'='*80}")
        print(f"ANALYZING ARCHITECTURE {self.arch_num}: {self.arch_name}")
        print(f"{'='*80}")
        
        total_cases = len(df)
        total_errors = (df['y_pred'] != df['y_true']).sum()
        
        print(f"\n   Total Cases:    {total_cases:,}")
        print(f"   Total Errors:   {total_errors:,} ({total_errors/total_cases:.1%})")
        
        fp_cases, fp_stats = self.analyze_fp_cases(df)
        fn_cases, fn_stats = self.analyze_fn_cases(df)
        
        print(f"\n   False Positives: {fp_stats.get('total', 0):,} ({fp_stats.get('rate', 0):.1%})")
        print(f"   False Negatives: {fn_stats.get('total', 0):,} ({fn_stats.get('rate', 0):.1%})")
        
        # Save CSVs
        if len(fp_cases) > 0:
            fp_file = os.path.join(self.output_dir, 'fp_cases.csv')
            fp_cases.to_csv(fp_file, index=False)
            print(f"\n   ✅ Saved FP cases: {fp_file}")
        
        if len(fn_cases) > 0:
            fn_file = os.path.join(self.output_dir, 'fn_cases.csv')
            fn_cases.to_csv(fn_file, index=False)
            print(f"   ✅ Saved FN cases: {fn_file}")
        
        # Generate detailed report
        if detailed:
            print(f"\n   Generating detailed failure report...")
            report_file = os.path.join(self.output_dir, 'failure_report.txt')
            self._generate_detailed_report(report_file, top_k, fp_cases, fn_cases)
            print(f"   ✅ Saved report: {report_file}")
        
        return {
            'arch_num': self.arch_num,
            'arch_name': self.arch_name,
            'total_cases': total_cases,
            'total_errors': total_errors,
            'error_rate': total_errors / total_cases,
            'fp_stats': fp_stats,
            'fn_stats': fn_stats,
        }
    
    def _generate_detailed_report(self, output_file: str, top_k: int, 
                                   fp_cases: pd.DataFrame, fn_cases: pd.DataFrame):
        """Generate detailed failure report."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CLINICAL AI FAILURE ANALYSIS REPORT\n")
            f.write(f"Architecture: {self.arch_name} (#{self.arch_num})\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Cases: {len(fp_cases) + len(fn_cases):,}\n")
            f.write(f"False Positives: {len(fp_cases):,} | False Negatives: {len(fn_cases):,}\n")
            f.write("="*80 + "\n\n")
            
            # Methodology note
            f.write("METHODOLOGY NOTE:\n")
            f.write("-" * 80 + "\n")
            f.write("Specialist scores derived from multi-modal analysis:\n")
            f.write("   • Note: ClinicalBERT Embeddings + LDA Topics + Manual Features\n")
            f.write("   • Lab: Organ dysfunction scores + Lab trajectories\n")
            f.write("   • 'WHY' explanations are heuristic interpretations of model outputs\n")
            f.write("=" * 80 + "\n\n")
            
            # Top FP cases
            if len(fp_cases) > 0:
                f.write("="*80 + "\n")
                f.write("TOP FALSE POSITIVE CASES\n")
                f.write("="*80 + "\n\n")
                
                top_fp = fp_cases.nlargest(top_k, 'y_prob')
                for idx, (_, case) in enumerate(top_fp.iterrows(), 1):
                    f.write(self.generate_case_report(case, idx, 'FALSE POSITIVE'))
            
            # Top FN cases
            if len(fn_cases) > 0:
                f.write("\n" + "="*80 + "\n")
                f.write("TOP FALSE NEGATIVE CASES\n")
                f.write("="*80 + "\n\n")
                
                top_fn = fn_cases.nsmallest(top_k, 'y_prob')
                for idx, (_, case) in enumerate(top_fn.iterrows(), 1):
                    f.write(self.generate_case_report(case, idx, 'FALSE NEGATIVE'))


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Analyze failures in multi-agent readmission prediction'
    )
    parser.add_argument('--arch', type=int, default=None,
        help='Architecture number to analyze (default: all)')
    parser.add_argument('--detailed', action='store_true',
        help='Generate detailed case reports')
    parser.add_argument('--top-k', type=int, default=25,
        help='Number of top cases to report')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CLINICAL AI FAILURE ANALYSIS SYSTEM")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find architectures to analyze
    if args.arch is not None:
        arch_nums = [args.arch]
    else:
        arch_nums = []
        for arch_num in range(7):
            arch_name = ARCHITECTURE_NAMES.get(arch_num, {}).get('abbrev', f'arch{arch_num}')
            pred_file = os.path.join(ANALYSIS_DIR, f'arch{arch_num}_{arch_name}', 
                                      'predictions_with_features.csv')
            if os.path.exists(pred_file):
                arch_nums.append(arch_num)
        
        if not arch_nums:
            print("\n⚠️  No prediction files found!")
            print(f"   Looking in: {ANALYSIS_DIR}")
            print("\n   Run compare_all_architectures.py first.")
            return
    
    print(f"\nAnalyzing architectures: {arch_nums}")
    
    # Run analysis
    all_results = []
    
    for arch_num in arch_nums:
        arch_name = ARCHITECTURE_NAMES.get(arch_num, {}).get('abbrev', f'arch{arch_num}')
        pred_file = os.path.join(ANALYSIS_DIR, f'arch{arch_num}_{arch_name}', 
                                  'predictions_with_features.csv')
        
        if not os.path.exists(pred_file):
            print(f"\n⚠️  Skipping Architecture {arch_num}: file not found")
            continue
        
        analyzer = FailureAnalyzer(arch_num, arch_name)
        
        try:
            df = analyzer.load_predictions(pred_file)
            results = analyzer.run_analysis(df, top_k=args.top_k, detailed=args.detailed)
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ Error analyzing Architecture {arch_num}: {e}")
            if args.detailed:
                import traceback
                traceback.print_exc()
    
    # Generate comparison
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("CROSS-ARCHITECTURE COMPARISON")
        print(f"{'='*80}\n")
        
        comparison_data = []
        for res in all_results:
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
        
        comparison_file = os.path.join(ANALYSIS_DIR, 'comparison_failure_analysis.csv')
        df_comparison.to_csv(comparison_file, index=False)
        print(f"\n✅ Comparison saved: {comparison_file}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()