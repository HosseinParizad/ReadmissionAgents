# Readmission Agents

A multi-agent system for predicting 30-day hospital readmissions using MIMIC-IV data. This project implements a "doctor brain" that coordinates multiple specialist agents, each analyzing different aspects of patient data.

## Project Structure

```
ReadmissionAgents/
├── src/
│   ├── ExtractMimicData.py      # Data extraction from MIMIC-IV
│   ├── specialist_agents.py      # Specialist agent implementations
│   └── doctor_brain.py           # Main doctor agent coordinator
├── data/                         # Raw data files (not in repo)
├── model_outputs/                # Generated features and model outputs
├── venv/                         # Virtual environment
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup

1. **Create and activate virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure MIMIC-IV data path:**
   - Edit `src/ExtractMimicData.py` and update `BASE_PATH` to point to your MIMIC-IV data directory

## Usage

1. **Extract features from MIMIC-IV:**
   ```bash
   python src/ExtractMimicData.py
   ```
   This will create `model_outputs/features_4_agents.csv`

2. **Train and evaluate the model:**
   ```bash
   python src/doctor_brain.py
   ```

## System Architecture

The system consists of:

- **Doctor Agent**: Coordinates all specialists and makes final predictions
- **Lab Specialist**: Analyzes lab values and vital signs
- **Note Specialist**: Processes clinical notes using sentence transformers
- **Pharmacy Specialist**: Analyzes medication lists
- **History Specialist**: Analyzes diagnosis history

Each specialist provides an "opinion" (probability) to the doctor agent, which combines them with patient context to make the final readmission prediction.

## Requirements

- Python 3.8+
- MIMIC-IV database access
- See `requirements.txt` for full dependency list

## License

[Add your license here]

