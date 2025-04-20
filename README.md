[This repository has been modified from an original private repository to remove potentially sensitive data files]

# VoltronX: Federated Learning for Privacy-Preserving Coordination in Fraud Detection
This project implements a federated learning framework for fraud detection across financial institutions while ensuring data privacy and regulatory compliance. Each bank trains a local fraud detection model on its proprietary transaction data, sharing only encrypted model updates with a central aggregator. The global model benefits from collaborative learning without exposing sensitive data.

Key features:
- Privacy-Preserving Training: No raw data is shared across institutions.
- Global Model Aggregation: Improved fraud detection across banks.
- Incentive Mechanism: Shapley values ensure fair contribution assessment.

## Usage
### Clone the Repository
```bash
git clone https://github.com/KennethSee/BISAnalyticsChallenge2025.git
cd BISAnalyticsChallenge2025
```

### Install Required Packages
```bash
pip install -r requirements.txt
```
### Launch App
```bash
streamlit run app.py
```
