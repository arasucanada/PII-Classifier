# PII Classifier

A Proof of Concept (POC) application for detecting Personally Identifiable Information (PII) using machine learning models, including Roblox sequence classifier and Piiranha token classifier.

## Requirements

- Python >= 3.12

## Installation

1. Install uv if not already installed:
   ```bash
   pip install uv
   ```

2. Install dependencies:
   ```bash
   uv pip install -e .
   ```

## Running the Application

Start the Streamlit app with:
```bash
streamlit run app/ui/streamlit_app.py
```

This launches a web interface where you can enter text and analyze it for PII using the integrated models.