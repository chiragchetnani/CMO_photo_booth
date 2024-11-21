# Clustering Project Setup Guide

This guide will help you set up and run the clustering project on Windows.

## Prerequisites

- Python 3.9 or 3.10
- Git (optional, for cloning the repository)

## Installation Steps

### 1. Install Python

1. Download Python 3.9 or 3.10 from [Python's official website](https://www.python.org/downloads/)
2. Run the installer
3. **Important**: During installation, check the box that says "Add Python to PATH"
4. Verify installation by opening Command Prompt and typing:
   ```bash
   python --version
   ```

### 2. Set Up Python Environment Variables (If Not Done During Installation)

1. Right-click on 'This PC' or 'My Computer'
2. Click 'Properties'
3. Click 'Advanced system settings'
4. Click 'Environment Variables'
5. Under 'System Variables', find and select 'Path'
6. Click 'Edit'
7. Click 'New'
8. Add these paths (replace with your Python installation path):
   ```
   C:\Users\YourUsername\AppData\Local\Programs\Python\Python39
   C:\Users\YourUsername\AppData\Local\Programs\Python\Python39\Scripts
   ```
9. Click 'OK' on all windows

### 3. Create and Set Up Virtual Environment

Open Command Prompt and run the following commands:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

### 4. Project Setup

With the virtual environment activated, run these commands:

```bash
# Install required packages
pip install -r requirements.txt

# Run the application
python3 fast.py
streamlit run app.py
```
