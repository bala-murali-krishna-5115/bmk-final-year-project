# Setup and Environment Guide

## Prerequisites
- Python 3.11+
- Virtual environment (recommended)
- Git

## Step-by-Step Installation

1. **Clone / Navigate to Workspace**:
   ```bash
   cd "c:\Users\balam\OneDrive\Desktop\Sarvagna Innovation Internship\Final Year Project"
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize Database**:
   ```bash
   python -m backend.database.init_db
   ```

4. **Start Backend Server**:
   ```bash
   python -m backend.app.main
   ```

5. **Run Test Suite**:
   ```bash
   pytest
   ```
