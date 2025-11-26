#!/bin/bash
echo "============================================"
echo "  Suspension Setup Evaluator"
echo "============================================"
echo ""

# Verifică dacă există venv
if [ ! -d "venv" ]; then
    echo "Creez mediul virtual..."
    python3 -m venv venv
fi

# Activează venv
echo "Activez mediul virtual..."
source venv/bin/activate

# Verifică dacă sunt instalate pachetele
python -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Instalez dependințele..."
    pip install -r requirements.txt
fi

# Pornește aplicația
echo ""
echo "============================================"
echo "  Pornesc aplicația..."
echo "  Accesează: http://localhost:8501"
echo "============================================"
echo ""
streamlit run app.py
