# Suspension Setup Evaluator

## Pornire rapidă

```bash
# 1. Instalează pachetele
pip install -r requirements.txt

# 2. Rulează aplicația
streamlit run app.py

# 3. Deschide browser-ul: http://localhost:8501
```

## Ce face aplicația

1. Generează date telemetrice sintetice
2. Antrenează o rețea neuronală
3. Evaluează comportamentul (understeer/oversteer)
4. Recomandă ajustări pentru setup

## Fluxul de lucru

```
Generate & Train -> Generează date -> Antrenează modelul
Evaluate -> Generează telemetrie de test -> Vezi recomandări
```

## Caracteristici

- Tot codul într-un singur fișier
- Interfață web modernă (Streamlit)
- Antrenament în 1-2 minute
- Evaluare instantanee
- Recomandări concrete

## Probleme frecvente

**Eroare: "No module named..."**
```bash
pip install -r requirements.txt
```

**Portul e deja folosit:**
```bash
streamlit run app.py --server.port 8502
```

**PyTorch prea mare:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Fișiere necesare

- `app.py` - Aplicația completă
- `requirements.txt` - Dependințe

Doar 2 fișiere!

---

**Creat pentru Formula Student**
