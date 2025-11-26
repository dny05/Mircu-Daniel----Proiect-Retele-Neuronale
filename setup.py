import os
import subprocess
import sys
from pathlib import Path

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def run_command(cmd, description):
    """Rulează o comandă și afișează progresul"""
    print(f"[*] {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[✓] {description} completat!")
            return True
        else:
            print(f"[✗] {description} eșuat!")
            print(f"Eroare: {result.stderr}")
            return False
    except Exception as e:
        print(f"[✗] {description} - Eroare: {e}")
        return False

def create_requirements():
    """requirements.txt"""
    content = """numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
torch>=2.0.0
streamlit>=1.28.0
plotly>=5.17.0
"""
    with open("requirements.txt", "w", encoding='utf-8') as f:
        f.write(content)
    print("[✓] Fișier requirements.txt creat")

def create_readme():
    """Creează README simplu"""
    content = """# Suspension Setup Evaluator

# 1. Instalează pachetele
pip install -r requirements.txt

# 2. Rulează aplicația
streamlit run app.py

# 3. Deschide browser-ul: http://localhost:8501
```

## Feature-uri:

1. Generează date telemetrice sintetice
2. Antrenează o rețea neuronală
3. Evaluează comportamentul (understeer/oversteer)
4. Recomandă ajustări pentru setup


```
Generate & Train -> Generează date -> Antrenează modelul
Evaluate ->  Vezi recomandări
```


## Probleme posibile

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
"""
    with open("README.md", "w", encoding='utf-8') as f:
        f.write(content)
    print("[✓] Fișier README.md creat")

def create_launcher_bat():
    """Creează launcher pentru Windows"""
    content = """@echo off
echo ============================================
echo  Suspension Setup Evaluator
echo ============================================
echo.

REM Verifică dacă există venv
if not exist "venv" (
    echo Creez mediul virtual...
    python -m venv venv
)

REM Activează venv
echo Activez mediul virtual...
call venv\\Scripts\\activate.bat

REM Verifică dacă sunt instalate pachetele
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Instalez dependințele...
    pip install -r requirements.txt
)

REM Pornește aplicația
echo.
echo ============================================
echo  Pornesc aplicația...
echo  Accesează: http://localhost:8501
echo ============================================
echo.
streamlit run app.py

pause
"""
    with open("launch.bat", "w", encoding='utf-8') as f:
        f.write(content)
    print("[✓] Fișier launch.bat creat (Windows)")

def create_launcher_sh():
    """Creează launcher pentru Linux/Mac"""
    content = """#!/bin/bash
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
"""
    with open("launch.sh", "w", encoding='utf-8') as f:
        f.write(content)
    
    # Face fișierul executabil
    try:
        os.chmod("launch.sh", 0o755)
        print("[✓] Fișier launch.sh creat (Linux/Mac)")
    except:
        print("[✓] Fișier launch.sh creat (rulează 'chmod +x launch.sh' pentru a-l face executabil)")

def main():
    print_header("SUSPENSION SETUP EVALUATOR - SETUP")
    
    print(f"Director curent: {os.getcwd()}")
    print()
    
    # Verifică versiunea Python
    python_version = sys.version_info
    print(f"Versiune Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("[✗] Este necesară versiunea Python 3.8 sau mai nouă!")
        return
    
    print("[✓] Versiunea Python este ok")
    print()
    
    # Creează fișierele
    print("Creez fișierele de configurare...")
    try:
        create_requirements()
        create_readme()
        create_launcher_bat()
        create_launcher_sh()
        print()
    except Exception as e:
        print(f"[✗] Eroare la crearea fișierelor: {e}")
        print()
        return
    
    # Verifică dacă există app.py
    if not Path("app.py").exists():
        print("[⚠] Fișierul app.py nu a fost găsit!")
        print("    Te rog salvează codul principal ca 'app.py'")
        print()
    else:
        print("[✓] Fișier app.py găsit")
        print()
    
    # Întreabă dacă vrea să instaleze
    print_header("INSTALARE")
    
    response = input("Vrei să instalez dependințele acum? (y/n): ").strip().lower()
    
    if response == 'y':
        print()
        
        # Creează venv
        if not Path("venv").exists():
            success = run_command(f"{sys.executable} -m venv venv", "Creez mediul virtual")
            if not success:
                print("\n[✗] Nu am putut crea mediul virtual")
                print("Încearcă manual: python -m venv venv")
                return
        else:
            print("[✓] Mediul virtual deja există")
        
        # Determină calea către pip
        if os.name == 'nt':  # Windows
            pip_path = "venv\\Scripts\\pip.exe"
            python_path = "venv\\Scripts\\python.exe"
        else:  # Linux/Mac
            pip_path = "venv/bin/pip"
            python_path = "venv/bin/python"
        
        # Verifică dacă există pip
        if not Path(pip_path).exists():
            print(f"[✗] pip nu a fost găsit la {pip_path}")
            return
        
        # Instalează pachetele
        run_command(f'"{pip_path}" install --upgrade pip', "Actualizez pip")
        
        print("\n[*] Instalez pachetele (poate dura 2-5 minute)...")
        success = run_command(f'"{pip_path}" install -r requirements.txt', "Instalez pachetele")
        
        if success:
            print()
            print_header("INSTALARE COMPLETĂ!")
            
            print("""
[✓] Setup complet! Pornire rapidă:
  Windows: Click dublu pe 'launch.bat'
  Linux/Mac: Rulează './launch.sh'


            """)
        else:
            print()
            print("[✗] Instalarea a eșuat!")
            print("\nÎncearcă instalarea manuală:")
            print("1. venv\\Scripts\\activate  (Windows)")
            print("   source venv/bin/activate  (Linux/Mac)")
            print("2. pip install numpy pandas scipy torch streamlit plotly")
            print("3. streamlit run app.py")
    else:
        print()
        print_header("FIȘIERE SETUP CREATE")
        
        print("""
Fișierele au fost create cu succes:
  [✓] requirements.txt
  [✓] README.md
  [✓] launch.bat (launcher Windows)
  [✓] launch.sh (launcher Linux/Mac)

Pentru instalare manuală:
  1. Creează venv: python -m venv venv
  2. Activează venv:
     Windows: venv\\Scripts\\activate
     Linux/Mac: source venv/bin/activate
  3. Instalează: pip install -r requirements.txt
  4. Rulează: streamlit run app.py
        """)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[ℹ] Setup întrerupt de utilizator")
    except Exception as e:
        print(f"\n[✗] Eroare neașteptată: {e}")
        import traceback
        traceback.print_exc()