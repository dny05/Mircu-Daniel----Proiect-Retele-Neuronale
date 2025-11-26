# Mircu-Daniel----Proiect-Retele-Neuronale
# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** MIRCU Daniel Ioan  
**Data:** 20/11/2025


---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
suspension_setup_evaluator/
├── src/
│   ├── data_loader.py         
│   ├── preprocessing.py       
│   ├── feature_extraction.py   
│   ├── models.py              
│   ├── trainer.py             
│   ├── evaluator.py            
│   └── utils.py                
├── gui/
│   └── main_window.py         
├── scripts/
│   ├── generate_sample_data.py 
│   ├── train_model.py          
│   └── evaluate_telemetry.py   
├── app.py                       
├── main.py                      
└── test_installation.py         
```
---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Achizitie de date - senzori monopost Formula Student - potentiometre liniare, IMU
* **Modul de achiziție:**  Senzori reali 
* **Perioada / condițiile colectării:** Octombrie 2025 - Decembrie 2025, condiții experimentale pe circuit cu diferite setup-uri de suspensie]

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 40000
* **Număr de caracteristici (features):** 12
* **Tipuri de date:**  Numerice 
* **Format fișiere:**  CSV 
### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|--------------------|---------|-------------|---------------|--------------------|
|       index        | numeric |      -      |  index citire |      0 – 40000     |
|     elapse_time    | numeric |     ms      |      timp     |     0 – 2000000    |
|       susp_fl      | numeric |     mm      |susp. fata st. |       26 – 42      |
|       susp_fr      | numeric |     mm      |susp. fata dr. |       32 – 48      |
|       susp_rl      | numeric |     mm      |susp. spate st.|       45 – 52      |
|       susp_rr      | numeric |     mm      |susp. spate dr.|       45 – 52      |
|        acc_x       | numeric |   m/s^2     |acc. axa long. |      -20 - 20      |
|        acc_y       | numeric |   m/s^2     |acc. axa trans.|      -20 - 20      |
|        acc_z       | numeric |   m/s^2     |acc. axa vert. |      -20 - 20  |           
|        rot_x       | numeric |     deg     |acc. axa long. |        0 - 1       |
|        rot_y       | numeric |     deg     |acc. axa trans.|        0 - 1       |
|        rot_z       | numeric |     deg     |acc. axa vert. |        0 - 1    |          


**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

| Caracteristică | Medie     | Mediană   | Min   | Max       | Std Dev |
|----------------|-----------|-----------|-------|-----------|---------|
| elapse_time    | 1,000,234 | 1,000,120 | 0     | 2,000,000 | 577,350 |
| susp_fl        | 34.2      | 34        | 26    | 42        | 4.1     |
| susp_fr        | 40.1      | 40        | 32    | 48        | 4.3     |
| susp_rl        | 48.2      | 48        | 45    | 52        | 1.9     |
| susp_rr        | 48.0      | 48        | 45    | 52        | 2.0     |
| acc_x          | 0.2       | 0.1       | -20   | 20        | 4.5     |
| acc_y          | 0.0       | 0.0       | -20   | 20        | 4.7     |
| acc_z          | 9.81      | 9.81      | -20   | 20        | 3.2     |
| rot_x          | 0.5       | 0.5       | 0     | 1         | 0.2     |
| rot_y          | 0.5       | 0.5       | 0     | 1         | 0.2     |
| rot_z          | 0.5       | 0.5       | 0     | 1         | 0.2     |

*Observație:* Valorile medii ale suspensiilor indică un setup echilibrat față-spate, iar acc_z este centrat pe gravitație (~9.81 m/s²).

* **Valori lipsă detectate:**  
  * susp_fl – 0.2%  
  * acc_y – 0.5%  

* **Valori anormale / eronate:**  
  * rot_x, rot_y, rot_z – valori în afara intervalului 0–1 (corectate prin clipping)  
  * acc_x/acc_y – valori extreme > ±20 m/s² (outlier tratat prin limitare percentile 1–99%)

* **Corelații puternice:**  
  * susp_fl – susp_fr: r = 0.85  
  * susp_rl – susp_rr: r = 0.88  
  * acc_x – rot_x: r = 0.30 (moderată)  

### 3.3 Probleme identificate

* Feature `acc_y` are 0.5% valori lipsă – imputare necesară  
* Distribuția `rot_z` este ușor neuniformă – nu critic pentru model  
* Corelație ridicată între susp_rl și susp_rr – se poate opta pentru reducerea dimensionalității sau PCA

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor


* Eliminare duplicatelor – 12 observații duplicate eliminate  
* Tratarea valorilor lipsă:  
  * `acc_y` și `susp_fl` – imputare cu mediană  
* Tratarea outlierilor:  
  * `acc_x`, `acc_y` – valori limitate între percentila 1 și 99  


### 4.2 Transformarea caracteristicilor

* **Normalizare:** Min–Max / Standardizare


### 4.3 Structurarea seturilor de date

**Împărțire recomandată:**
* 70–80% – train
* 10–15% – validation
* 10–15% – test

**Principii respectate:**
* Stratificare pentru clasificare
* Fără scurgere de informație (data leakage)
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în `data/processed/`
* Seturi train/val/test în foldere dedicate
* Parametrii de preprocesare în `config/preprocessing_config.*` (opțional)

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute
* `data/processed/` – date curățate & transformate
* `data/train/`, `data/validation/`, `data/test/` – seturi finale
* `src/preprocessing/` – codul de preprocesare
* `data/README.md` – descrierea dataset-ului

---

##  6. Stare Etapă (de completat de student)

- [ ] Structură repository configurată
- [ ] Dataset analizat (EDA realizată)
- [ ] Date preprocesate
- [ ] Seturi train/val/test generate
- [ ] Documentație actualizată în README + `data/README.md`

---
