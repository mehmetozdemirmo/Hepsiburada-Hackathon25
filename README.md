# 🏆 TEKNOFEST Hepsiburada Address Parsing Hackathon  
**Address Normalization & Hierarchical Matching Pipeline**

This repository contains the full pipeline we developed for the **TEKNOFEST Hepsiburada Address Parsing Hackathon**.  
The project focuses on **address cleaning, normalization, hierarchical extraction, and machine learning–based prediction** to handle messy real-world address data.

---

## 📂 Project Overview  
The goal is to process large sets of Turkish address data and accurately predict their **province (il)**, **district (ilçe)**, and **neighborhood (mahalle)** components.  

The pipeline includes:  
- 🧹 **Data Cleaning & Normalization** → Corrects typos, expands abbreviations, and removes unnecessary characters to standardize address formats.  
- 🗂️ **Hierarchical Address Extraction** → Extracts structured components in a hierarchical order: Province → District → Neighborhood.  
- ✍️ **Text Feature Engineering (N-grams)** → Generates meaningful features using word-level and character-level n-grams.  
- 📊 **TF-IDF Vectorization & Nearest Neighbors Modeling** → Converts addresses into numerical vectors and applies nearest neighbors algorithms for classification.  
- ⚡ **Efficient Batch Inference** → Enables fast and memory-efficient predictions on large datasets using batch processing.  
- 🔗 **String Matching** → Uses `fuzzywuzzy` and `rapidfuzz` to detect and merge similar addresses.  
- 🧠 **Turkish NLP Integration** → Leverages models like **BerTurk** and HuggingFace **transformers** for better language understanding.  
- 🛠 **Tools & Libraries** → scikit-learn, pandas, rapidfuzz, tqdm, re, unicodedata  
- 📦 **Deployment Readiness** → API integration and batch processing infrastructure for real-world usage.  
- 📏 **Evaluation & Metrics** → Measures model performance using Accuracy, F1-score, and detailed error analysis.

---

## ⚡ Main Features

### 1️⃣ Safe CSV Reading  
- **`read_csv_safe`** is used to securely load the required datasets, ensuring proper encoding and error handling.  
- Supported files include:  
  - `train.csv`  
  - `test.csv`  
  - `train_sorted.csv`  
  - `test_normalized.csv`

---

### 2️⃣ Address Normalization  

The **`normalize_address`** function standardizes raw address strings into a **consistent and machine-readable format**.  
This step is crucial for reducing data noise and handling inconsistencies across large datasets.  

#### 🔧 Key Operations:
- ✅ **Province/District Name Corrections**  
  - Fixes spelling errors and variations in city/district names.  
  - Examples:  
    - `ist.` → `istanbul`  
    - `ankra` → `ankara`  

- ✅ **Abbreviation Expansion**  
  - Converts common abbreviations into their full forms to ensure uniformity.  
  - Examples:  
    - `mh` → `mahalle`  
    - `cd` → `cadde`  
    - `sk` → `sokak`  

- ✅ **Whitespace & Special Character Handling**  
  - Removes extra spaces, punctuation, or unnecessary symbols that might cause mismatches.  

#### 📦 Output:

 The normalized results are stored in a new `address` column for both **train** and **test** datasets.
 
---

### 3️⃣ Clean Data Export
- Saves cleaned data into:
  - `train_normalized.csv` → contains **label** & **address_normalized**
  - `test_normalized.csv` → contains **id** & **address_normalized**

---

### 4️⃣ Hierarchical Address Processing
- Loads an **address hierarchy file** to build:
  - **Province list**
  - **District–province lookup**
  - **Neighborhood–location lookup** (supports fast matching with variants)
- Extracts components:
  - **Province** → exact match
  - **District** → exact match, else **fuzzy matching**
  - **Neighborhood** → variant table & regex, else fuzzy matching
- Produces a **standard hierarchical address** in the format:  

- Saves results to **`test_hierarchical.csv`**.

---

### 5️⃣ Feature Engineering
- Generates **N-grams (bigrams, trigrams)** and merges them with the address to create richer text features.

---

### 6️⃣ Model Training
- Reads `train_benzer.csv` and creates feature-rich text data.
- Uses **TF-IDF vectorization** to convert addresses into numerical vectors.
- Builds a **CPU-based brute-force Nearest Neighbors model** with **cosine similarity**.

---

### 7️⃣ Address Prediction (Inference)
- For each test address:
- Finds nearest neighbors and collects their labels.
- Uses **weighted voting** based on similarity scores.
- If similarity is too low → **fallback to the most frequent label**.
- Supports **batch processing** for memory-efficient, high-speed predictions.

---

### 8️⃣ Full Pipeline Execution
The `main` function:
- Configures parameters (fuzzy thresholds, debug limits, etc.)
- Trains the model on normalized data.
- Processes test addresses.
- Generates prediction outputs and summary statistics.

---

## 📁 Key Output Files
| File Name               | Description                                      |
|--------------------------|---------------------------------------------------|
| `train_normalized.csv`   | Cleaned training data (label + normalized address)|
| `test_normalized.csv`    | Cleaned test data (id + normalized address)       |
| `test_hierarchical.csv`  | Test data with hierarchical components            |
| `predictions.csv`        | Final model predictions                           |

---


## 🛠️ Tech Stack
| Category             | Technologies Used |
|-----------------------|-------------------|
| **Programming**      | Python 3.9+ |
| **Libraries**        | pandas, NumPy, scikit-learn, fuzzywuzzy, rapidfuzz, regex |
| **NLP Tools**        | BerTurk, HuggingFace Transformers |
| **Environment**      | Jupyter Notebook, Colab |
| **Version Control**  | Git & GitHub |

---

## 📁 Project Structure

hackathon/  
├── adres_hiyerarsi.json          # Turkish address hierarchy (province → district → neighborhood mapping)  
├── turkiye.json                  # Reference data for Turkish provinces, districts, and neighborhoods  
├── hierarchical_organization.py  # Pipeline for cleaning, normalizing, and parsing addresses  
├── address_matcher.py            # Core matching module (TF-IDF + Nearest Neighbors)  
└── standardization.py            # Address normalization and preprocessing utilities  

---
