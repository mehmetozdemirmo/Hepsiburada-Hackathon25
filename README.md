# 🏆 TEKNOFEST Hepsiburada Address Parsing Hackathon  
**Address Normalization & Hierarchical Matching Pipeline**

This repository contains the full pipeline we developed for the **TEKNOFEST Hepsiburada Address Parsing Hackathon**.  
The project focuses on **address cleaning, normalization, hierarchical extraction, and machine learning–based prediction** to handle messy real-world address data.

---

## 📂 Project Overview
The goal is to process large sets of Turkish address data and accurately predict their **province (il)**, **district (ilçe)**, and **neighborhood (mahalle)** components.  
The pipeline includes:
- **Data Cleaning & Normalization**  
- **Hierarchical Address Extraction**  
- **Text Feature Engineering (N-grams)**  
- **TF-IDF Vectorization & Nearest Neighbors Modeling**  
- **Efficient Batch Inference**

---

## ⚡ Main Features

### 1️⃣ Safe CSV Reading
- **`read_csv_safe`** securely loads:
  - `train_sorted.csv`
  - `test_normalized.csv`

---

### 2️⃣ Address Normalization
- **`normalize_address`** converts raw addresses into a **standard format**:
  - Fixes typos and abbreviations in provinces/districts  
    - e.g. `ist.` → `istanbul`, `ankra` → `ankara`
  - Expands common abbreviations for streets and neighborhoods  
    - e.g. `mh` → `mahalle`, `cd` → `cadde`, `sk` → `sokak`
- The normalized results are stored in a new `address` column for both **train** and **test** datasets.

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

## ⚙️ Tech Stack
- **Python 3.x**
- `pandas`, `numpy`
- `scikit-learn` – TF-IDF vectorization & Nearest Neighbors
- `rapidfuzz` – fuzzy string matching
- `tqdm`, `logging` – progress tracking and monitoring

---

## 🚀 Installation & Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/address-matching.git
cd address-matching

# 🚀 cleaRoute – Turkish Address Parsing & Normalization
> **Clean, organize, and analyze complex Turkish address data with NLP-powered accuracy.**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)
<!-- Add more badges as needed: build status, stars, forks, etc. -->

---

## 🎯 Project Overview
This project was developed during the **TEKNOFEST Hepsiburada Address Parsing Hackathon**,  
where our team tackled the challenge of **parsing and structuring messy Turkish addresses**.

Unstandardized addresses often lead to:
- 🚚 **Delivery errors**
- ⏱️ **Operational delays**
- 💸 **Increased logistics costs**

**cleaRoute** provides a streamlined solution by:
- Cleaning raw address data
- Normalizing inconsistent formats
- Extracting key components (city, district, neighborhood, street)
- Matching similar addresses for deduplication

---

## 📊 Dataset
The project used a combination of:
- **Hackathon-provided datasets** (private)
- Custom synthetic datasets for testing and validation

Key files:
- `train.csv` – Training data used to build the model  
- `test.csv` – Test set for evaluation  
- `submission.csv` – Sample output of parsed and normalized addresses

---

## ✨ Key Features
- **Data Cleaning & Normalization** – Handle typos, inconsistent spacing, and casing.
- **Component Extraction** – Identify city, district, neighborhood, and street details.
- **String Matching** – Apply `fuzzywuzzy` and `rapidfuzz` to detect and merge similar addresses.
- **Turkish NLP Integration** – Utilize models like **BerTurk** and HuggingFace **transformers** for better language understanding.
- **Scalable Workflow** – Works efficiently with large datasets.

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
├── KaggleEmbedding.ipynb    # Data preprocessing & embedding creation
├── colab_training.ipynb     # Main model training and evaluation
├── main.py                   # Command-line pipeline to clean & parse addresses
├── requirements.txt          # Python dependencies
└── submission.csv            # Sample prediction results
