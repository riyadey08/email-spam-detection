# email-spam-detection
# 📧 Email Spam and Phishing Detection  
### A Multi-Model Approach using Machine Learning and Deep Learning  

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧩 Overview
This repository presents an Email Spam and Phishing Detection System built using a blend of Machine Learning (ML) and Deep Learning (DL) models.  
It experiments with multiple algorithms — from Logistic Regression and Naive Bayes to LightGBM and LSTM — to detect spam or phishing messages with high accuracy.  

The project demonstrates preprocessing, model training, evaluation, and performance visualization in separate modular notebooks for clarity and scalability.

---

## 🗂️ Repository Structure

📦 Email_spam_and_phishing-main
│
├── 📁 NN/
│   ├── 📄 Spam detection Neural networks.ipynb     # Dense & CNN-based model
│   └── 📁 dataset/                                 # Dataset used by NN model
│
├── 📁 spam_LGBM/
│   ├── 📄 Spam_LGBM.ipynb                          # LightGBM-based classifier
│   └── 📄 spam.csv                                 # Dataset for LGBM model
│
├── 📁 Spam-Classifier-master ksd/
│   └── 📁 Spam-Classifier-master/
│       ├── 📄 Spam_Classifier_with_LSTM.ipynb      # LSTM implementation
│       ├── 📄 sms_using_lemmatizer_with_TFIdf_Vectorizer.ipynb
│       ├── 📄 sms_using_PorterStemmer_with_TFIdf_Vectorizer.ipynb
│       ├── 📄 spam.csv                             # Dataset for experiments
│       ├── 📄 workflow.gif                         # Workflow visualization
│       ├── 📁 images/                              # Supporting visuals
│       ├── 📄 LICENSE
│       └── 📄 README.md
│
├── 📄 Spam_detection.ipynb                         # Traditional ML models
├── 📄 spamCollection.csv                           # SMS spam dataset
├── 📄 spam.csv                                     # Secondary dataset
└── 📄 README.md                                    # Main project documentation



---

## ⚙️ Key Features
- Complete **end-to-end pipeline** for spam and phishing detection  
- Combines **traditional ML** and **deep learning** techniques  
- Text cleaning, tokenization, stemming, and lemmatization  
- Model comparison using consistent evaluation metrics  
- Visual analysis with confusion matrices and learning curves  
- Modular architecture for easy experimentation and scaling  

---

## 🧠 Models Implemented

| Category | Model | Description | Accuracy |
|-----------|--------|-------------|-----------|
| **Traditional ML** | Logistic Regression, SVM, Naive Bayes | TF-IDF vectorized classification | ~95% |
| **Ensemble ML** | LightGBM | Gradient boosting with TF-IDF features | ~97% |
| **Deep Learning (NN)** | Dense + Dropout layers | Multi-layer neural architecture | ~96% |
| **Sequential (LSTM)** | Long Short-Term Memory | Context-aware sequential modeling | ~98% |

---

## 🧹 Data Preprocessing
1. Remove special characters, punctuation, and stopwords  
2. Convert all text to lowercase  
3. Tokenize and vectorize text using **CountVectorizer**, **TF-IDF**, or **Word2Vec**  
4. Encode target labels (`spam` / `ham`)  
5. Split data into **training (80%)** and **testing (20%)** sets  

---

## 📊 Evaluation Metrics
- Accuracy  
- Precision, Recall, and F1-Score  
- Confusion Matrix  
- ROC-AUC Curve  
- Training vs Validation Loss  

---

## 💻 Tech Stack
- **Language:** Python 3.10+  
- **Libraries:**  
  - `numpy`  
  - `pandas`  
  - `scikit-learn`  
  - `lightgbm`  
  - `tensorflow` / `keras`  
  - `nltk`  
  - `matplotlib`  
  - `seaborn`  
- **Environment:** Jupyter Notebook / Google Colab  

---

### 1. Clone this repository
bash

git clone https://github.com/<your-username>/Email_spam_and_phishing.git
cd Email_spam_and_phishing-main
`
### 2. Install dependencies
pip install -r requirements.txt
### 3. Launch Jupyter Notebook
jupyter notebook
### 4. Run notebooks
Open and execute any .ipynb file (e.g., LSTM, LGBM, NN, or ML models) to train and evaluate results.

---

✅ Tip:  
GitHub sometimes collapses code blocks if they are inside numbered lists.  
To avoid this, use level-3 headings (###) instead of numbered lists for clean rendering — like above.

## 📈 Results Summary
- Traditional ML models deliver strong baselines for structured datasets.  
- LightGBM achieves high accuracy with low training time.  
- Neural Networks provide better generalization on unseen data.  
- LSTM achieves the best overall performance (~98% accuracy).  

---

## 🔮 Future Work
- Integrate transformer-based architectures (BERT, RoBERTa)  
- Deploy using Flask or Streamlit for real-time detection  
- Extend to multilingual and phishing URL detection  
- Add explainability via LIME or SHAP  

---

## 👩‍💻 Author
Riya Dey  
National Institute of Technology Durgapur  

📧 [Email](mailto:riyadey3134@gmail.com)  
🌐 [LinkedIn](https://www.linkedin.com/in/riya-dey-a31b43286)
