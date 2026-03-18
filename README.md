#  Email Spam and Phishing Detection  
### A Multi-Model Approach using Machine Learning and Deep Learning  

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  Overview
This repository presents an Email Spam and Phishing Detection System built using a blend of Machine Learning (ML) and Deep Learning (DL) models.  
It experiments with multiple algorithms — from Logistic Regression and Naive Bayes to LightGBM and LSTM — to detect spam or phishing messages with high accuracy.  

The project demonstrates preprocessing, model training, evaluation, and performance visualization in separate modular notebooks for clarity and scalability.

---

##  Repository Structure

```
email-spam-detection/
│
├── dataset/
│   └── spam.csv
│
├── notebooks/
│   └── spam_detection.ipynb
│
├── README.md
└── requirements.txt
```


---

##  Key Features
- Complete **end-to-end pipeline** for spam and phishing detection  
- Combines **traditional ML** and **deep learning** techniques  
- Text cleaning, tokenization, stemming, and lemmatization  
- Model comparison using consistent evaluation metrics  
- Visual analysis with confusion matrices and learning curves  
- Modular architecture for easy experimentation and scaling  

---

##  Models Implemented

| Category | Model | Description | Accuracy |
|-----------|--------|-------------|-----------|
| **Traditional ML** | Logistic Regression, SVM, Naive Bayes | TF-IDF vectorized classification | ~98% |
| **Ensemble ML** | LightGBM | Gradient boosting with TF-IDF features | ~97% |
| **Deep Learning (NN)** | Dense + Dropout layers | Multi-layer neural architecture | ~96% |
| **Sequential (LSTM)** | Long Short-Term Memory | Context-aware sequential modeling | ~86% |

---

##  Data Preprocessing
1. Remove special characters, punctuation, and stopwords  
2. Convert all text to lowercase  
3. Tokenize and vectorize text using **CountVectorizer**, **TF-IDF**, or **Word2Vec**  
4. Encode target labels (`spam` / `ham`)  
5. Split data into **training (80%)** and **testing (20%)** sets  

---

##  Evaluation Metrics
- Accuracy  
- Precision, Recall, and F1-Score  
- Confusion Matrix  
- ROC-AUC Curve  
- Training vs Validation Loss  

---

##  Tech Stack
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


##  Results Summary
- Traditional ML models deliver strong baselines for structured datasets.  
- LightGBM achieves high accuracy with low training time.  
- Neural Networks provide better generalization on unseen data.  
- SVM achieves the best overall performance (~98% accuracy).  

---


##  Limitations
- The model may not perform well on real-world email data as it is trained on specific dataset.
- The model mainly focuses on textual content and may fail to detect spam in: Images, Attachments, Embedded links.  
- May genarate False predictions(False Positive, False Negative).  
- Trained only on English data, the model may struggle with Multilingual emails or Regional slang or mixed languages.

---

##  Future Work
- Integrate transformer-based architectures (BERT, RoBERTa)  
- Deploy using Flask or Streamlit for real-time detection  
- Extend to multilingual and phishing URL detection  
- Add explainability via LIME or SHAP
- Add image-based spam detection, and continuous learning systems  

---

##  Author
**Riya Dey**  
*National Institute of Technology Durgapur*  

📧 [Email](mailto:riyadey3134@gmail.com)  
🌐 [LinkedIn](https://www.linkedin.com/in/riya-dey-a31b43286)
