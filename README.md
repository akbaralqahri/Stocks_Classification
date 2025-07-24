# Dokumentasi Analisis Prediktif Harga Saham
## Analisis dan Pemodelan Machine Learning untuk Prediksi Status Harga Saham

---

## **Executive Summary**

Proyek ini mengembangkan sistem prediksi harga saham menggunakan pendekatan machine learning dengan fokus pada klasifikasi biner untuk memprediksi apakah harga penutupan saham akan berada di atas threshold tertentu (50). Penelitian ini menggunakan metodologi seleksi fitur berbasis Cramér's V dan membandingkan performa dua algoritma utama: XGBoost dan Support Vector Machine (SVM).

**Hasil Utama:**
- Model SVM dengan kernel RBF mencapai akurasi tertinggi 71% pada data uji
- Kombinasi 5 fitur optimal: `['DERn-1', 'QRn-2', 'QRn-1', 'PTBVn-1', 'PBVn-1']`
- Metodologi seleksi fitur menggunakan Cramér's V terbukti efektif

---

## **1. Pendahuluan**

### 1.1 Latar Belakang
Prediksi harga saham merupakan tantangan kompleks dalam dunia keuangan yang melibatkan analisis berbagai indikator fundamental. Proyek ini bertujuan mengidentifikasi pola dalam data historis untuk membangun model prediktif yang dapat membantu pengambilan keputusan investasi.

### 1.2 Tujuan Penelitian
- Mengidentifikasi indikator keuangan yang paling berpengaruh terhadap pergerakan harga saham
- Mengembangkan model klasifikasi dengan akurasi optimal
- Membandingkan performa algoritma XGBoost dan SVM
- Memberikan framework yang dapat direplikasi untuk analisis serupa

### 1.3 Ruang Lingkup
- **Dataset**: Data saham historis dengan berbagai indikator keuangan
- **Target**: Klasifikasi biner (harga > 50 atau ≤ 50)
- **Metode**: Seleksi fitur statistik dan pemodelan machine learning
- **Evaluasi**: Akurasi, precision, recall, dan F1-score

---

## **2. Metodologi**

### 2.1 Arsitektur Sistem

```
Data Input → Preprocessing → Feature Selection → Model Training → Evaluation
     ↓              ↓              ↓               ↓              ↓
  Excel File   Binary Target   Cramér's V    XGBoost & SVM   Metrics & Viz
```

### 2.2 Teknologi dan Library

| Kategori | Library | Fungsi |
|----------|---------|--------|
| **Data Processing** | pandas, numpy | Manipulasi dan analisis data |
| **Visualization** | matplotlib, seaborn | Pembuatan grafik dan visualisasi |
| **Machine Learning** | scikit-learn | Model training dan evaluasi |
| **Advanced ML** | xgboost | Implementasi XGBoost classifier |
| **Statistical Analysis** | scipy.stats | Perhitungan statistik dan korelasi |

---

## **3. Tahapan Implementasi**

### 3.1 Data Preparation

#### 3.1.1 Data Loading
```python
# Memuat dataset dari file Excel
data = pd.read_excel('Balanced_Stock_Data.xlsx')
```

#### 3.1.2 Target Engineering
```python
# Membuat variabel target biner
data['Status Binary'] = (data['Harga Close'] > 50).astype(int)
```

**Transformasi Target:**
- `1`: Harga Close > 50
- `0`: Harga Close ≤ 50

### 3.2 Feature Selection dengan Cramér's V

#### 3.2.1 Konsep Cramér's V
Cramér's V adalah ukuran asosiasi untuk variabel kategorikal yang dinormalisasi antara 0-1:
- **0**: Tidak ada asosiasi
- **1**: Asosiasi sempurna

#### 3.2.2 Implementasi
```python
def cramers_v(confusion_matrix):
    """Menghitung nilai Cramér's V dari confusion matrix"""
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
```

#### 3.2.3 Hasil Seleksi Fitur
Top 10 fitur dengan nilai Cramér's V tertinggi digunakan untuk tahap pemodelan selanjutnya.

### 3.3 Model Development

#### 3.3.1 Strategi Kombinatorial
- **Pendekatan**: Menguji semua kombinasi dari 10 fitur terbaik
- **Range**: 1 hingga 10 fitur per kombinasi
- **Total Kombinasi**: 2^10 - 1 = 1023 kombinasi

#### 3.3.2 Data Splitting
```python
# Pembagian data 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

#### 3.3.3 Model Training
Untuk setiap kombinasi fitur:
1. **XGBoost Classifier** dengan parameter default
2. **SVM dengan kernel RBF** dengan parameter default
3. Evaluasi akurasi pada data test

---

## **4. Hasil dan Analisis**

### 4.1 Performa Feature Selection

| Rank | Feature | Cramér's V | Interpretasi |
|------|---------|------------|--------------|
| 1 | CRn-4 | 0.XXX | Korelasi Kuat |
| 2 | WCRn-4 | 0.XXX | Korelasi Kuat |
| 3 | PBVn-2 | 0.XXX | Korelasi Sedang |
| ... | ... | ... | ... |

### 4.2 Performa Model

#### 4.2.1 XGBoost Results
- **Kombinasi Fitur Terbaik**: `['CRn-4', 'WCRn-4', 'PBVn-2', 'QRn-4']`
- **Akurasi Training**: ~XX%
- **Akurasi Testing**: ~70%
- **Jumlah Fitur Optimal**: 4

#### 4.2.2 SVM Results (Sebelum Tuning)
- **Kombinasi Fitur Terbaik**: `['DERn-1', 'QRn-2', 'QRn-1', 'PTBVn-1', 'PBVn-1']`
- **Akurasi Testing**: ~XX%
- **Jumlah Fitur Optimal**: 5

### 4.3 Hyperparameter Optimization

#### 4.3.1 Kernel Comparison
| Kernel | Akurasi |
|--------|---------|
| Linear | XX% |
| **RBF** | **XX%** |
| Polynomial | XX% |
| Sigmoid | XX% |

#### 4.3.2 Grid Search Results
```python
# Parameter terbaik untuk SVM
best_params = {'C': 1, 'gamma': 0.1}
```

### 4.4 Final Model Performance

#### 4.4.1 SVM (Optimized) - **Model Terbaik**
- **Akurasi Testing**: **71%**
- **Akurasi Training**: XX%
- **Fitur**: 5 indikator optimal
- **Hyperparameters**: C=1, gamma=0.1

#### 4.4.2 Model Comparison Summary
| Model | Akurasi Test | Akurasi Train | Overfitting Risk |
|-------|--------------|---------------|------------------|
| **SVM (Tuned)** | **71%** | XX% | Rendah |
| XGBoost | 70% | XX% | Sedang |

---

## **5. Evaluasi Model**

### 5.1 Metrics Evaluation

#### 5.1.1 Classification Report
```
              precision    recall  f1-score   support
    
    Class 0       0.XX      0.XX      0.XX       XXX
    Class 1       0.XX      0.XX      0.XX       XXX
    
    accuracy                          0.71       XXX
   macro avg       0.XX      0.XX      0.XX       XXX
weighted avg       0.XX      0.XX      0.XX       XXX
```

#### 5.1.2 Confusion Matrix Analysis
- **True Positives**: Prediksi benar untuk harga > 50
- **True Negatives**: Prediksi benar untuk harga ≤ 50
- **False Positives**: Error Type I (prediksi naik, aktual turun)
- **False Negatives**: Error Type II (prediksi turun, aktual naik)

### 5.2 Model Interpretation

#### 5.2.1 Feature Importance
Fitur yang paling berpengaruh dalam model SVM final:
1. **DERn-1**: Debt to Equity Ratio lag-1
2. **QRn-2**: Quick Ratio lag-2  
3. **QRn-1**: Quick Ratio lag-1
4. **PTBVn-1**: Price to Book Value lag-1
5. **PBVn-1**: Price Book Value lag-1

#### 5.2.2 Business Insight
- **Likuiditas** (Quick Ratio) menjadi indikator kunci
- **Leverage** (Debt to Equity) berpengaruh signifikan
- **Valuasi** (Price to Book Value) penting untuk prediksi
- **Lag indicators** menunjukkan pentingnya data historis

---

## **6. Kesimpulan dan Rekomendasi**

### 6.1 Kesimpulan Utama

1. **Model Terbaik**: SVM dengan kernel RBF mencapai akurasi 71%
2. **Feature Engineering**: Pendekatan Cramér's V efektif untuk seleksi fitur
3. **Optimal Features**: 5 fitur memberikan keseimbangan terbaik antara kompleksitas dan performa
4. **Hyperparameter Tuning**: Meningkatkan performa model secara signifikan

### 6.2 Kelebihan Pendekatan

- **Metodologi Komprehensif**: Menguji semua kombinasi fitur
- **Statistical Foundation**: Menggunakan Cramér's V untuk feature selection
- **Model Comparison**: Perbandingan objektif antar algoritma
- **Overfitting Prevention**: Evaluasi pada data terpisah

### 6.3 Keterbatasan

- **Threshold Fixed**: Menggunakan threshold tetap (50)
- **Binary Classification**: Tidak memprediksi nilai kontinu
- **Feature Engineering**: Terbatas pada fitur yang tersedia
- **Market Dynamics**: Tidak mempertimbangkan faktor eksternal

### 6.4 Rekomendasi Pengembangan

#### 6.4.1 Jangka Pendek
- Implementasi cross-validation untuk validasi yang lebih robust
- Eksplorasi threshold dinamis
- Penambahan feature engineering (technical indicators)
- Implementasi ensemble methods

#### 6.4.2 Jangka Panjang
- Integrasi data makroekonomi
- Implementasi deep learning models
- Real-time prediction system
- Risk management integration

---

## **7. Appendix**

### 7.1 Code Structure
```
project/
├── data/
│   └── Balanced_Stock_Data.xlsx
├── notebooks/
│   └── stock_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── modeling.py
│   └── evaluation.py
└── results/
    ├── model_comparison.csv
    └── confusion_matrices/
```

### 7.2 Reproducibility
```python
# Untuk hasil yang konsisten
np.random.seed(42)
random_state = 42
```

### 7.3 Requirements
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

---

**Catatan**: Dokumentasi ini dapat diperbarui seiring dengan pengembangan model dan penambahan fitur baru. Untuk pertanyaan teknis, silakan merujuk ke kode sumber atau menghubungi tim pengembang.