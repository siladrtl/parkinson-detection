# Parkinson Tespit Sistemi
### Çoklu Ses Kayıtları Üzerinde Robust Ağırlıklı Özetleme ile Parkinson Hastalığı Tespiti

[🇹🇷 Türkçe](#-türkçe) | [🇬🇧 English](#-english)

---

# 🇹🇷 Türkçe

## Proje Hakkında

Çalışmanın amacı, farklı türlerdeki konuşma ve ses kayıtlarından çıkarılan akustik özellikleri kullanarak Parkinson hastalığının makine öğrenmesi ve derin öğrenme yöntemleriyle sınıflandırılmasıdır.

Çalışmada özellikle çoklu ses kayıtlarının özne seviyesinde temsil edilmesi problemi ele alınmış ve aykırı kayıtların etkisini azaltmak amacıyla robust ağırlıklı özetleme yaklaşımı geliştirilmiştir.

Model değerlendirmelerinde veri sızıntısını önlemek ve gerçekçi bir genelleme performansı elde etmek amacıyla Subject-wise Leave-One-Out (s-LOO) yaklaşımı kullanılmıştır.

Proje kapsamında ayrıca geliştirilen modellerin sonuçlarının görüntülenebilmesi için Flask tabanlı bir web uygulaması oluşturulmuştur.

---

## Veri Seti

Projede UCI Machine Learning Repository üzerinde yayımlanan **Parkinson's Speech with Multiple Types of Sound Recordings** veri seti kullanılmıştır.

Veri seti eğitim ve bağımsız test olmak üzere iki bölümden oluşmaktadır.

### Eğitim Seti

- 20 Parkinson hastası (PD)
- 20 sağlıklı kontrol (HC)
- Toplam 40 özne
- Her özne için 26 farklı ses kaydı
- Toplam 1.040 kayıt

Ses kayıtları sürekli ünlüler, sayılar, kelimeler ve kısa cümleler gibi farklı konuşma türlerinden oluşmaktadır.

Her ses kaydından jitter, shimmer, pitch, ses kırılmaları ve benzeri toplam 26 akustik özellik çıkarılmıştır.

### Bağımsız Test Seti

Bağımsız test seti:

- 28 Parkinson hastası
- 168 ses kaydı
- Sürekli `/a/` ve `/o/` ünlü kayıtları

içermektedir.

Bu bireyler eğitim sürecinde kullanılmadığından, bağımsız test seti modelin daha önce görmediği bireyler üzerindeki genelleme performansının değerlendirilmesinde kullanılmıştır.

---

## Uygulanan Yaklaşımlar

Çalışmada farklı özellik temsil ve sınıflandırma yöntemleri karşılaştırılmıştır.

### 1. VIF Sonrası Baseline Model

Akustik özellikler arasındaki çoklu doğrusal bağlantıyı azaltmak amacıyla Variance Inflation Factor (VIF) analizi uygulanmıştır.

Seçilen özellikler özne seviyesinde özetlenmiş ve Support Vector Machine (SVM) tabanlı baseline model oluşturulmuştur.

### 2. Robust Ağırlıklı Özetleme

Çalışmanın temel önerisini oluşturan bu yöntemde, farklı ses kayıtlarının özne seviyesinde daha güvenilir biçimde temsil edilmesi amaçlanmıştır.

Yaklaşımda:

- Median Absolute Deviation (MAD)
- Robust z-score
- Ses tipine bağlı ağırlık katsayıları

kullanılarak aykırı veya tutarsız kayıtların özne temsili üzerindeki etkisi azaltılmıştır.

Elde edilen robust temsil SVM tabanlı sınıflandırıcı ile değerlendirilmiştir.

### 3. Tutarsızlık ve Geçiş Profili

Robust temsilin genişletilmesi amacıyla kayıtlar arasındaki davranış farklılıklarını temsil eden tutarsızlık ve geçiş profili özellikleri oluşturulmuştur.

Bu özelliklerin sınıflandırma performansına katkısı ayrıca incelenmiştir.

### 4. Conv1D + SE Attention

Robust ağırlıklı özetleme sonucunda oluşturulan özne temsili üzerinde derin öğrenme tabanlı bir sınıflandırma yaklaşımı geliştirilmiştir.

Model:

- Conv1D katmanları
- Squeeze-and-Excitation (SE) Attention mekanizması

kullanılarak oluşturulmuştur.


### 5. Ünlü-Odaklı Bağımsız Test Modeli

Bağımsız test setinin yapısına uygun olarak yalnızca sürekli ünlü kayıtlarının kullanıldığı ayrı bir sınıflandırma stratejisi geliştirilmiştir.

Model, eğitim sürecinde görülmemiş 28 Parkinson hastasından oluşan bağımsız test seti üzerinde değerlendirilmiştir.

---

## Model Sonuçları

| Yaklaşım | Accuracy | MCC |
|---|---:|---:|
| VIF Sonrası Baseline | %82,50 | 0,6574 |
| Robust Ağırlıklı Özetleme | **%87,50** | **0,7586** |
| Tutarsızlık / Geçiş Profili | %80,00 | 0,6124 |
| Conv1D + SE Attention | %85,00 | 0,7000 |

Bağımsız test setinde kullanılan ünlü-odaklı final model ise:

**%96,43 doğruluk ve %96,43 duyarlılık**

elde etmiştir.

Sonuçlar, robust ağırlıklı özetleme yaklaşımının klasik baseline modele göre sınıflandırma performansını artırdığını göstermektedir.

---

## Notebooklar

Deneysel süreç aşağıdaki Jupyter Notebook dosyalarında yer almaktadır:

```text
notebooks/
├── 01_Iki_Yeni_Metrik_Ekleme.ipynb
├── 02_Iteratif_VIF_Analizi.ipynb
├── 03_ARI_t_SNE_analizler.ipynb
├── 04_VIF_Sonrasi_Baseline_Model.ipynb
├── 05_Robust_Agirlikli_Ozetleme_Model.ipynb
├── 06_Conv1D_SE_Attention_Model.ipynb
├── 07_Tutarsizlik_Gecis_Profili.ipynb
└── 08_Unlu_Odakli_Bagimsiz_Test_Model.ipynb
```

Notebooklar veri analizi, özellik seçimi, model geliştirme ve model değerlendirme aşamalarını içermektedir.

---

## Web Uygulaması

Çalışmanın modellerini etkileşimli olarak test edebilmek amacıyla Flask tabanlı bir web arayüzü geliştirilmiştir.

Arayüz üzerinden:

- Kullanılacak model seçilebilir.
- Test edilecek özne seçilebilir.
- Model tahmini görüntülenebilir.
- Güven skoru incelenebilir.
- HC ve PD sınıf olasılıkları görüntülenebilir.
- Gerçek sınıf ile model tahmini karşılaştırılabilir.

Web uygulaması araştırma ve eğitim amaçlı geliştirilmiştir.

---

## Proje Yapısı

```text
ParkinsonArayuz/
│
├── model/
│   ├── config_robust.pkl
│   ├── config_vif.pkl
│   ├── conv1d_fold_results.pkl
│   ├── summary_df_conv1d.pkl
│   ├── test_subject_unlu.pkl
│   ├── train_df_robust.pkl
│   ├── train_df_vif.pkl
│   ├── train_subject_unlu.pkl
│   ├── unlu_final_model.pkl
│   ├── unlu_test_results.pkl
│   └── vif_features.pkl
│
├── notebooks/
│   └── Deneysel çalışmalar ve model geliştirme notebookları
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── utils/
│   ├── __init__.py
│   └── predictors.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Kurulum

Projeyi bilgisayarınıza klonladıktan sonra proje dizinine geçin.

```bash
git clone <repository-url>
cd ParkinsonArayuz
```

Sanal ortam oluşturulması önerilir:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Gerekli Python paketlerini yükleyin:

```bash
pip install -r requirements.txt
```

---

## Uygulamayı Çalıştırma

Flask uygulamasını başlatmak için:

```bash
python app.py
```

Uygulama çalıştırıldığında web arayüzü:

```text
http://localhost:5000
```

adresinden kullanılabilir.

---

## Veri Seti Atfı

Bu projede aşağıdaki UCI Machine Learning Repository veri seti kullanılmıştır:

> Kursun, O., Sakar, B., Isenkul, M., Sakar, C., Sertbas, A., & Gurgen, F. (2013).  
> Parkinson's Speech with Multiple Types of Sound Recordings.  
> UCI Machine Learning Repository.  
> DOI: 10.24432/C5NC8M

Veri seti **Creative Commons Attribution 4.0 International (CC BY 4.0)** lisansı altında yayımlanmıştır.

Bu repository içerisindeki veri setinden türetilmiş model ve işlenmiş veri dosyaları, ilgili veri seti kullanılarak gerçekleştirilen akademik çalışmanın çıktılarıdır.

---

## Uyarı

Bu proje yalnızca **akademik araştırma ve eğitim amacıyla** geliştirilmiştir.

Uygulama tarafından üretilen sonuçlar tıbbi tanı olarak değerlendirilmemeli ve profesyonel sağlık hizmetlerinin yerine kullanılmamalıdır.

---

# 🇬🇧 English

## About the Project

This repository contains the experimental code and application developed as part of an undergraduate thesis in the Department of Computer Engineering at Düzce University.

The project investigates the detection of Parkinson's disease using acoustic features extracted from multiple types of speech and voice recordings.

A major focus of the study is the subject-level representation of multiple voice recordings. A robust weighted summarization approach was developed to reduce the influence of anomalous or inconsistent recordings.

To prevent data leakage and obtain a realistic estimate of generalization performance, the models were evaluated using a **Subject-wise Leave-One-Out (s-LOO)** protocol.

A Flask-based web application was also developed to provide an interactive interface for testing and visualizing model predictions.

---

## Dataset

The project uses the **Parkinson's Speech with Multiple Types of Sound Recordings** dataset available from the UCI Machine Learning Repository.

The dataset contains separate training and independent test sets.

### Training Set

- 20 Parkinson's disease (PD) patients
- 20 healthy controls (HC)
- 40 subjects in total
- 26 voice recordings per subject
- 1,040 recordings

The recordings contain different types of speech including sustained vowels, numbers, words and short sentences.

A total of 26 acoustic features, including jitter, shimmer, pitch and voice-break-related measurements, are provided for each recording.

### Independent Test Set

The independent test set contains:

- 28 Parkinson's disease patients
- 168 recordings
- Sustained `/a/` and `/o/` vowel recordings

These subjects were not included during model training, allowing the generalization performance of the proposed approach to be evaluated on previously unseen individuals.

---

## Methodology

Several feature representation and classification approaches were systematically evaluated.

### 1. VIF-Based Baseline Model

Variance Inflation Factor (VIF) analysis was applied to reduce multicollinearity among acoustic features.

The selected features were summarized at the subject level and classified using a Support Vector Machine (SVM).

### 2. Robust Weighted Summarization

The main proposed approach aims to obtain a more reliable subject-level representation from multiple types of voice recordings.

The method combines:

- Median Absolute Deviation (MAD)
- Robust z-scores
- Voice-type-specific weighting coefficients

to reduce the influence of anomalous or inconsistent recordings.

The resulting robust representation was evaluated using an SVM classifier.

### 3. Inconsistency and Transition Profile

Additional inconsistency and transition-profile features were constructed to investigate whether differences between recordings could provide complementary information to the robust representation.

Their contribution to classification performance was evaluated separately.

### 4. Conv1D + SE Attention

A deep-learning-based classification approach was developed using the subject representation generated by the robust weighted summarization method.

The architecture combines:

- One-Dimensional Convolutional Neural Networks (Conv1D)
- Squeeze-and-Excitation (SE) Attention

This experiment evaluates whether the proposed subject representation can also be effectively used with deep learning architectures.

### 5. Vowel-Focused Independent Test Model

A separate classification strategy was developed according to the structure of the independent test set using sustained vowel recordings.

The final model was evaluated on an independent set containing 28 Parkinson's disease patients who were not observed during training.

---

## Results

| Approach | Accuracy | MCC |
|---|---:|---:|
| VIF-Based Baseline | 82.50% | 0.6574 |
| Robust Weighted Summarization | **87.50%** | **0.7586** |
| Inconsistency / Transition Profile | 80.00% | 0.6124 |
| Conv1D + SE Attention | 85.00% | 0.7000 |

The vowel-focused final model achieved:

**96.43% accuracy and 96.43% sensitivity**

on the independent test set.

The experimental results indicate that robust weighted summarization improves classification performance compared with the classical baseline approach.

---

## Notebooks

The experimental workflow is organized into the following Jupyter Notebooks:

```text
notebooks/
├── 01_Iki_Yeni_Metrik_Ekleme.ipynb
├── 02_Iteratif_VIF_Analizi.ipynb
├── 03_ARI_t_SNE_analizler.ipynb
├── 04_VIF_Sonrasi_Baseline_Model.ipynb
├── 05_Robust_Agirlikli_Ozetleme_Model.ipynb
├── 06_Conv1D_SE_Attention_Model.ipynb
├── 07_Tutarsizlik_Gecis_Profili.ipynb
└── 08_Unlu_Odakli_Bagimsiz_Test_Model.ipynb
```

These notebooks cover data analysis, feature selection, model development and model evaluation.

---

## Web Application

A Flask-based web interface was developed to interactively evaluate the models.

The application allows users to:

- Select a classification model
- Select a test subject
- Run a prediction
- View the model confidence score
- Inspect HC and PD class probabilities
- Compare the prediction with the actual class

The application is intended for research and educational purposes.

---

## Project Structure

```text
ParkinsonArayuz/
│
├── model/
│   ├── config_robust.pkl
│   ├── config_vif.pkl
│   ├── conv1d_fold_results.pkl
│   ├── summary_df_conv1d.pkl
│   ├── test_subject_unlu.pkl
│   ├── train_df_robust.pkl
│   ├── train_df_vif.pkl
│   ├── train_subject_unlu.pkl
│   ├── unlu_final_model.pkl
│   ├── unlu_test_results.pkl
│   └── vif_features.pkl
│
├── notebooks/
│   └── Experimental notebooks
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── utils/
│   ├── __init__.py
│   └── predictors.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd ParkinsonArayuz
```

Creating a virtual environment is recommended:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Flask application with:

```bash
python app.py
```

The application can then be accessed at:

```text
http://localhost:5000
```

---

## Dataset Citation

This project uses the following dataset from the UCI Machine Learning Repository:

> Kursun, O., Sakar, B., Isenkul, M., Sakar, C., Sertbas, A., & Gurgen, F. (2013).  
> Parkinson's Speech with Multiple Types of Sound Recordings.  
> UCI Machine Learning Repository.  
> DOI: 10.24432/C5NC8M

The dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

Model artifacts and processed data files included in this repository were derived from this dataset as part of the academic study.

---

## Disclaimer

This project was developed solely for **academic research and educational purposes**.

Predictions produced by the application must not be interpreted as medical diagnoses and should not be used as a substitute for professional medical evaluation.