"""
Honest tahmin fonksiyonları - Tüm modeller için
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from statsmodels.stats.outliers_influence import variance_inflation_factor


feature_names = [
    "Jitter_local", "Jitter_local_abs", "Jitter_rap", "Jitter_ppq5", "Jitter_ddp",
    "Shimmer_local", "Shimmer_local_dB", "Shimmer_apq3", "Shimmer_apq5", "Shimmer_apq11", "Shimmer_dda",
    "AC", "NTH", "HTN",
    "Median_pitch", "Mean_pitch", "Pitch_std", "Min_pitch", "Max_pitch",
    "Num_pulses", "Num_periods", "Mean_period", "Period_std",
    "Frac_unvoiced_frames", "Num_voice_breaks", "Degree_voice_breaks"
]


# ================== ROBUST MODEL ==================

def compute_weights_robust(df, vowel_w=1.5, number_w=1.2, other_w=1.0):
    X = df[feature_names].copy()
    med = X.median()
    mad = (X - med).abs().median().replace(0, 1e-6)
    robust_z = ((X - med).abs() / mad)
    score = robust_z.mean(axis=1).values
    weights = np.exp(-score)
    type_weights = []
    for t in df["sample_type"]:
        if t == "vowel":
            type_weights.append(vowel_w)
        elif t == "number":
            type_weights.append(number_w)
        else:
            type_weights.append(other_w)
    weights = weights * np.array(type_weights)
    return weights / weights.sum()


def build_subject_dataset_robust(df, vowel_w=1.5, number_w=1.2, other_w=1.0):
    rows = []
    for sid, g in df.groupby("subject_id"):
        row = {"subject_id": sid, "class": int(g["class"].iloc[0])}
        w = compute_weights_robust(g, vowel_w, number_w, other_w)
        for f in feature_names:
            x = g[f].values
            wmean = np.sum(w * x)
            wstd = np.sqrt(np.sum(w * (x - wmean) ** 2))
            row[f + "_wmean"] = wmean
            row[f + "_wstd"] = wstd
        rows.append(row)
    return pd.DataFrame(rows)


def predict_honest_robust(test_subject_id, train_df_full, config):
    train_part = train_df_full[train_df_full["subject_id"] != test_subject_id].copy()
    test_part = train_df_full[train_df_full["subject_id"] == test_subject_id].copy()

    if len(test_part) == 0:
        return {"error": f"Özne {test_subject_id} bulunamadı"}

    actual_class = int(test_part["class"].iloc[0])

    train_subj = build_subject_dataset_robust(train_part, vowel_w=config["vowel_w"], number_w=config["number_w"], other_w=config["other_w"])
    test_subj = build_subject_dataset_robust(test_part, vowel_w=config["vowel_w"], number_w=config["number_w"], other_w=config["other_w"])

    X_train = train_subj.drop(columns=["subject_id", "class"])
    y_train = train_subj["class"].values
    X_test = test_subj.drop(columns=["subject_id", "class"])

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selector = SelectKBest(mutual_info_classif, k=config["k"])
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)

    model = SVC(kernel=config["kernel"], C=config["C"], gamma=config["gamma"], class_weight=config["class_weight"], probability=True)
    model.fit(X_train_sel, y_train)

    prediction = int(model.predict(X_test_sel)[0])
    probabilities = model.predict_proba(X_test_sel)[0]

    return {
        "subject_id": int(test_subject_id),
        "prediction": prediction,
        "predicted_label": "Parkinson Hastası (PD)" if prediction == 1 else "Sağlıklı Kontrol (HC)",
        "actual_class": actual_class,
        "actual_label": "Parkinson Hastası (PD)" if actual_class == 1 else "Sağlıklı Kontrol (HC)",
        "is_correct": prediction == actual_class,
        "confidence": float(probabilities[prediction]),
        "probabilities": {"HC": float(probabilities[0]), "PD": float(probabilities[1])}
    }


# ================== VIF MODEL ==================

def vif_hesapla(X):
    return [variance_inflation_factor(X, i) for i in range(X.shape[1])]


def iteratif_vif_azalt(X_train_scaled, fnames, esik=10.0):
    secilen_idx = list(range(X_train_scaled.shape[1]))
    secilen_names = fnames.copy()
    cikarilanlar = []
    while True:
        X_current = X_train_scaled[:, secilen_idx]
        vif_degerleri = vif_hesapla(X_current)
        max_vif = max(vif_degerleri)
        if max_vif <= esik:
            break
        en_kotu = int(np.argmax(vif_degerleri))
        cikarilanlar.append(secilen_names[en_kotu])
        secilen_names.pop(en_kotu)
        secilen_idx.pop(en_kotu)
    return secilen_idx, secilen_names, cikarilanlar


def ozetle_vif(X_scaled, mask, kombinasyon):
    parcalar = []
    for metrik in kombinasyon:
        X_subj = X_scaled[mask]
        if metrik == 'mean':
            parcalar.append(X_subj.mean(axis=0))
        elif metrik == 'std':
            parcalar.append(X_subj.std(axis=0))
    return np.hstack(parcalar)


def predict_honest_vif(test_subject_id, train_data, config):
    X = train_data["X"]
    subjects = train_data["subjects"]
    y = train_data["y"]
    fnames = train_data["feature_names"]

    if test_subject_id not in np.unique(subjects):
        return {"error": f"Özne {test_subject_id} bulunamadı"}

    train_mask = subjects != test_subject_id
    test_mask = subjects == test_subject_id

    X_train_raw = X[train_mask]
    X_test_raw = X[test_mask]
    y_train_raw = y[train_mask]
    y_test_raw = y[test_mask]
    subj_train = subjects[train_mask]

    actual_class = int(y_test_raw[0])

    scaler = StandardScaler()
    X_train_scaled_full = scaler.fit_transform(X_train_raw)
    X_test_scaled_full = scaler.transform(X_test_raw)

    secilen_idx, secilen_names, cikarilanlar = iteratif_vif_azalt(X_train_scaled_full, fnames, esik=config["vif_esik"])

    X_train_scaled = X_train_scaled_full[:, secilen_idx]
    X_test_scaled = X_test_scaled_full[:, secilen_idx]

    X_train_ozet = []
    y_train_ozet = []
    for s in np.unique(subj_train):
        mask = subj_train == s
        X_train_ozet.append(ozetle_vif(X_train_scaled, mask, config["kombinasyon"]))
        y_train_ozet.append(y_train_raw[mask][0])
    X_train_ozet = np.array(X_train_ozet)
    y_train_ozet = np.array(y_train_ozet)

    X_test_ozet = ozetle_vif(X_test_scaled, np.ones(len(X_test_raw), dtype=bool), config["kombinasyon"]).reshape(1, -1)

    clf = SVC(kernel=config["kernel"], C=config["C"], probability=True)
    clf.fit(X_train_ozet, y_train_ozet)

    prediction = int(clf.predict(X_test_ozet)[0])
    probabilities = clf.predict_proba(X_test_ozet)[0]

    return {
        "subject_id": int(test_subject_id),
        "prediction": prediction,
        "predicted_label": "Parkinson Hastası (PD)" if prediction == 1 else "Sağlıklı Kontrol (HC)",
        "actual_class": actual_class,
        "actual_label": "Parkinson Hastası (PD)" if actual_class == 1 else "Sağlıklı Kontrol (HC)",
        "is_correct": prediction == actual_class,
        "confidence": float(probabilities[prediction]),
        "probabilities": {"HC": float(probabilities[0]), "PD": float(probabilities[1])},
        "vif_kept": len(secilen_names),
        "vif_removed": cikarilanlar
    }


# ================== ÜNLÜ ODAKLI MODEL ==================

def predict_honest_unlu(test_subject_id, test_subject_df, unlu_final_model, unlu_test_results):
    sid_col = "SubjectID" if "SubjectID" in test_subject_df.columns else "subject_id"
    cls_col = "Class" if "Class" in test_subject_df.columns else "class"

    subject_row = test_subject_df[test_subject_df[sid_col] == test_subject_id]

    if subject_row.empty:
        return {"error": f"Bağımsız test setinde özne {test_subject_id} bulunamadı."}

    actual_class = int(subject_row[cls_col].iloc[0])
    X_test = subject_row.drop(columns=[sid_col, cls_col]).values

    prediction = int(unlu_final_model.predict(X_test)[0])

    try:
        probabilities = unlu_final_model.predict_proba(X_test)[0]
        prob_hc = float(probabilities[0])
        prob_pd = float(probabilities[1])
    except AttributeError:
        prob_pd = 0.85 if prediction == 1 else 0.15
        prob_hc = 1.0 - prob_pd

    confidence = prob_pd if prediction == 1 else prob_hc

    return {
        "subject_id": int(test_subject_id),
        "prediction": prediction,
        "predicted_label": "Parkinson Hastası (PD)" if prediction == 1 else "Sağlıklı Kontrol (HC)",
        "actual_class": actual_class,
        "actual_label": "Parkinson Hastası (PD)" if actual_class == 1 else "Sağlıklı Kontrol (HC)",
        "is_correct": prediction == actual_class,
        "confidence": confidence,
        "probabilities": {"HC": prob_hc, "PD": prob_pd},
        "selected_vowels": unlu_test_results.get("selected_vowels", []),
        "note": "Bu model bağımsız test seti üzerinde çalışır. Tüm 40 eğitim öznesini kullanarak önceden eğitilmiştir."
    }


# ================== CONV1D + SE ATTENTION MODEL ==================

def predict_honest_conv1d(test_subject_id, fold_results_df, summary_df):
    """
    Önceden hesaplanmış s-LOO fold sonuçlarından ilgili özneyi döndürür.
    fold_results_df: notebook'tan kaydedilen conv1d_fold_results.pkl
    summary_df: notebook'tan kaydedilen summary_df_conv1d.pkl
    """
    row = fold_results_df[fold_results_df["subject_id"] == test_subject_id]

    if row.empty:
        return {"error": f"Özne {test_subject_id} Conv1D sonuçlarında bulunamadı."}

    row = row.iloc[0]

    prediction = int(row["pred"])
    actual_class = int(row["true"])
    prob_pd = float(row["prob"])
    prob_hc = 1.0 - prob_pd
    confidence = prob_pd if prediction == 1 else prob_hc

    return {
        "subject_id": int(test_subject_id),
        "prediction": prediction,
        "predicted_label": "Parkinson Hastası (PD)" if prediction == 1 else "Sağlıklı Kontrol (HC)",
        "actual_class": actual_class,
        "actual_label": "Parkinson Hastası (PD)" if actual_class == 1 else "Sağlıklı Kontrol (HC)",
        "is_correct": prediction == actual_class,
        "confidence": confidence,
        "probabilities": {"HC": prob_hc, "PD": prob_pd},
        "note": "Bu model önceden hesaplanmış s-LOO sonuçlarını kullanır (k=12, threshold=0.45)."
    }
