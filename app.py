"""
Parkinson Tespit Sistemi - Flask Web Arayüzü
PyInstaller uyumlu sürüm
"""

from flask import Flask, render_template, request
import joblib
import os
import sys
import webbrowser
import threading

from utils.predictors import (
    predict_honest_robust,
    predict_honest_vif,
    predict_honest_unlu,
    predict_honest_conv1d
)


def resource_path(relative_path):
    """
    Normal Python çalıştırmasında ve PyInstaller exe çalıştırmasında
    dosya yollarını doğru bulmak için kullanılır.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


app = Flask(
    __name__,
    template_folder=resource_path("templates"),
    static_folder=resource_path("static")
)

MODEL_DIR = resource_path("model")

print("Model dosyaları yükleniyor...")


# Robust Model
train_df_robust = joblib.load(
    os.path.join(MODEL_DIR, "train_df_robust.pkl")
)

config_robust = joblib.load(
    os.path.join(MODEL_DIR, "config_robust.pkl")
)

print("✓ Robust modeli yüklendi")


# VIF Model
train_df_vif = joblib.load(
    os.path.join(MODEL_DIR, "train_df_vif.pkl")
)

config_vif = joblib.load(
    os.path.join(MODEL_DIR, "config_vif.pkl")
)

print("✓ VIF modeli yüklendi")


# Ünlü Odaklı Model
test_subject_unlu = joblib.load(
    os.path.join(MODEL_DIR, "test_subject_unlu.pkl")
)

unlu_final_model = joblib.load(
    os.path.join(MODEL_DIR, "unlu_final_model.pkl")
)

unlu_test_results = joblib.load(
    os.path.join(MODEL_DIR, "unlu_test_results.pkl")
)

print("✓ Ünlü odaklı model yüklendi")


# Conv1D Model
conv1d_fold_results = joblib.load(
    os.path.join(MODEL_DIR, "conv1d_fold_results.pkl")
)

summary_df_conv1d = joblib.load(
    os.path.join(MODEL_DIR, "summary_df_conv1d.pkl")
)

print("✓ Conv1D modeli yüklendi")


AVAILABLE_MODELS = {
    "robust": {
        "name": "Robust Ağırlıklı Özetleme",
        "accuracy": 87.50,
        "mcc": 0.7586,
        "sensitivity": 80.00,
        "specificity": 95.00,
        "description": (
            "26 farklı ses kayıt türünü kayıt kalitesi ve "
            "ses tipi katsayılarına göre ağırlıklandırarak özetleyen "
            "SVM tabanlı sınıflandırma yaklaşımı."
        ),
        "available": True,
        "dataset": "train"
    },

    "vif": {
        "name": "VIF Sonrası Baseline",
        "accuracy": 82.50,
        "mcc": 0.6574,
        "sensitivity": 90.00,
        "specificity": 75.00,
        "description": (
            "Çoklu doğrusallığın VIF analizi ile azaltıldığı "
            "Linear SVM tabanlı baseline sınıflandırma modeli."
        ),
        "available": True,
        "dataset": "train"
    },

    "conv1d": {
        "name": "Conv1D + SE Attention",
        "accuracy": 85.00,
        "mcc": 0.7000,
        "sensitivity": 85.00,
        "specificity": 85.00,
        "description": (
            "Conv1D ve Squeeze-and-Excitation Attention "
            "mekanizmasını kullanan derin öğrenme modeli."
        ),
        "available": True,
        "dataset": "train"
    },

    "unlu": {
        "name": "Ünlü-Odaklı Model",
        "accuracy": unlu_test_results.get("accuracy", 0) * 100,
        "mcc": None,
        "sensitivity": unlu_test_results.get("sensitivity", 0) * 100,
        "specificity": None,
        "description": (
            "Sürekli ünlü ses kayıtlarını kullanan SVM tabanlı model. "
            "Bağımsız test seti üzerinde değerlendirilmiştir. "
            f"Seçilen ünlüler: "
            f"{'+'.join(unlu_test_results.get('selected_vowels', []))}."
        ),
        "available": True,
        "dataset": "test"
    }
}


# Özne listeleri
ROBUST_SUBJECTS = sorted(
    train_df_robust["subject_id"].unique().tolist()
)

VIF_SUBJECTS = sorted(
    [int(s) for s in set(train_df_vif["subjects"].tolist())]
)

_sid_col = (
    "SubjectID"
    if "SubjectID" in test_subject_unlu.columns
    else "subject_id"
)

UNLU_SUBJECTS = sorted(
    test_subject_unlu[_sid_col].unique().tolist()
)

CONV1D_SUBJECTS = sorted(
    summary_df_conv1d["subject_id"].unique().tolist()
)


@app.route("/")
def index():
    return render_template(
        "index.html",
        models=AVAILABLE_MODELS,
        robust_subjects=ROBUST_SUBJECTS,
        vif_subjects=VIF_SUBJECTS,
        unlu_subjects=UNLU_SUBJECTS,
        conv1d_subjects=CONV1D_SUBJECTS
    )


@app.route("/predict", methods=["POST"])
def predict():

    model_choice = request.form.get("model")
    subject_id_raw = request.form.get("subject_id")

    if model_choice not in AVAILABLE_MODELS:
        return render_template(
            "result.html",
            error="Bilinmeyen model seçimi.",
            model_info=None
        )

    try:
        subject_id = int(subject_id_raw)

        if model_choice == "robust":
            result = predict_honest_robust(
                subject_id,
                train_df_robust,
                config_robust
            )

        elif model_choice == "vif":
            result = predict_honest_vif(
                subject_id,
                train_df_vif,
                config_vif
            )

        elif model_choice == "unlu":
            result = predict_honest_unlu(
                subject_id,
                test_subject_unlu,
                unlu_final_model,
                unlu_test_results
            )

        elif model_choice == "conv1d":
            result = predict_honest_conv1d(
                subject_id,
                conv1d_fold_results,
                summary_df_conv1d
            )

        else:
            return render_template(
                "result.html",
                error="Bilinmeyen model.",
                model_info=None
            )

        if "error" in result:
            return render_template(
                "result.html",
                error=result["error"],
                model_info=AVAILABLE_MODELS[model_choice]
            )

        return render_template(
            "result.html",
            result=result,
            model_info=AVAILABLE_MODELS[model_choice],
            subject_id=subject_id
        )

    except (TypeError, ValueError):
        return render_template(
            "result.html",
            error="Geçerli bir özne seçiniz.",
            model_info=AVAILABLE_MODELS.get(model_choice)
        )

    except Exception as e:
        return render_template(
            "result.html",
            error=f"Tahmin sırasında hata oluştu: {str(e)}",
            model_info=AVAILABLE_MODELS.get(model_choice)
        )


def open_browser():
    """1.5 saniye gecikme ile tarayıcıyı otomatik aç."""
    webbrowser.open_new("http://localhost:5000")


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("PARKİNSON TESPİT SİSTEMİ")
    print("Tarayıcı otomatik açılıyor: http://localhost:5000")
    print("Durdurmak için: Ctrl+C")
    print("=" * 60 + "\n")

    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        threading.Timer(1.5, open_browser).start()

    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000
    )