import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px

# ==============================
# Konfigurasi halaman
# ==============================
st.set_page_config(
    page_title="Sistem Klasifikasi Sampah",
    page_icon="♻️",
    layout="wide"
)

# ==============================
# Class names
# ==============================
CLASS_NAMES = ['anorganik', 'b3', 'organik']

CLASS_EMOJIS = {
    'anorganik': '🔩',
    'organik': '🥝',
    'b3': '😷'
}

# ==============================
# Load model (FULL MODEL .keras)
# ==============================
@st.cache_resource
def load_trained_model(model_path):
    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        # compile ulang hanya untuk keperluan inference
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# ==============================
# Preprocess image
# ==============================
def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image).astype(np.float32)

    # Normalisasi (samakan dengan training kamu jika pakai rescale 1./255)
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ==============================
# Prediction
# ==============================
def predict_image(model, image):

    if model is None:
        return None, None, None

    try:
        processed_img = preprocess_image(image)
        predictions = model.predict(processed_img, verbose=0)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        all_predictions = predictions[0]

        return predicted_class, confidence, all_predictions

    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")
        return None, None, None


# ==============================
# Chart
# ==============================
def create_prediction_chart(predictions, class_names):

    df = pd.DataFrame({
        'Kelas': class_names,
        'Probabilitas': predictions,
        'Emoji': [CLASS_EMOJIS[cls] for cls in class_names]
    })

    df = df.sort_values('Probabilitas', ascending=True)

    fig = px.bar(
        df,
        x='Probabilitas',
        y='Kelas',
        orientation='h',
        title='Probabilitas untuk Setiap Kelas Sampah',
        color='Probabilitas',
        color_continuous_scale='viridis',
        text='Probabilitas'
    )

    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(
        height=400,
        xaxis_tickformat='.0%',
        showlegend=False
    )

    return fig


# ==============================
# Informasi sampah
# ==============================
def display_waste_info(predicted_class):

    waste_info = {
        'anorganik': {
            'deskripsi': 'Sampah yang tidak mudah terurai secara alami, terutama dari bahan logam, plastik, dan kaca yang bisa didaur ulang.',
            'cara_pemrosesan': 'Dapat diproses kembali menjadi produk baru, seperti logam daur ulang atau plastik yang bisa digunakan kembali.',
            'contoh': 'Kaleng, paku, kawat, peralatan logam, botol plastik, kemasan plastik.',
            'warna_tempat': 'Kuning',
            'dampak': 'Sampah anorganik membutuhkan waktu lama untuk terurai dan dapat menyebabkan pencemaran lingkungan jika tidak didaur ulang.'
        },
        'organik': {
            'deskripsi': 'Sampah yang berasal dari bahan alami yang dapat terurai secara alami, seperti sisa makanan dan daun.',
            'cara_pemrosesan': 'Dapat diolah menjadi kompos atau pupuk organik yang berguna untuk pertanian.',
            'contoh': 'Sisa makanan, daun, sayuran, buah-buahan.',
            'warna_tempat': 'Hijau',
            'dampak': 'Sampah organik dapat terurai dalam waktu singkat, tetapi jika tidak dikelola dengan baik, bisa menghasilkan gas metana.'
        },
        'b3': {
            'deskripsi': 'Sampah yang memiliki potensi besar mencemari tanah dan air serta membahayakan kesehatan.',
            'cara_pemrosesan': 'Diolah secara khusus oleh pihak berizin.',
            'contoh': 'Baterai, bahan kimia, limbah rumah sakit.',
            'warna_tempat': 'Merah (B3)',
            'dampak': 'Mencemari tanah dan air serta membahayakan kesehatan manusia.'
        }
    }

    if predicted_class in waste_info:

        info = waste_info[predicted_class]

        st.markdown(f"""
        ### {CLASS_EMOJIS[predicted_class]} Informasi Sampah: {predicted_class.title()}

        **Deskripsi:** {info['deskripsi']}

        **Cara Pemrosesan:** {info['cara_pemrosesan']}

        **Contoh:** {info['contoh']}

        **Tempat Sampah:** {info['warna_tempat']}

        **Dampak Lingkungan:** {info['dampak']}
        """)


# ==============================
# MAIN APP
# ==============================
def main():

    st.title("♻️ Sistem Klasifikasi Sampah")
    st.markdown("---")

    st.sidebar.title("📋 Menu")
    page = st.sidebar.selectbox(
        "Pilih Halaman:",
        ["🔍 Klasifikasi Gambar", "📊 Informasi Dataset", "ℹ️ Tentang Sistem"]
    )

    # ==============================
    # Halaman klasifikasi
    # ==============================
    if page == "🔍 Klasifikasi Gambar":

        st.header("🔍 Klasifikasi Gambar Sampah")

        model_path = st.sidebar.text_input(
            "Path Model (.keras):",
            value="best_model4.keras"
        )

        model = load_trained_model(model_path) if model_path else None

        if model is None:
            st.warning("⚠️ Model belum dimuat. Silakan masukkan path model yang valid.")
            return

        st.success("✅ Model berhasil dimuat!")

        uploaded_file = st.file_uploader(
            "Upload gambar sampah:",
            type=['png', 'jpg', 'jpeg']
        )

        if uploaded_file is not None:

            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📸 Gambar yang Diupload")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("🤖 Hasil Prediksi")

                with st.spinner("Memproses gambar..."):
                    predicted_class, confidence, all_predictions = predict_image(model, image)

                if predicted_class is not None:

                    st.metric(
                        label="Jenis Sampah",
                        value=f"{CLASS_EMOJIS[predicted_class]} {predicted_class.title()}",
                        delta=f"Confidence: {confidence:.1%}"
                    )

                    st.progress(confidence)

                    if confidence > 0.7:
                        st.success(f"✅ Prediksi dengan confidence tinggi ({confidence:.1%})")
                    elif confidence > 0.5:
                        st.warning(f"⚠️ Prediksi dengan confidence sedang ({confidence:.1%})")
                    else:
                        st.error(f"❌ Prediksi dengan confidence rendah ({confidence:.1%})")

            if all_predictions is not None:
                st.subheader("📊 Detail Probabilitas")
                fig = create_prediction_chart(all_predictions, CLASS_NAMES)
                st.plotly_chart(fig, use_container_width=True)

            if predicted_class is not None:
                st.markdown("---")
                display_waste_info(predicted_class)

    # ==============================
    # Dataset
    # ==============================
    elif page == "📊 Informasi Dataset":

        st.header("📊 Informasi Dataset")

        class_info = pd.DataFrame({
            'Kelas': CLASS_NAMES,
            'Emoji': [CLASS_EMOJIS[cls] for cls in CLASS_NAMES],
            'Deskripsi': [
                'Sampah anorganik (logam, plastik, kaca)',
                'Sampah bahan berbahaya dan beracun (B3)',
                'Sampah organik dari bahan alami'
            ]
        })

        st.dataframe(class_info, use_container_width=True)

        st.subheader("📈 Distribusi Dataset")

        sample_data = {
            'Kelas': CLASS_NAMES,
            'Training': [1440, 1440, 1440],
            'Test': [180, 180, 180],
            'Validation': [180, 180, 180]
        }

        df = pd.DataFrame(sample_data)

        fig = px.bar(
            df.melt(id_vars=['Kelas'], value_vars=['Training', 'Test', 'Validation']),
            x='Kelas',
            y='value',
            color='variable',
            title='Distribusi Data'
        )

        st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # Tentang sistem
    # ==============================
    elif page == "ℹ️ Tentang Sistem":

        st.header("ℹ️ Tentang Sistem Klasifikasi Sampah")

        st.markdown("""
        Sistem ini dibuat untuk membantu klasifikasi sampah organik,
        anorganik, dan bahan berbahaya dan beracun (B3)
        menggunakan Convolutional Neural Network (CNN)
        berbasis arsitektur MobileNetV3.
        """)


if __name__ == "__main__":
    main()
