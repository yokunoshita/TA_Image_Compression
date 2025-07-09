import streamlit as st
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
import io
import os

st.set_page_config(page_title="Kompresi Gambar dengan MiniBatchKMeans", layout="wide")
st.title("Kompresi Gambar Digital dengan Ekstraksi Palet Warna")

uploaded_file = st.file_uploader("Unggah gambar (bisa drag & drop)", type=["jpg", "jpeg"], accept_multiple_files=False)

# Slider
k = st.slider("Jumlah warna (k)", min_value=2, max_value=128, value=96)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    st.image(image, caption="Gambar berhasil diunggah !", use_column_width=False)

    if st.button("Mulai Kompresi"):
        img_np = np.array(image)

        original_buffer = io.BytesIO()
        image.save(original_buffer, format="JPEG")
        original_size = len(original_buffer.getvalue())

        w, h = image.size
        img_flat = img_np.reshape((-1, 3))

        st.info("Melakukan clustering dengan MiniBatchKMeans...")
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=100)
        labels = kmeans.fit_predict(img_flat)
        quant = kmeans.cluster_centers_.astype("uint8")[labels]
        quant_img = quant.reshape((img_np.shape))

        result_image = Image.fromarray(quant_img)

        mse = mean_squared_error(img_np, quant_img)
        psnr = peak_signal_noise_ratio(img_np, quant_img, data_range=255)

        result_buffer = io.BytesIO()
        result_image.save(result_buffer, format="JPEG", quality=50)
        compressed_size = len(result_buffer.getvalue())

        compression_ratio = compressed_size / original_size * 100

### Visualisasi
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔍 Gambar Asli")
            st.image(image, use_column_width=True)
            st.caption(f"Ukuran: {original_size / 1024:.2f} KB")
            # with st.expander("Zoom Gambar Asli"):
            #     st.image(image, use_column_width=False)

        with col2:
            st.subheader("📉 Gambar Setelah Kompresi")
            st.image(result_image, use_column_width=True)
            st.caption(f"Ukuran: {compressed_size / 1024:.2f} KB ({compression_ratio:.2f}% dari ukuran asli)")
            # with st.expander("Zoom Gambar Kompresi"):
            #     st.image(result_image, use_column_width=False)

        st.success(f"🔧 Kompresi selesai! Gambar berhasil dikurangi menjadi {k} warna.")

        st.markdown("### 📈 Hasil Evaluasi Kualitas")
        col_mse, col_psnr = st.columns(2)
        with col_mse:
            st.metric(label="📏 MSE (Mean Squared Error)", value=f"{mse:.2f}")
        with col_psnr:
            st.metric(label="🔊 PSNR (Peak Signal-to-Noise Ratio)", value=f"{psnr:.2f} dB")

        st.download_button(
            label="💾 Unduh Gambar Hasil Kompresi",
            data=result_buffer.getvalue(),
            file_name="compressed_image.jpg",
            mime="image/jpg"
        )
