import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, WhiteKernel
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# ==============================================================================
# 1. KONFIGURASI PARAMETER DAN KREDENSIAL SISTEM
# ==============================================================================

# URL server InfluxDB (Port 8086 dipetakan dari Docker Compose nig-stack)
INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")

# All-Access API Token InfluxDB
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "t6WLmXomN9H9ZFG1RWE6kPakc0bOncdzxUMejfswPJ_8foFVgQsgCPZrzY8DQ7M9inoVymdGEsF4ezw1rjbRsw==")

# Nama Organization dan Bucket (Sesuai dengan Node-RED & InfluxDB Docker)
INFLUX_ORG = os.getenv("INFLUX_ORG", "inforlab")
BUCKET_INPUT = "1RUL-AC-MOTOR"   # Bucket asal tempat penyimpanan fitur dari ESP32
BUCKET_OUTPUT = "1RUL-AC-MOTOR"  # Bucket tujuan untuk menyimpan hasil estimasi RUL
MEASUREMENT_INPUT = "ac_sensor_raw" # Measurement aktual dari Node-RED
MEASUREMENT_OUTPUT = "ac_sensor_prediction"

# Interval waktu pengecekan data baru (dalam detik)
PREDICTION_INTERVAL = 1

# --- Fitur yang dikirim oleh ESP32 melalui Node-RED (5 Fitur Degradasi Utama) ---
PCA_FEATURES = ["v_rms", "crest_factor", "amp_1x", "amp_2x", "suhu"]
RMS_IDX = [PCA_FEATURES.index("v_rms")]

# --- Parameter Health Index (PCA) ---
BASELINE_WINDOWS = 30      # Jumlah data poin awal saat motor SEHAT untuk referensi baseline
PC1_AT_HI_ZERO = 3.0       # Proyeksi PC1 di mana motor dianggap rusak total (HI = 0.0)
HI_FAIL = 0.3              # Ambang batas HI kegagalan (HI <= 0.3 dianggap End-of-Life)

# --- Parameter RUL (Gaussian Process Regression) ---
GPR_MIN_POINTS = 10        # Minimal riwayat HI untuk mengaktifkan regresi GPR
GPR_MAX_POINTS = 150       # Maksimal titik riwayat yang digunakan (subsampling efisien)
RUL_HORIZON_HOURS = 48.0   # Batas horizon ekstrapolasi RUL (jam)
FEATURE_HISTORY_MAX = 5000 # Batas maksimal riwayat fitur di memori

# File state persistent (baseline & riwayat fitur) agar tidak hilang saat restart
STATE_PATH = "ml_pipeline/models/hi_state.joblib"
os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)


# ==============================================================================
# 2. INISIALISASI INFLUXDB CLIENT & LOAD STATE HEALTH INDEX
# ==============================================================================

print("[INIT] Menginisialisasi Client InfluxDB...")
client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()
write_api = client.write_api(write_options=SYNCHRONOUS)

# State internal sistem: baseline fitur sehat, riwayat fitur, dan timestamp
state = {"baseline": None, "features": [], "times": []}

if os.path.exists(STATE_PATH):
    try:
        loaded_state = joblib.load(STATE_PATH)
        # Validasi kecocokan dimensi fitur (16 fitur)
        if loaded_state.get("features") and len(loaded_state["features"][0]) != len(PCA_FEATURES):
            print(f"[WARN] Jumlah fitur pada file state ({len(loaded_state['features'][0])}) tidak cocok dengan konfigurasi baru ({len(PCA_FEATURES)}). State di-reset.")
        else:
            state = loaded_state
            print(f"[INIT] State HI berhasil dimuat: {len(state['features'])} data riwayat tersimpan.")
    except Exception as e:
        print(f"[WARN] Gagal memuat state HI ({e}). Memulai baseline baru.")

if state["baseline"] is None:
    print(f"[INIT] Mengumpulkan {BASELINE_WINDOWS} data awal dari ESP32 untuk membentuk baseline kondisi sehat...")


# ==============================================================================
# 3. FUNGSI HEALTH INDEX (PCA) & PREDIKSI RUL (GAUSSIAN PROCESS)
# ==============================================================================

def compute_health_index(features):
    """
    Fusi 16 fitur sinyal getaran & suhu menjadi skalar Health Index (HI) via PCA.
    """
    eps = 1e-6
    X = np.log((np.abs(np.asarray(features)) + eps) / (np.abs(state["baseline"]) + eps))
    pca = PCA(n_components=1).fit(X)

    # Selaraskan arah PC1 agar naiknya getaran berbanding lurus dengan degradasi
    if np.sum(pca.components_[0][RMS_IDX]) < 0:
        pca.components_ = -pca.components_

    pc1 = X @ pca.components_[0]
    return np.clip(1.0 - pc1 / PC1_AT_HI_ZERO, 0.0, 1.0)


def predict_rul_gpr(times_hours, hi_series):
    """
    Prediksi Remaining Useful Life (RUL) menggunakan Gaussian Process Regression.
    """
    t = np.asarray(times_hours)
    y = np.asarray(hi_series)

    # Gunakan paruh terbaru riwayat agar merefleksikan laju degradasi mutakhir
    start = min(len(t) // 2, max(len(t) - GPR_MIN_POINTS, 0))
    t, y = t[start:], y[start:]

    if len(t) > GPR_MAX_POINTS:
        idx = np.linspace(0, len(t) - 1, GPR_MAX_POINTS).astype(int)
        t, y = t[idx], y[idx]

    # Geser titik awal ke waktu saat ini (t = 0) agar ekstrapolasi stabil
    t_now = t[-1]
    t = t - t_now

    kernel = ConstantKernel(1.0) * DotProduct(sigma_0=1.0) ** 2 + WhiteKernel(noise_level=1e-2)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0)
    gpr.fit(t.reshape(-1, 1), y)

    t_future = np.linspace(0.0, RUL_HORIZON_HOURS, 500)
    mean, std = gpr.predict(t_future.reshape(-1, 1), return_std=True)

    def time_to_threshold(curve):
        hit = np.where(curve <= HI_FAIL)[0]
        return float(t_future[hit[0]]) if len(hit) else RUL_HORIZON_HOURS

    rul = time_to_threshold(mean)
    rul_lower = time_to_threshold(mean - 1.96 * std)  # Skenario pesimis (95% CI)
    rul_upper = time_to_threshold(mean + 1.96 * std)  # Skenario optimis (95% CI)
    return rul, rul_lower, rul_upper


# ==============================================================================
# 4. MAIN LOOP ENGINE PROGNOSIS
# ==============================================================================

def main():
    print("[INFO] Engine Prognosis RUL Real-Time Berjalan. Menunggu data fitur dari ESP32...")
    last_record_time = None

    while True:
        try:
            # Query record fitur terbaru dari measurement ac_sensor_features
            flux_query = f'''
            from(bucket: "{BUCKET_INPUT}")
              |> range(start: -5m)
              |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENT_INPUT}" or r["_measurement"] == "ac_sensor_raw")
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
              |> sort(columns: ["_time"], desc: true)
              |> limit(n: 1)
            '''

            df_result = query_api.query_data_frame(flux_query)
            if isinstance(df_result, list):
                df_result = pd.concat(df_result, ignore_index=True) if df_result else pd.DataFrame()

            if not df_result.empty:
                latest_row = df_result.iloc[0]
                record_time = latest_row['_time']

                # Cek ketersediaan seluruh 16 fitur dalam record
                available_features = [f for f in PCA_FEATURES if f in latest_row and pd.notnull(latest_row[f])]
                missing_features = [f for f in PCA_FEATURES if f not in available_features]

                # DEBUG: Jika ada fitur yang kurang dari Node-RED, beritahu di terminal
                if missing_features:
                    print(f"[WARN] Data masuk di InfluxDB, tapi kurang {len(missing_features)} fitur: {missing_features}")
                    time.sleep(PREDICTION_INTERVAL)
                    continue

                if len(available_features) == len(PCA_FEATURES) and record_time != last_record_time:
                    last_record_time = record_time

                    # Ekstrak vektor fitur 16-dimensi sesuai urutan PCA_FEATURES
                    feature_vector = np.array([float(latest_row[f]) for f in PCA_FEATURES])
                    state["features"].append(feature_vector)
                    state["times"].append(record_time.timestamp())

                    # Batasi ukuran memori riwayat
                    state["features"] = state["features"][-FEATURE_HISTORY_MAX:]
                    state["times"] = state["times"][-FEATURE_HISTORY_MAX:]

                    # --- Step 1: Inisialisasi Baseline Sehat ---
                    if state["baseline"] is None:
                        current_baseline_count = len(state["features"])
                        if current_baseline_count < BASELINE_WINDOWS:
                            print(f"[BASELINE] Mengumpulkan data sehat... ({current_baseline_count}/{BASELINE_WINDOWS})")
                            time.sleep(PREDICTION_INTERVAL)
                            continue
                        
                        # Hitung rata-rata fitur selama periode motor sehat
                        state["baseline"] = np.mean(state["features"][:BASELINE_WINDOWS], axis=0)
                        
                        # Pastikan folder direktori state ada sebelum disimpan
                        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
                        joblib.dump(state, STATE_PATH)
                        print("[BASELINE] Baseline kondisi motor sehat berhasil terbentuk!")

                    # --- Step 2: Hitung Health Index (PCA) ---
                    hi_series = compute_health_index(state["features"])
                    hi_now = float(hi_series[-1])

                    # --- Step 3: Estimasi RUL (Gaussian Process Regression) ---
                    times_hours = (np.asarray(state["times"]) - state["times"][0]) / 3600.0
                    if hi_now <= HI_FAIL:
                        rul, rul_lo, rul_hi = 0.0, 0.0, 0.0
                    elif len(hi_series) >= GPR_MIN_POINTS:
                        rul, rul_lo, rul_hi = predict_rul_gpr(times_hours, hi_series)
                    else:
                        rul, rul_lo, rul_hi = RUL_HORIZON_HOURS, RUL_HORIZON_HOURS, RUL_HORIZON_HOURS

                    # Cetak hasil prognosis ke terminal
                    print(f"[PROGNOSIS] Suhu: {latest_row['suhu']:.1f}°C | "
                          f"V_RMS: {latest_row['v_rms']:.3f} mm/s | CF: {latest_row['crest_factor']:.2f} | "
                          f"HI: {hi_now:.3f} | RUL: {rul:.2f} Jam [{rul_lo:.2f} - {rul_hi:.2f}]")

                    # --- Step 4: Write-Back Hasil Prediksi ke InfluxDB ---
                    point = Point(MEASUREMENT_OUTPUT) \
                        .field("health_index", hi_now) \
                        .field("rul_hours", float(rul)) \
                        .field("rul_lower_hours", float(rul_lo)) \
                        .field("rul_upper_hours", float(rul_hi)) \
                        .field("rul_percentage", hi_now * 100.0) \
                        .field("health_status", 1 if hi_now > HI_FAIL else 0)

                    # Simpan juga seluruh metrik fitur untuk kebutuhan visualisasi detail di Grafana
                    for f in PCA_FEATURES:
                        point = point.field(f, float(latest_row[f]))

                    write_api.write(bucket=BUCKET_OUTPUT, org=INFLUX_ORG, record=point)
                    joblib.dump(state, STATE_PATH)

            else:
                # Menampilkan log berkala jika belum ada data sama sekali dalam rentang waktu query
                print(f"[WAIT] Menunggu data masuk di bucket '{BUCKET_INPUT}' (measurement: '{MEASUREMENT_INPUT}')...", end="\r")

        except Exception as error:
            print(f"\n[ERROR] Kesalahan dalam pipeline prognosis: {error}")

        time.sleep(PREDICTION_INTERVAL)

if __name__ == "__main__":
    main()