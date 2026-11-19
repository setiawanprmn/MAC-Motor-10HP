/*
 * Edge Monitor Pompa 10 HP (7.5 kW) v2.1
 * Sensor: MPU-6050 (I2C) + SCT-013 (ADC1)
 * Komunikasi: Ethernet W5500 (SPI, DHCP) -> HTTP Client ke Node-RED (Port 1880)
 *
 * Alokasi Pin:
 *  - MPU-6050 : I2C SDA = GPIO 21, SCL = GPIO 22
 *  - SCT-013  : Analog Input = GPIO 34 (ADC1_CH6)
 *  - W5500    : SPI SCK = GPIO 18, MISO = GPIO 19, MOSI = GPIO 23, CS = GPIO 5
 */

#include <SPI.h>
#include <Ethernet.h>
#include <Wire.h>
#include "arduinoFFT.h"

// ====================================================
// 1. KONFIGURASI ETHERNET W5500 & NODE-RED
// ====================================================
byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED };

// Tujuan Node-RED (IP Laptop/PC Linux)
IPAddress serverIP(10, 176, 217, 179);
const int serverPort   = 1880;
const char* serverPath = "/kirim-data";

// ====================================================
// 2. KONFIGURASI SENSOR & FFT
// ====================================================
#define W5500_CS     5                        // Pin CS modul Ethernet W5500 (VSPI)
#define SCT_PIN      34                       // SCT-013 di GPIO 34 (ADC1_CH6)
#define SAMPLES      4096                     // 2^12, jendela ~4,1 s
#define HOP          (SAMPLES / 2)            // Overlap 50% (~2 detik pembaruan)
#define RING_SIZE    (SAMPLES + 256)          // Margin aman snapshot

const float FS = 1000.0f;                     // Hz
const float DF = FS / SAMPLES;                // ~0,244 Hz

// Konfigurasi Kalibrasi SCT-013
// - Gunakan 30.0f untuk SCT-013-030 (30A / 1V internal burden)
// - Gunakan 60.6f untuk SCT-013-000 (100A / 50mA dengan R_burden 33 Ohm)
const float SCT_ICAL         = 30.0f;
const float CURRENT_NOISE_TH = 0.25f;         // Cut-off noise ADC saat motor mati (Ampere)
const float CURRENT_NOMINAL  = 15.0f;         // Arus nominal motor 7.5 kW (3-Fasa ~14.5 - 15.5 A)
const float CURRENT_OVERLOAD = 16.5f;         // Batas indikasi beban berlebih (+10% nominal)

// MPU-6050 Register Direct Access
const uint8_t MPU_ADDR      = 0x68;
const float   ACC_LSB_PER_G = 8192.0f;        // Rentang +-4 g
const float   GRAV          = 9.80665f;

// Parameter Analisis Frekuensi
const float F_1X_NOMINAL = 24.15f;            // Hz (putaran motor ~1450 RPM)
const float F_1X_SEARCH  = 3.0f;              // Jendela pencarian 1x
const float BAND_LO_HZ   = 10.0f;             // Batas bawah ISO 10816
const float BAND_HI_HZ   = 200.0f;            // Batas atas bandwidth
const float V_IDLE       = 0.4f;              // Batas deteksi mesin mati (mm/s)

// Ambang Batas Zona ISO 10816-7 Cat 2 / 10816-3 (mm/s RMS)
const float ZONE_A_MAX = 1.4f;
const float ZONE_B_MAX = 2.8f;
const float ZONE_C_MAX = 4.5f;

// Ambang Diagnostik Mekanis
const float AMP2X_WARN     = 1.4f;            // mm/s RMS (komponen 2x kopling FCL)
const float LOOSENESS_WARN = 0.40f;           // Rasio harmonisa kelonggaran baut

// ====================================================
// 3. OBJEK & DATA GLOBAL
// ====================================================
float vReal[SAMPLES];
float vImag[SAMPLES];
float velSpec[SAMPLES / 2];                   // Spektrum kecepatan puncak (mm/s)
ArduinoFFT<float> FFT = ArduinoFFT<float>(vReal, vImag, SAMPLES, FS);

int16_t ring[RING_SIZE];
portMUX_TYPE ringMux = portMUX_INITIALIZER_UNLOCKED;
volatile uint32_t widx        = 0;
volatile uint32_t sampleCount = 0;
volatile uint32_t i2cErrors   = 0;

bool     mpuReady = false;
uint32_t seq      = 0;

char latestJson[448];

struct Telemetry {
  float i_rms, v_rms, f_1x, amp_1x, f_2x, amp_2x, crest;
  int   health_index;
  char  iso_zone[8];
  char  kondisi[20];
  char  dugaan[28];
};
Telemetry liveData = {0, 0, 0, 0, 0, 0, 1.41, 100, "-", "MENGUMPULKAN_DATA", "-"};

// ====================================================
// 4. AKSES MPU-6050 & SCT-013
// ====================================================
bool mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

bool mpuInit() {
  if (!mpuWrite(0x6B, 0x01)) return false;    // PWR_MGMT_1: Bangun, PLL Gyro X
  delay(100);
  mpuWrite(0x1A, 0x01);                       // CONFIG: DLPF 184 Hz
  mpuWrite(0x19, 0x00);                       // SMPLRT_DIV = 0 -> 1 kHz
  mpuWrite(0x1C, 0x08);                       // ACCEL_CONFIG: +-4 g
  return true;
}

bool mpuReadZ(int16_t &z) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3F);                           // ACCEL_ZOUT_H
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, (uint8_t)2) != 2) return false;
  int16_t hi = Wire.read();
  int16_t lo = Wire.read();
  z = (int16_t)((hi << 8) | lo);
  return true;
}

void samplingTask(void*) {
  TickType_t last = xTaskGetTickCount();
  int16_t z = 0;
  for (;;) {
    vTaskDelayUntil(&last, 1);
    if (!mpuReadZ(z)) i2cErrors = i2cErrors + 1;
    portENTER_CRITICAL(&ringMux);
    ring[widx] = z;
    widx = (widx + 1) % RING_SIZE;
    sampleCount = sampleCount + 1;
    portEXIT_CRITICAL(&ringMux);
  }
}

uint32_t getSampleCount() {
  portENTER_CRITICAL(&ringMux);
  uint32_t c = sampleCount;
  portEXIT_CRITICAL(&ringMux);
  return c;
}

bool grabWindow(uint32_t &endCount) {
  uint32_t cnt, w;
  portENTER_CRITICAL(&ringMux);
  cnt = sampleCount;
  w   = widx;
  portEXIT_CRITICAL(&ringMux);
  if (cnt < SAMPLES) return false;
  uint32_t idx = (w + RING_SIZE - SAMPLES) % RING_SIZE;
  for (int i = 0; i < SAMPLES; i++) {
    vReal[i] = (float)ring[idx];
    if (++idx >= RING_SIZE) idx = 0;
  }
  endCount = cnt;
  return true;
}

// Akuisisi Arus AC RMS selama 100 ms (5 gelombang penuh @ 50 Hz)
void updateCurrentSCT() {
  const int NUM_SAMPLES = 400;                // 400 sampel x 250 us = 100 ms (tepat 5 siklus 50 Hz)
  int raw[NUM_SAMPLES];
  long sum = 0;

  for (int i = 0; i < NUM_SAMPLES; i++) {
    raw[i] = analogRead(SCT_PIN);
    sum += raw[i];
    delayMicroseconds(250);
  }

  float mean = (float)sum / NUM_SAMPLES;
  double sumSq = 0.0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    float diff = (float)raw[i] - mean;
    sumSq += (double)(diff * diff);
  }

  // Konversi ADC count RMS ke Volt RMS (12-bit ADC: 0-4095, Vref ~3.3V)
  float vRms = sqrtf((float)(sumSq / NUM_SAMPLES)) * (3.3f / 4095.0f);
  float iRms = vRms * SCT_ICAL;
  liveData.i_rms = (iRms < CURRENT_NOISE_TH) ? 0.0f : iRms;
}

// ====================================================
// 5. PEMROSESAN SINYAL (DSP & DIAGNOSTIK)
// ====================================================
float findPeak(float fCenter, float halfWidth, float &fPeak) {
  int k0 = max(1, (int)floorf((fCenter - halfWidth) / DF));
  int k1 = min((SAMPLES / 2) - 1, (int)ceilf((fCenter + halfWidth) / DF));
  float best = 0.0f;
  int kb = k0;
  for (int k = k0; k <= k1; k++) {
    if (velSpec[k] > best) { best = velSpec[k]; kb = k; }
  }
  fPeak = kb * DF;
  return best / 1.41421356f;
}

void classify() {
  if (liveData.v_rms <= ZONE_A_MAX)      strcpy(liveData.iso_zone, "Zone A");
  else if (liveData.v_rms <= ZONE_B_MAX) strcpy(liveData.iso_zone, "Zone B");
  else if (liveData.v_rms <= ZONE_C_MAX) strcpy(liveData.iso_zone, "Zone C");
  else                                   strcpy(liveData.iso_zone, "Zone D");
}

void processWindow() {
  // 0. Baca Arus Beban Motor via SCT-013
  updateCurrentSCT();

  // 1. Detrending & Scaling Getaran
  double sum = 0;
  for (int i = 0; i < SAMPLES; i++) sum += vReal[i];
  float mean = (float)(sum / SAMPLES);

  float sumSq = 0, peak = 0;
  const float scale = GRAV / ACC_LSB_PER_G;
  for (int i = 0; i < SAMPLES; i++) {
    float a = (vReal[i] - mean) * scale;
    vReal[i] = a;
    vImag[i] = 0.0f;
    sumSq += a * a;
    if (fabsf(a) > peak) peak = fabsf(a);
  }
  float aRms = sqrtf(sumSq / SAMPLES);
  liveData.crest = (aRms > 1e-6f) ? (peak / aRms) : 1.41f;

  // 2. Hann Windowing & FFT
  FFT.windowing(FFTWindow::Hann, FFTDirection::Forward);
  FFT.compute(FFTDirection::Forward);
  FFT.complexToMagnitude();

  // 3. Spektrum Kecepatan (mm/s RMS)
  int kLo = (int)ceilf(BAND_LO_HZ / DF);
  int kHi = (int)floorf(BAND_HI_HZ / DF);
  float sumV2 = 0;
  for (int k = 0; k < SAMPLES / 2; k++) {
    if (k < kLo || k > kHi) { velSpec[k] = 0.0f; continue; }
    float f   = k * DF;
    float aPk = 4.0f * vReal[k] / SAMPLES;
    velSpec[k] = (aPk / (2.0f * PI * f)) * 1000.0f;
    sumV2 += velSpec[k] * velSpec[k];
  }
  liveData.v_rms = sqrtf(sumV2 / 3.0f);

  // 4. Deteksi Mesin Mati (Getaran di bawah ambang batas idle)
  if (liveData.v_rms < V_IDLE) {
    liveData.f_1x = liveData.amp_1x = liveData.f_2x = liveData.amp_2x = 0;
    liveData.health_index = 100;
    strcpy(liveData.iso_zone, "-");
    strcpy(liveData.kondisi, "MESIN_OFF");
    strcpy(liveData.dugaan, "-");
    return;
  }

  // 5. Dynamic Peak Tracking
  liveData.amp_1x = findPeak(F_1X_NOMINAL, F_1X_SEARCH, liveData.f_1x);
  liveData.amp_2x = findPeak(2.0f * liveData.f_1x, 1.5f, liveData.f_2x);

  // 6. Harmonisa Kelonggaran (3x - 5x)
  float dummy, harm = 0;
  for (int order = 3; order <= 5; order++) {
    harm += findPeak(order * liveData.f_1x, 1.5f, dummy);
  }
  float looseness = harm / (liveData.amp_1x + liveData.amp_2x + 1e-3f);

  // 7. Klasifikasi
  classify();
  int level = 0;
  if (strcmp(liveData.iso_zone, "Zone C") == 0) level = 1;
  if (strcmp(liveData.iso_zone, "Zone D") == 0) level = 2;

  if (looseness > LOOSENESS_WARN) {
    strcpy(liveData.dugaan, "LONGGAR_MEKANIK");
    if (level < 1) level = 1;
  } else if (liveData.amp_2x >= AMP2X_WARN) {
    strcpy(liveData.dugaan, "MISALIGNMENT_ATAU_KOPLING");
    if (level < 1) level = 1;
  } else if (liveData.i_rms >= CURRENT_OVERLOAD) {
    strcpy(liveData.dugaan, "OVERLOAD_ARUS");
    if (level < 1) level = 1;
  } else {
    strcpy(liveData.dugaan, "-");
  }

  const char* names[] = {"NORMAL", "WARNING", "PERLU_PERBAIKAN"};
  strcpy(liveData.kondisi, names[level]);

  // 8. Health Index (On-Device Heuristic)
  float v_clamped = constrain(liveData.v_rms, 0.4f, 4.5f);
  float hi_cont = 100.0f * (1.0f - ((v_clamped - 0.4f) / 4.1f));

  if (liveData.amp_2x > 1.0f) {
    hi_cont -= (liveData.amp_2x - 1.0f) * 10.0f;
  }
  if (liveData.i_rms > CURRENT_NOMINAL) {
    hi_cont -= (liveData.i_rms - CURRENT_NOMINAL) * 2.0f;
  }

  liveData.health_index = (int)round(constrain(hi_cont, 0.0f, 100.0f));
}

// ====================================================
// 6. PENGIRIMAN DATA KE NODE-RED
// ====================================================
int buildJson(char* buf, size_t n) {
  return snprintf(buf, n,
    "{\"seq\":%lu,\"i_rms\":%.2f,\"arus\":%.2f,\"v_rms\":%.2f,\"crest_factor\":%.2f,"
    "\"f_1x\":%.2f,\"amp_1x\":%.2f,\"f_2x\":%.2f,\"amp_2x\":%.2f,"
    "\"hi\":%d,\"iso_zone\":\"%s\",\"kondisi\":\"%s\",\"dugaan\":\"%s\",\"i2c_err\":%lu}",
    (unsigned long)seq, liveData.i_rms, liveData.i_rms, liveData.v_rms, liveData.crest,
    liveData.f_1x, liveData.amp_1x, liveData.f_2x, liveData.amp_2x,
    liveData.health_index, liveData.iso_zone, liveData.kondisi, liveData.dugaan,
    (unsigned long)i2cErrors);
}

void sendNodeRed() {
  if (Ethernet.hardwareStatus() == EthernetNoHardware) {
    Serial.println(F("[Node-RED] Lewati pengiriman: Modul W5500 tidak terdeteksi."));
    return;
  }
  if (Ethernet.linkStatus() == LinkOFF) {
    Serial.println(F("[Node-RED] Lewati pengiriman: Kabel Ethernet terputus."));
    return;
  }
  if (Ethernet.localIP() == IPAddress(0, 0, 0, 0)) {
    Serial.println(F("[Node-RED] Lewati pengiriman: Belum mendapat IP DHCP."));
    return;
  }

  EthernetClient client;
  Serial.printf("[Node-RED] Menghubungkan ke %s:%d... ", serverIP.toString().c_str(), serverPort);

  if (client.connect(serverIP, serverPort)) {
    Serial.println(F("Terhubung!"));

    client.print("POST ");
    client.print(serverPath);
    client.println(" HTTP/1.1");
    client.print("Host: ");
    client.print(serverIP);
    client.print(":");
    client.println(serverPort);
    client.println("Content-Type: application/json");
    client.println("Connection: close");
    client.print("Content-Length: ");
    client.println(strlen(latestJson));
    client.println();
    client.print(latestJson);

    unsigned long t0 = millis();
    bool responseReceived = false;
    while (client.connected() && (millis() - t0 < 500)) {
      if (client.available()) {
        String line = client.readStringUntil('\n');
        if (line.startsWith("HTTP/1.")) {
          Serial.printf("[Node-RED] Status Response: %s\n", line.c_str());
          responseReceived = true;
          break;
        }
      }
    }
    if (!responseReceived) {
      Serial.println(F("[Node-RED] Terkirim, tetapi tidak ada respons HTTP dari Node-RED (Periksa node 'http response')."));
    }
    client.stop();
  } else {
    Serial.println(F("GAGAL KONEK! (Cek IP Server, Port 1880, atau Firewall Linux)"));
  }
}

void publishResult() {
  seq++;
  buildJson(latestJson, sizeof(latestJson));
  sendNodeRed();
}

void printSerial() {
  Serial.println(F("------------------------------------------------"));
  Serial.printf("V_RMS: %5.2f mm/s (%s) | Arus: %5.2f A | CF: %4.2f\n",
                liveData.v_rms, liveData.iso_zone, liveData.i_rms, liveData.crest);
  Serial.printf("1x: %5.2f Hz / %5.2f mm/s | 2x: %5.2f Hz / %5.2f mm/s\n",
                liveData.f_1x, liveData.amp_1x, liveData.f_2x, liveData.amp_2x);
  Serial.printf("Kondisi: %s | HI: %d %% | Dugaan: %s | I2C err: %lu | Heap: %u\n",
                liveData.kondisi, liveData.health_index, liveData.dugaan, (unsigned long)i2cErrors, (unsigned)ESP.getFreeHeap());
}

// ====================================================
// 7. SETUP & LOOP
// ====================================================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println(F("\n[INIT] Memulai Edge Monitor Pompa 10 HP (MPU6050 + SCT013)..."));

  // Konfigurasi ADC1 untuk SCT-013 (Pin GPIO 34, Resolusi 12-bit, Atenuasi 11dB untuk 0 - 3.3V)
  pinMode(SCT_PIN, INPUT);
  analogReadResolution(12);
  analogSetPinAttenuation(SCT_PIN, ADC_11db);

  // Inisialisasi Ethernet W5500 (VSPI + DHCP)
  SPI.begin(18, 19, 23, W5500_CS);
  Ethernet.init(W5500_CS);
  Ethernet.setRetransmissionTimeout(200);
  Ethernet.setRetransmissionCount(3);

  Serial.println(F("[INIT] Menghubungkan ke Ethernet W5500 (DHCP)..."));
  if (Ethernet.begin(mac) == 0) {
    Serial.println(F("[ERROR] Gagal memperoleh IP via DHCP."));
    if (Ethernet.hardwareStatus() == EthernetNoHardware) {
      Serial.println(F("[ERROR] Modul W5500 tidak terdeteksi! Cek wiring SPI & CS=5."));
    } else if (Ethernet.linkStatus() == LinkOFF) {
      Serial.println(F("[WARNING] Kabel Ethernet tidak terhubung."));
    }
  } else {
    Serial.printf("[OK] Terhubung ke Ethernet! IP ESP32: %s\n", Ethernet.localIP().toString().c_str());
  }

  // Sensor MPU-6050 (I2C Fast 400kHz)
  Wire.begin(21, 22);
  Wire.setClock(400000);
  mpuReady = mpuInit();
  if (mpuReady) {
    Serial.println(F("[OK] MPU-6050 siap (+-4 g, DLPF 184 Hz, 1 kHz)"));
    xTaskCreatePinnedToCore(samplingTask, "sampling", 4096, NULL,
                            configMAX_PRIORITIES - 2, NULL, 1);
  } else {
    Serial.println(F("[WARNING] MPU-6050 tidak terdeteksi! Cek wiring SDA=21, SCL=22"));
    strcpy(liveData.kondisi, "SENSOR_ERROR");
  }
}

void loop() {
  // Jaga lease DHCP & coba ulang DHCP jika IP belum didapat
  Ethernet.maintain();
  static unsigned long lastEthCheck = 0;
  if (millis() - lastEthCheck > 10000) {
    lastEthCheck = millis();
    if (Ethernet.localIP() == IPAddress(0, 0, 0, 0) &&
        Ethernet.hardwareStatus() != EthernetNoHardware &&
        Ethernet.linkStatus() == LinkON) {
      Serial.println(F("[W5500] Mencoba memperoleh IP via DHCP..."));
      if (Ethernet.begin(mac) != 0) {
        Serial.printf("[OK] IP ESP32: %s\n", Ethernet.localIP().toString().c_str());
      }
    }
  }

  static uint32_t lastCount = 0;
  if (mpuReady) {
    uint32_t cnt = getSampleCount();
    if (cnt >= SAMPLES && (cnt - lastCount) >= HOP) {
      uint32_t endCount;
      if (grabWindow(endCount)) {
        lastCount = endCount;
        processWindow();
        publishResult();
        printSerial();
      }
    }
  }
  yield();
}