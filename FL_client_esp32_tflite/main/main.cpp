/*
 * Federated Learning Client — ESP32 (ESP-IDF + TFLite)
 *
 * Exact same logic as Arduino version but using ESP-IDF framework.
 * Downloads TFLite model from server for inference after training.
 *
 * Build: idf.py build
 * Flash: idf.py -p COM_PORT flash monitor
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_http_client.h"
#include "nvs_flash.h"
#include "driver/i2c.h"
#include "esp_random.h"
#include "esp_timer.h"

// TFLite includes
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "FL_CLIENT";

// ── Config ────────────────────────────────────────────────────────────────────
#define WIFI_SSID      "Cybergaar"
#define WIFI_PASS      "curedata4low"
#define SERVER_IP      "10.219.84.92"
#define SERVER_PORT    5000
#define CLIENT_ID      "esp32-node-A"

#define POLL_INTERVAL_MS      3000
#define INFERENCE_INTERVAL_MS 500
#define SAMPLE_INTERVAL_MS    200
#define LOCAL_SAMPLES_PER_CLASS 40
#define LOCAL_EPOCHS          10
#define LEARNING_RATE         0.01f
#define ACTIVE_CLASSES        4
#define CLASS_TRANSITION_DELAY_MS 5000

// ── Model dimensions ─────────────────────────────────────────────────────────
#define INPUT_DIM     9
#define HIDDEN1      16
#define HIDDEN2       8
#define OUTPUT_DIM    4
#define TOTAL_WEIGHTS 332

// ── Activity classes ─────────────────────────────────────────────────────────
#define CLASS_STILL   0
#define CLASS_WALKING 1
#define CLASS_SHAKING 2
#define CLASS_TAPPING 3

const char* CLASS_NAMES[OUTPUT_DIM] = {"STILL", "WALKING", "SHAKING", "TAPPING"};

const float ONE_HOT[OUTPUT_DIM][OUTPUT_DIM] = {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1},
};

// ── Network weights ───────────────────────────────────────────────────────────
float W1[INPUT_DIM][HIDDEN1], b1[HIDDEN1];
float W2[HIDDEN1][HIDDEN2], b2[HIDDEN2];
float W3[HIDDEN2][OUTPUT_DIM], b3[OUTPUT_DIM];

// ── Client state ──────────────────────────────────────────────────────────────
enum ClientPhase { PHASE_IDLE, PHASE_TRAIN, PHASE_INFERENCE };
ClientPhase clientPhase = PHASE_IDLE;
int currentRound = 0;
bool registered = false;
float epochLosses[LOCAL_EPOCHS];

// ── TFLite globals ────────────────────────────────────────────────────────────
#define TFLITE_ARENA_SIZE (32 * 1024)
static uint8_t tensor_arena[TFLITE_ARENA_SIZE] __attribute__((aligned(32)));

static const tflite::Model* tflite_model = nullptr;
static tflite::MicroInterpreter* tflite_interpreter = nullptr;
static TfLiteTensor* tflite_input = nullptr;
static TfLiteTensor* tflite_output = nullptr;
static bool tflite_initialized = false;
static bool tflite_model_downloaded = false;

#define MAX_MODEL_SIZE 10240
static uint8_t tflite_model_buffer[MAX_MODEL_SIZE];
static size_t tflite_model_size = 0;

// ── HTTP response buffer ──────────────────────────────────────────────────────
#define HTTP_RESPONSE_MAX 8192

// ── WiFi event group ──────────────────────────────────────────────────────────
static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1
static int s_retry_num = 0;
#define MAXIMUM_RETRY 10

// ── I2C handles ───────────────────────────────────────────────────────────────
// (using legacy driver/i2c.h)

// ── Forward declarations ──────────────────────────────────────────────────────
void initWeightsLocal();
void flattenWeights(float* flat);
void unflattenWeights(const float* flat);
bool read_sensors(float* out9);
void runLocalTraining();
void runInference();
bool downloadTFLiteModel();
bool initTFLite();
bool httpRegister();
bool pollStatus(float* weights_out, int* round_out, char* instruction, int instr_max);
bool downloadModel();
bool uploadWeights(int n_samples);
void verifyWeightsWithServer();

// ── String helper (replaces Arduino String) ───────────────────────────────────
class SimpleString {
public:
    char* data;
    int len;
    int capacity;
    
    SimpleString(int cap = 512) : len(0), capacity(cap) {
        data = (char*)malloc(cap);
        data[0] = '\0';
    }
    
    ~SimpleString() { free(data); }
    
    void clear() { len = 0; data[0] = '\0'; }
    
    void append(const char* s) {
        int slen = strlen(s);
        while (len + slen + 1 > capacity) {
            capacity *= 2;
            data = (char*)realloc(data, capacity);
        }
        memcpy(data + len, s, slen + 1);
        len += slen;
    }
    
    void append(float f, int decimals = 6) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.*f", decimals, f);
        append(buf);
    }
    
    void append(int i) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", i);
        append(buf);
    }
    
    int indexOf(const char* s) {
        char* found = strstr(data, s);
        return found ? (int)(found - data) : -1;
    }
    
    SimpleString substring(int start, int end) {
        SimpleString result(end - start + 1);
        int copyLen = end - start;
        if (copyLen > 0) {
            memcpy(result.data, data + start, copyLen);
            result.data[copyLen] = '\0';
            result.len = copyLen;
        }
        return result;
    }
    
    float toFloat() { return atof(data); }
    int toInt() { return atoi(data); }
};

// ── WiFi event handler ────────────────────────────────────────────────────────
static void event_handler(void* arg, esp_event_base_t event_base,
                          int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_num < MAXIMUM_RETRY) {
            esp_wifi_connect();
            s_retry_num++;
            ESP_LOGI(TAG, "Retry to connect to the AP");
        } else {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got IP:" IPSTR, IP2STR(&event->ip_info.ip));
        s_retry_num = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

// ── WiFi init ─────────────────────────────────────────────────────────────────
void wifi_init_sta(void) {
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &event_handler,
                                                        NULL,
                                                        &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &event_handler,
                                                        NULL,
                                                        &instance_got_ip));

    wifi_config_t wifi_config = {};
    strncpy((char*)wifi_config.sta.ssid, WIFI_SSID, sizeof(wifi_config.sta.ssid) - 1);
    strncpy((char*)wifi_config.sta.password, WIFI_PASS, sizeof(wifi_config.sta.password) - 1);
    
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
                                           WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
                                           pdFALSE,
                                           pdFALSE,
                                           portMAX_DELAY);

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Connected to SSID:%s", WIFI_SSID);
    } else if (bits & WIFI_FAIL_BIT) {
        ESP_LOGI(TAG, "Failed to connect to SSID:%s", WIFI_SSID);
    }
}

// ── I2C init (legacy driver) ───────────────────────────────────────────────────
void i2c_master_init() {
    i2c_config_t conf;
    conf.mode = I2C_MODE_MASTER;
    conf.sda_io_num = (gpio_num_t)21;
    conf.scl_io_num = (gpio_num_t)22;
    conf.sda_pullup_en = GPIO_PULLUP_ENABLE;
    conf.scl_pullup_en = GPIO_PULLUP_ENABLE;
    conf.master.clk_speed = 100000;
    conf.clk_flags = 0;
    
    i2c_param_config(I2C_NUM_0, &conf);
    i2c_driver_install(I2C_NUM_0, I2C_MODE_MASTER, 0, 0, 0);
    ESP_LOGI(TAG, "I2C initialized");
}

// ── MPU6050 read ──────────────────────────────────────────────────────────────
bool read_sensors(float* out9) {
    uint8_t reg = 0x3B;
    uint8_t data[14];
    
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (0x68 << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write(cmd, &reg, 1, true);
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (0x68 << 1) | I2C_MASTER_READ, true);
    i2c_master_read(cmd, data, 14, I2C_MASTER_LAST_NACK);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(I2C_NUM_0, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);
    
    if (ret != ESP_OK) return false;
    
    int16_t ax = (data[0] << 8) | data[1];
    int16_t ay = (data[2] << 8) | data[3];
    int16_t az = (data[4] << 8) | data[5];
    int16_t gx = (data[8] << 8) | data[9];
    int16_t gy = (data[10] << 8) | data[11];
    int16_t gz = (data[12] << 8) | data[13];
    
    out9[0] = ax / 16384.0f;
    out9[1] = ay / 16384.0f;
    out9[2] = az / 16384.0f;
    out9[3] = gx / 131.0f / 250.0f;
    out9[4] = gy / 131.0f / 250.0f;
    out9[5] = gz / 131.0f / 250.0f;
    out9[6] = 0.0f;
    out9[7] = 0.0f;
    out9[8] = 0.0f;
    
    return true;
}

// ── Weight initialization ─────────────────────────────────────────────────────
void initWeightsLocal() {
    auto xav = [](int fan_in, int fan_out) -> float {
        float limit = sqrtf(6.0f / (fan_in + fan_out));
        return ((float)((int32_t)(esp_random() % 2001) - 1000) / 1000.0f) * limit;
    };
    
    for (int i = 0; i < INPUT_DIM; i++)
        for (int j = 0; j < HIDDEN1; j++) W1[i][j] = xav(INPUT_DIM, HIDDEN1);
    for (int j = 0; j < HIDDEN1; j++) b1[j] = 0.0f;
    
    for (int i = 0; i < HIDDEN1; i++)
        for (int j = 0; j < HIDDEN2; j++) W2[i][j] = xav(HIDDEN1, HIDDEN2);
    for (int j = 0; j < HIDDEN2; j++) b2[j] = 0.0f;
    
    for (int i = 0; i < HIDDEN2; i++)
        for (int j = 0; j < OUTPUT_DIM; j++) W3[i][j] = xav(HIDDEN2, OUTPUT_DIM);
    for (int j = 0; j < OUTPUT_DIM; j++) b3[j] = 0.0f;
    
    ESP_LOGI(TAG, "Weights initialised via Xavier + ESP32 RNG");
}

// ── Weight serialization ──────────────────────────────────────────────────────
void flattenWeights(float* flat) {
    int idx = 0;
    for (int i = 0; i < INPUT_DIM; i++)
        for (int j = 0; j < HIDDEN1; j++) flat[idx++] = W1[i][j];
    for (int j = 0; j < HIDDEN1; j++) flat[idx++] = b1[j];
    for (int i = 0; i < HIDDEN1; i++)
        for (int j = 0; j < HIDDEN2; j++) flat[idx++] = W2[i][j];
    for (int j = 0; j < HIDDEN2; j++) flat[idx++] = b2[j];
    for (int i = 0; i < HIDDEN2; i++)
        for (int j = 0; j < OUTPUT_DIM; j++) flat[idx++] = W3[i][j];
    for (int j = 0; j < OUTPUT_DIM; j++) flat[idx++] = b3[j];
}

void unflattenWeights(const float* flat) {
    int idx = 0;
    for (int i = 0; i < INPUT_DIM; i++)
        for (int j = 0; j < HIDDEN1; j++) W1[i][j] = flat[idx++];
    for (int j = 0; j < HIDDEN1; j++) b1[j] = flat[idx++];
    for (int i = 0; i < HIDDEN1; i++)
        for (int j = 0; j < HIDDEN2; j++) W2[i][j] = flat[idx++];
    for (int j = 0; j < HIDDEN2; j++) b2[j] = flat[idx++];
    for (int i = 0; i < HIDDEN2; i++)
        for (int j = 0; j < OUTPUT_DIM; j++) W3[i][j] = flat[idx++];
    for (int j = 0; j < OUTPUT_DIM; j++) b3[j] = flat[idx++];
}

// ── Manual forward pass (same as inference.h) ─────────────────────────────────
inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

void forward_manual(const float* x, float* h1_out, float* h2_out, float* out) {
    for (int j = 0; j < HIDDEN1; j++) {
        float s = b1[j];
        for (int i = 0; i < INPUT_DIM; i++) s += x[i] * W1[i][j];
        h1_out[j] = relu(s);
    }
    for (int j = 0; j < HIDDEN2; j++) {
        float s = b2[j];
        for (int i = 0; i < HIDDEN1; i++) s += h1_out[i] * W2[i][j];
        h2_out[j] = relu(s);
    }
    float logits[OUTPUT_DIM];
    float max_logit = -1e9f;
    for (int j = 0; j < OUTPUT_DIM; j++) {
        float s = b3[j];
        for (int i = 0; i < HIDDEN2; i++) s += h2_out[i] * W3[i][j];
        logits[j] = s;
        if (s > max_logit) max_logit = s;
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < OUTPUT_DIM; j++) {
        out[j] = expf(logits[j] - max_logit);
        sum_exp += out[j];
    }
    for (int j = 0; j < OUTPUT_DIM; j++) out[j] /= sum_exp;
}

int infer_manual(const float* x) {
    float h1[HIDDEN1], h2[HIDDEN2], out[OUTPUT_DIM];
    forward_manual(x, h1, h2, out);
    int best = 0;
    for (int j = 1; j < OUTPUT_DIM; j++)
        if (out[j] > out[best]) best = j;
    return best;
}

// ── TFLite forward pass ───────────────────────────────────────────────────────
void forward_tflite(const float* x, float* h1_out, float* h2_out, float* out) {
    if (!tflite_initialized) {
        memset(out, 0, OUTPUT_DIM * sizeof(float));
        return;
    }
    
    memcpy(tflite_input->data.f, x, INPUT_DIM * sizeof(float));
    
    if (tflite_interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "TFLite Invoke failed");
        return;
    }
    
    memcpy(out, tflite_output->data.f, OUTPUT_DIM * sizeof(float));
    memset(h1_out, 0, HIDDEN1 * sizeof(float));
    memset(h2_out, 0, HIDDEN2 * sizeof(float));
}

int infer_tflite(const float* x) {
    if (!tflite_initialized) return infer_manual(x);
    
    memcpy(tflite_input->data.f, x, INPUT_DIM * sizeof(float));
    
    if (tflite_interpreter->Invoke() != kTfLiteOk) return infer_manual(x);
    
    const float* probs = tflite_output->data.f;
    int best = 0;
    for (int j = 1; j < OUTPUT_DIM; j++)
        if (probs[j] > probs[best]) best = j;
    return best;
}

// ── SGD step with gradient clipping ──────────────────────────────────────────
#define GRAD_CLIP 1.0f

inline float clip_grad(float g) {
    if (g > GRAD_CLIP) return GRAD_CLIP;
    if (g < -GRAD_CLIP) return -GRAD_CLIP;
    return g;
}

void sgd_step(const float* x, const float* one_hot_label) {
    float h1[HIDDEN1], h2[HIDDEN2], out[OUTPUT_DIM];
    forward_manual(x, h1, h2, out);
    
    float dout[OUTPUT_DIM];
    for (int j = 0; j < OUTPUT_DIM; j++)
        dout[j] = clip_grad(out[j] - one_hot_label[j]);
    
    float dh2[HIDDEN2] = {};
    for (int i = 0; i < HIDDEN2; i++) {
        for (int j = 0; j < OUTPUT_DIM; j++) {
            float grad = clip_grad(dout[j] * h2[i]);
            W3[i][j] -= LEARNING_RATE * grad;
            dh2[i] += dout[j] * W3[i][j];
        }
    }
    for (int j = 0; j < OUTPUT_DIM; j++) b3[j] -= LEARNING_RATE * dout[j];
    
    for (int i = 0; i < HIDDEN2; i++) dh2[i] = (h2[i] > 0.0f) ? clip_grad(dh2[i]) : 0.0f;
    
    float dh1[HIDDEN1] = {};
    for (int i = 0; i < HIDDEN1; i++) {
        for (int j = 0; j < HIDDEN2; j++) {
            float grad = clip_grad(dh2[j] * h1[i]);
            W2[i][j] -= LEARNING_RATE * grad;
            dh1[i] += dh2[j] * W2[i][j];
        }
    }
    for (int j = 0; j < HIDDEN2; j++) b2[j] -= LEARNING_RATE * dh2[j];
    
    for (int i = 0; i < HIDDEN1; i++) dh1[i] = (h1[i] > 0.0f) ? clip_grad(dh1[i]) : 0.0f;
    
    for (int i = 0; i < INPUT_DIM; i++)
        for (int j = 0; j < HIDDEN1; j++) {
            float grad = clip_grad(dh1[j] * x[i]);
            W1[i][j] -= LEARNING_RATE * grad;
        }
    for (int j = 0; j < HIDDEN1; j++) b1[j] -= LEARNING_RATE * dh1[j];
}

float cross_entropy(const float* x, int true_class) {
    float h1[HIDDEN1], h2[HIDDEN2], out[OUTPUT_DIM];
    forward_manual(x, h1, h2, out);
    float p = out[true_class];
    if (p < 1e-7f) p = 1e-7f;
    return -logf(p);
}

// ── HTTP helper ───────────────────────────────────────────────────────────────
char server_url[128];

const char* serverUrl(const char* path) {
    snprintf(server_url, sizeof(server_url), "http://%s:%d%s", SERVER_IP, SERVER_PORT, path);
    return server_url;
}

// ── HTTP GET helper ───────────────────────────────────────────────────────────
bool httpGet(const char* url, char* response, int max_len) {
    esp_http_client_config_t config = {};
    config.url = url;
    config.timeout_ms = 10000;
    config.method = HTTP_METHOD_GET;
    
    esp_http_client_handle_t client = esp_http_client_init(&config);
    
    esp_err_t err = esp_http_client_open(client, 0);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "HTTP GET open failed: %s", esp_err_to_name(err));
        esp_http_client_cleanup(client);
        return false;
    }
    
    int content_length = esp_http_client_fetch_headers(client);
    
    // Read all data in loop
    int total_read = 0;
    while (total_read < max_len - 1) {
        int read_len = esp_http_client_read(client, response + total_read, max_len - 1 - total_read);
        if (read_len <= 0) break;
        total_read += read_len;
    }
    response[total_read] = '\0';
    
    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    
    ESP_LOGI(TAG, "HTTP GET %d bytes from %s", total_read, url);
    return true;
}

// ── HTTP POST helper ──────────────────────────────────────────────────────────
int httpPost(const char* url, const char* body, char* response, int max_len) {
    esp_http_client_config_t config = {};
    config.url = url;
    config.timeout_ms = 15000;
    config.method = HTTP_METHOD_POST;
    
    esp_http_client_handle_t client = esp_http_client_init(&config);
    esp_http_client_set_header(client, "Content-Type", "application/json");
    esp_http_client_set_post_field(client, body, strlen(body));
    
    esp_err_t err = esp_http_client_open(client, strlen(body));
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "HTTP POST open failed: %s", esp_err_to_name(err));
        esp_http_client_cleanup(client);
        return -1;
    }
    
    int written = esp_http_client_write(client, body, strlen(body));
    if (written < 0) {
        ESP_LOGE(TAG, "HTTP POST write failed");
        esp_http_client_close(client);
        esp_http_client_cleanup(client);
        return -1;
    }
    
    esp_http_client_fetch_headers(client);
    
    if (response && max_len > 0) {
        int read_len = esp_http_client_read(client, response, max_len - 1);
        if (read_len < 0) read_len = 0;
        response[read_len] = '\0';
    }
    
    int status = esp_http_client_get_status_code(client);
    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    return status;
}

// ── JSON helper ───────────────────────────────────────────────────────────────
bool parseWeightsFromJson(const char* payload, float* weights_out, int& round_out) {
    const char* ri = strstr(payload, "\"round\":");
    if (ri) round_out = atoi(ri + 8);
    
    const char* start = strstr(payload, "\"weights\":");
    if (!start) return false;
    start = strchr(start, '[');
    if (!start) return false;
    
    int idx = 0;
    const char* pos = start + 1;
    const char* end = payload + strlen(payload);
    
    while (idx < TOTAL_WEIGHTS && pos < end) {
        char c = *pos;
        if (c == '[' || c == ']' || c == ' ' || c == '\n' || c == '\r' || c == ',') {
            pos++;
            continue;
        }
        const char* token_end = pos;
        while (token_end < end && *token_end != ',' && *token_end != ']' && 
               *token_end != '[' && *token_end != ' ') token_end++;
        
        char token[32];
        int token_len = token_end - pos;
        if (token_len >= sizeof(token)) token_len = sizeof(token) - 1;
        memcpy(token, pos, token_len);
        token[token_len] = '\0';
        
        // Trim whitespace
        char* t = token;
        while (*t == ' ' || *t == '\t') t++;
        
        if (strlen(t) > 0) weights_out[idx++] = atof(t);
        pos = token_end;
    }
    return (idx == TOTAL_WEIGHTS);
}

// ── Register with server ──────────────────────────────────────────────────────
bool httpRegister() {
    char body[128];
    snprintf(body, sizeof(body), "{\"client_id\":\"%s\"}", CLIENT_ID);
    
    char response[256];
    int status = httpPost(serverUrl("/register"), body, response, sizeof(response));
    
    if (status == 200) {
        ESP_LOGI(TAG, "Registered as %s", CLIENT_ID);
        return true;
    }
    ESP_LOGW(TAG, "Registration failed HTTP %d", status);
    return false;
}

// ── Poll status ───────────────────────────────────────────────────────────────
bool pollStatus(float* weights_out, int* round_out, char* instruction, int instr_max) {
    char url[256];
    snprintf(url, sizeof(url), "%s?client_id=%s", serverUrl("/status"), CLIENT_ID);
    
    char response[HTTP_RESPONSE_MAX];
    if (!httpGet(url, response, sizeof(response))) {
        strncpy(instruction, "wait", instr_max);
        return false;
    }
    
    // Extract instruction
    const char* ii = strstr(response, "\"instruction\":\"");
    if (!ii) {
        strncpy(instruction, "wait", instr_max);
        return false;
    }
    const char* is = ii + 15;
    const char* ie = strchr(is, '"');
    if (!ie) {
        strncpy(instruction, "wait", instr_max);
        return false;
    }
    int len = ie - is;
    if (len >= instr_max) len = instr_max - 1;
    memcpy(instruction, is, len);
    instruction[len] = '\0';
    
    // Extract round
    if (round_out) {
        const char* ri = strstr(response, "\"round\":");
        if (ri) *round_out = atoi(ri + 8);
    }
    
    // Extract weights
    if (weights_out && strstr(response, "\"weights\"")) {
        int dummy = 0;
        if (!parseWeightsFromJson(response, weights_out, dummy)) {
            ESP_LOGE(TAG, "Failed to parse weights from status response");
            return false;
        }
    }
    
    return true;
}

// ── Download model ────────────────────────────────────────────────────────────
bool downloadModel() {
    char response[HTTP_RESPONSE_MAX];
    if (!httpGet(serverUrl("/model"), response, sizeof(response))) {
        ESP_LOGW(TAG, "Model download failed — using local weights");
        return false;
    }
    
    float flat[TOTAL_WEIGHTS];
    int round = 0;
    if (parseWeightsFromJson(response, flat, round)) {
        unflattenWeights(flat);
        currentRound = round;
        ESP_LOGI(TAG, "Model downloaded (round %d)", currentRound);
        return true;
    }
    ESP_LOGW(TAG, "Model parse failed");
    return false;
}

// ── Upload weights ────────────────────────────────────────────────────────────
bool uploadWeights(int n_samples) {
    float flat[TOTAL_WEIGHTS];
    flattenWeights(flat);
    
    // Build JSON body
    SimpleString body(4096);
    body.append("{\"client_id\":\"");
    body.append(CLIENT_ID);
    body.append("\",\"round\":");
    body.append(currentRound);
    body.append(",\"n_samples\":");
    body.append(n_samples);
    body.append(",\"epoch_losses\":[");
    for (int e = 0; e < LOCAL_EPOCHS; e++) {
        body.append(epochLosses[e], 5);
        if (e < LOCAL_EPOCHS - 1) body.append(",");
    }
    body.append("],\"weights\":[");
    for (int i = 0; i < TOTAL_WEIGHTS; i++) {
        body.append(flat[i], 6);
        if (i < TOTAL_WEIGHTS - 1) body.append(",");
    }
    body.append("]}");
    
    char response[512];
    int status = httpPost(serverUrl("/update"), body.data, response, sizeof(response));
    
    if (status == 200) {
        ESP_LOGI(TAG, "Upload OK");
        return true;
    }
    ESP_LOGW(TAG, "Upload FAILED HTTP %d — %s", status, response);
    return false;
}

// ── Verify weights ────────────────────────────────────────────────────────────
void verifyWeightsWithServer() {
    ESP_LOGI(TAG, "Verifying weights...");
    
    float flat[TOTAL_WEIGHTS];
    flattenWeights(flat);
    
    SimpleString body(4096);
    body.append("{\"client_id\":\"");
    body.append(CLIENT_ID);
    body.append("\",\"weights\":[");
    for (int i = 0; i < TOTAL_WEIGHTS; i++) {
        body.append(flat[i], 6);
        if (i < TOTAL_WEIGHTS - 1) body.append(",");
    }
    body.append("]}");
    
    char response[256];
    int status = httpPost(serverUrl("/verify_weights"), body.data, response, sizeof(response));
    
    if (status == 200) {
        bool matched = strstr(response, "\"match\":true") != NULL;
        ESP_LOGI(TAG, "Verification: %s", matched ? "PASSED" : "FAILED");
    }
}

// ── Download TFLite model from server ─────────────────────────────────────────
bool downloadTFLiteModel() {
    ESP_LOGI(TAG, "Downloading TFLite model from server...");
    
    esp_http_client_config_t config = {};
    config.url = serverUrl("/download_tflite");
    config.timeout_ms = 10000;
    
    esp_http_client_handle_t client = esp_http_client_init(&config);
    esp_err_t err = esp_http_client_open(client, 0);
    
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "HTTP open failed: %s", esp_err_to_name(err));
        esp_http_client_cleanup(client);
        return false;
    }
    
    int content_length = esp_http_client_fetch_headers(client);
    if (content_length <= 0 || content_length > MAX_MODEL_SIZE) {
        ESP_LOGE(TAG, "Invalid content length: %d", content_length);
        esp_http_client_close(client);
        esp_http_client_cleanup(client);
        return false;
    }
    
    int read_len = esp_http_client_read(client, (char*)tflite_model_buffer, content_length);
    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    
    if (read_len != content_length) {
        ESP_LOGE(TAG, "Download incomplete: %d/%d", read_len, content_length);
        return false;
    }
    
    // Validate TFLite model: must start with flatbuffer magic (0x1c) and be at least 2KB
    if (read_len < 2000 || tflite_model_buffer[0] != 0x1c) {
        ESP_LOGE(TAG, "Invalid model: %d bytes, first byte=0x%02x", read_len, tflite_model_buffer[0]);
        tflite_model_buffer[128] = 0;
        ESP_LOGE(TAG, "Server response: %s", (char*)tflite_model_buffer);
        return false;
    }
    
    tflite_model_size = read_len;
    tflite_model_downloaded = true;
    tflite_initialized = false;
    
    ESP_LOGI(TAG, "TFLite model downloaded: %d bytes", tflite_model_size);
    return true;
}

// ── Initialize TFLite ────────────────────────────────────────────────────────
bool initTFLite() {
    if (!tflite_model_downloaded || tflite_model_size == 0) {
        ESP_LOGE(TAG, "No model downloaded");
        return false;
    }
    
    tflite_model = tflite::GetModel(tflite_model_buffer);
    if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch");
        return false;
    }
    
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddReshape();
    
    static tflite::MicroInterpreter static_interpreter(
        tflite_model, resolver, tensor_arena, TFLITE_ARENA_SIZE);
    tflite_interpreter = &static_interpreter;
    
    if (tflite_interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        return false;
    }
    
    tflite_input = tflite_interpreter->input(0);
    tflite_output = tflite_interpreter->output(0);
    
    ESP_LOGI(TAG, "TFLite model initialized successfully (%d bytes)", tflite_model_size);
    tflite_initialized = true;
    return true;
}

// ── Run training (same as training.h) ─────────────────────────────────────────
void runLocalTraining() {
    ESP_LOGI(TAG, "Training started (interleaved classes)");
    
    // 1. Download latest global model
    if (!downloadModel()) {
        ESP_LOGI(TAG, "Using existing local weights");
    }
    
    const int TOTAL_SAMPLES = ACTIVE_CLASSES * LOCAL_SAMPLES_PER_CLASS;
    float samples[TOTAL_SAMPLES][INPUT_DIM];
    int labels[TOTAL_SAMPLES];
    
    // 2. Collect samples per-class
    int sampleIdx = 0;
    for (int cls = 0; cls < ACTIVE_CLASSES; cls++) {
        ESP_LOGI(TAG, "Class %d/%d: %s", cls + 1, ACTIVE_CLASSES, CLASS_NAMES[cls]);
        ESP_LOGI(TAG, "Perform %s now! Collecting %d samples...", CLASS_NAMES[cls], LOCAL_SAMPLES_PER_CLASS);
        
        int collected = 0;
        while (collected < LOCAL_SAMPLES_PER_CLASS) {
            float s[INPUT_DIM];
            if (read_sensors(s)) {
                memcpy(samples[sampleIdx], s, sizeof(s));
                labels[sampleIdx] = cls;
                sampleIdx++;
                collected++;
                if (collected % 10 == 0)
                    ESP_LOGI(TAG, "  %d / %d", collected, LOCAL_SAMPLES_PER_CLASS);
            }
            vTaskDelay(pdMS_TO_TICKS(SAMPLE_INTERVAL_MS));
        }
        ESP_LOGI(TAG, "Collected %d samples for %s", collected, CLASS_NAMES[cls]);
        
        if (cls < ACTIVE_CLASSES - 1) {
            ESP_LOGI(TAG, "Next class in %d s...", CLASS_TRANSITION_DELAY_MS / 1000);
            vTaskDelay(pdMS_TO_TICKS(CLASS_TRANSITION_DELAY_MS));
        }
    }
    
    // 3. Shuffle samples
    ESP_LOGI(TAG, "Shuffling samples...");
    for (int i = TOTAL_SAMPLES - 1; i > 0; i--) {
        int j = esp_random() % (i + 1);
        float temp[INPUT_DIM];
        memcpy(temp, samples[i], sizeof(temp));
        memcpy(samples[i], samples[j], sizeof(samples[i]));
        memcpy(samples[j], temp, sizeof(temp));
        int t = labels[i];
        labels[i] = labels[j];
        labels[j] = t;
    }
    
    // 4. Train epochs
    for (int e = 0; e < LOCAL_EPOCHS; e++) {
        float total_loss = 0.0f;
        for (int t = 0; t < TOTAL_SAMPLES; t++) {
            sgd_step(samples[t], ONE_HOT[labels[t]]);
            total_loss += cross_entropy(samples[t], labels[t]);
        }
        epochLosses[e] = total_loss / TOTAL_SAMPLES;
        ESP_LOGI(TAG, "Epoch %d/%d CE-loss=%.5f", e + 1, LOCAL_EPOCHS, epochLosses[e]);
    }
    
    // 5. Accuracy check
    int correct = 0;
    for (int t = 0; t < TOTAL_SAMPLES; t++)
        if (infer_manual(samples[t]) == labels[t]) correct++;
    ESP_LOGI(TAG, "Training accuracy: %d/%d (%.1f%%)", correct, TOTAL_SAMPLES, 100.0f * correct / TOTAL_SAMPLES);
    
    // 6. Check for NaN weights before uploading
    bool has_nan = false;
    float flat[TOTAL_WEIGHTS];
    flattenWeights(flat);
    for (int i = 0; i < TOTAL_WEIGHTS; i++) {
        if (isnan(flat[i]) || isinf(flat[i])) {
            has_nan = true;
            break;
        }
    }
    
    if (has_nan) {
        ESP_LOGE(TAG, "NaN/Inf detected in weights — skipping upload");
        ESP_LOGI(TAG, "Training complete (weights corrupted)");
        return;
    }
    
    // 7. Upload weights
    uploadWeights(TOTAL_SAMPLES);
    ESP_LOGI(TAG, "Training complete");
}

// ── Run inference (uses TFLite if available, else manual) ─────────────────────
void runInference() {
    // Try to download TFLite model if not done yet
    if (!tflite_model_downloaded) {
        if (downloadTFLiteModel()) {
            initTFLite();
        }
    }
    
    float x[INPUT_DIM];
    if (!read_sensors(x)) return;
    
    float h1[HIDDEN1], h2[HIDDEN2], out_tflite[OUTPUT_DIM], out_manual[OUTPUT_DIM];
    
    // ── TFLite inference with metrics ─────────────────────────────────────────
    if (tflite_initialized) {
        uint32_t heap_before = esp_get_free_heap_size();
        int64_t t0 = esp_timer_get_time();
        
        forward_tflite(x, h1, h2, out_tflite);
        
        int64_t t1 = esp_timer_get_time();
        uint32_t heap_after = esp_get_free_heap_size();
        
        int pred_tflite = 0;
        for (int j = 1; j < OUTPUT_DIM; j++)
            if (out_tflite[j] > out_tflite[pred_tflite]) pred_tflite = j;
        
        ESP_LOGI(TAG, "[TFLite] %s (%.1f%%) [%.3f %.3f %.3f %.3f] | %lld us | heap=%lu bytes",
                 CLASS_NAMES[pred_tflite], 100.0f * out_tflite[pred_tflite],
                 out_tflite[0], out_tflite[1], out_tflite[2], out_tflite[3],
                 (long long)(t1 - t0), (unsigned long)heap_after);
    }
    
    // ── Manual inference with metrics ─────────────────────────────────────────
    {
        uint32_t heap_before = esp_get_free_heap_size();
        int64_t t0 = esp_timer_get_time();
        
        forward_manual(x, h1, h2, out_manual);
        
        int64_t t1 = esp_timer_get_time();
        uint32_t heap_after = esp_get_free_heap_size();
        
        int pred_manual = 0;
        for (int j = 1; j < OUTPUT_DIM; j++)
            if (out_manual[j] > out_manual[pred_manual]) pred_manual = j;
        
        ESP_LOGI(TAG, "[Manual] %s (%.1f%%) [%.3f %.3f %.3f %.3f] | %lld us | heap=%lu bytes",
                 CLASS_NAMES[pred_manual], 100.0f * out_manual[pred_manual],
                 out_manual[0], out_manual[1], out_manual[2], out_manual[3],
                 (long long)(t1 - t0), (unsigned long)heap_after);
    }
}

// ── FL task ───────────────────────────────────────────────────────────────────
void fl_task(void *pvParameters) {
    initWeightsLocal();
    
    // Register with server
    ESP_LOGI(TAG, "Registering with server...");
    while (!registered) {
        registered = httpRegister();
        if (!registered) {
            ESP_LOGI(TAG, "Registration failed, retrying in 3s...");
            vTaskDelay(pdMS_TO_TICKS(3000));
        }
    }
    
    ESP_LOGI(TAG, "Registered! Starting FL loop...");
    
    while (1) {
        switch (clientPhase) {
            case PHASE_IDLE: {
                int serverRound = 0;
                float weights[TOTAL_WEIGHTS];
                char instruction[32];
                
                pollStatus(weights, &serverRound, instruction, sizeof(instruction));
                
                ESP_LOGI(TAG, "IDLE — %s round=%d", instruction, serverRound);
                
                if (strcmp(instruction, "train") == 0) {
                    clientPhase = PHASE_TRAIN;
                } else if (strcmp(instruction, "inference") == 0) {
                    ESP_LOGI(TAG, "INFERENCE MODE");
                    unflattenWeights(weights);
                    currentRound = serverRound;
                    ESP_LOGI(TAG, "Manual weights loaded successfully (round %d)", currentRound);
                    verifyWeightsWithServer();
                    clientPhase = PHASE_INFERENCE;
                } else {
                    vTaskDelay(pdMS_TO_TICKS(POLL_INTERVAL_MS));
                }
                break;
            }
            
            case PHASE_TRAIN: {
                runLocalTraining();
                clientPhase = PHASE_IDLE;
                vTaskDelay(pdMS_TO_TICKS(1000));
                break;
            }
            
            case PHASE_INFERENCE: {
                runInference();
                vTaskDelay(pdMS_TO_TICKS(INFERENCE_INTERVAL_MS));
                
                // Check every 60s if new FL cycle started
                static TickType_t lastCycleCheck = 0;
                if (xTaskGetTickCount() - lastCycleCheck > pdMS_TO_TICKS(60000)) {
                    lastCycleCheck = xTaskGetTickCount();
                    char instruction[32];
                    pollStatus(NULL, NULL, instruction, sizeof(instruction));
                    if (strcmp(instruction, "train") == 0) {
                        ESP_LOGI(TAG, "New cycle detected — resetting TFLite model");
                        tflite_model_downloaded = false;
                        tflite_initialized = false;
                        clientPhase = PHASE_IDLE;
                    }
                }
                break;
            }
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────
extern "C" void app_main(void) {
    // Suppress verbose logs - only show FL_CLIENT messages
    esp_log_level_set("*", ESP_LOG_WARN);
    esp_log_level_set("FL_CLIENT", ESP_LOG_INFO);
    
    ESP_LOGI(TAG, "FL Client — ESP32 (ESP-IDF + TFLite)");
    
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    // Initialize I2C
    i2c_master_init();
    
    // Wake MPU6050
    uint8_t wake_cmd[] = {0x6B, 0x00};
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (0x68 << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write(cmd, wake_cmd, 2, true);
    i2c_master_stop(cmd);
    i2c_master_cmd_begin(I2C_NUM_0, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);
    ESP_LOGI(TAG, "MPU6050 awake");
    
    // Initialize WiFi
    wifi_init_sta();
    
    // Start FL task
    xTaskCreate(fl_task, "fl_task", 32768, NULL, 5, NULL);
    
    // Block forever
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
