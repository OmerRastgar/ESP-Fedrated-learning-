/*
 * Federated Learning Client — ESP32 (ESP-IDF + TFLite + Edge Impulse)
 *
 * Uploads sensor data to local server → forwarded to Edge Impulse.
 * Downloads TFLite model from local server for inference.
 * No manual forward pass — TFLite only.
 * No local SGD training — EI cloud handles training.
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
#include "freertos/semphr.h"
#include "freertos/queue.h"
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

static const char *TAG = "EI_CLIENT";

// ── Config ────────────────────────────────────────────────────────────────────
#define WIFI_SSID      "Cybergaar"
#define WIFI_PASS      "curedata4low"
#define SERVER_IP      "10.219.84.92"     // Local server (middleware)
#define SERVER_PORT    5000
#define CLIENT_ID      "esp32-node-A"

#define POLL_INTERVAL_MS      3000
#define INFERENCE_INTERVAL_MS 500
#define SAMPLE_INTERVAL_MS    200
#define LOCAL_SAMPLES_PER_CLASS 40
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

// ── FreeRTOS synchronization primitives ───────────────────────────────────────
SemaphoreHandle_t tflite_mutex = NULL;

// ── Inter-task communication ──────────────────────────────────────────────────
enum CommandType { CMD_COLLECT, CMD_STOP };
struct Command {
    CommandType type;
};

struct CollectResult {
    int n_samples;
    bool success;
};

QueueHandle_t command_queue = NULL;
QueueHandle_t result_queue = NULL;

// ── Task handles ──────────────────────────────────────────────────────────────
TaskHandle_t network_task_handle = NULL;
TaskHandle_t collect_task_handle = NULL;
TaskHandle_t inference_task_handle = NULL;

// ── Sensor sample buffer (static, shared) ─────────────────────────────────────
static float sensor_samples[ACTIVE_CLASSES * LOCAL_SAMPLES_PER_CLASS][INPUT_DIM] __attribute__((aligned(4)));
static int sensor_labels[ACTIVE_CLASSES * LOCAL_SAMPLES_PER_CLASS] __attribute__((aligned(4)));

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

// ── HTTP helper ───────────────────────────────────────────────────────────────
char server_url[128];

const char* serverUrl(const char* path) {
    snprintf(server_url, sizeof(server_url), "http://%s:%d%s", SERVER_IP, SERVER_PORT, path);
    return server_url;
}

// ── HTTP GET helper ───────────────────────────────────────────────────────────
static char http_response[HTTP_RESPONSE_MAX] __attribute__((aligned(4)));

bool httpGet(const char* url, char* response, int max_len) {
    int64_t http_start = esp_timer_get_time();
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
    
    esp_http_client_fetch_headers(client);
    
    int total_read = 0;
    while (total_read < max_len - 1) {
        int read_len = esp_http_client_read(client, response + total_read, max_len - 1 - total_read);
        if (read_len <= 0) break;
        total_read += read_len;
    }
    response[total_read] = '\0';
    
    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    
    int64_t http_end = esp_timer_get_time();
    ESP_LOGI(TAG, "HTTP GET %d bytes from %s (%lld ms)", total_read, url, (http_end - http_start) / 1000);
    return true;
}

// ── HTTP POST helper ──────────────────────────────────────────────────────────
int httpPost(const char* url, const char* body, char* response, int max_len) {
    int64_t http_start = esp_timer_get_time();
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
    
    int64_t http_end = esp_timer_get_time();
    ESP_LOGI(TAG, "HTTP POST %s status=%d (%lld ms)", url, status, (http_end - http_start) / 1000);
    return status;
}

// ── Upload sensor data to local server (for Edge Impulse) ─────────────────────
bool uploadSensorData(const char* label, float samples[][INPUT_DIM], int num_samples) {
    SimpleString body(8192);
    
    body.append("{\"client_id\":\"");
    body.append(CLIENT_ID);
    body.append("\",\"label\":\"");
    body.append(label);
    body.append("\",\"interval_ms\":");
    body.append(SAMPLE_INTERVAL_MS);
    body.append(",\"sensors\":[");
    body.append("{\"name\":\"accX\",\"units\":\"m/s2\"},");
    body.append("{\"name\":\"accY\",\"units\":\"m/s2\"},");
    body.append("{\"name\":\"accZ\",\"units\":\"m/s2\"},");
    body.append("{\"name\":\"gyroX\",\"units\":\"deg/s\"},");
    body.append("{\"name\":\"gyroY\",\"units\":\"deg/s\"},");
    body.append("{\"name\":\"gyroZ\",\"units\":\"deg/s\"},");
    body.append("{\"name\":\"r1\",\"units\":\"m/s2\"},");
    body.append("{\"name\":\"r2\",\"units\":\"m/s2\"},");
    body.append("{\"name\":\"r3\",\"units\":\"m/s2\"}");
    body.append("],\"values\":[");
    
    for (int i = 0; i < num_samples; i++) {
        body.append("[");
        for (int j = 0; j < INPUT_DIM; j++) {
            body.append(samples[i][j], 4);
            if (j < INPUT_DIM - 1) body.append(",");
        }
        body.append("]");
        if (i < num_samples - 1) body.append(",");
    }
    body.append("]}");
    
    char response[512];
    int status = httpPost(serverUrl("/upload_data"), body.data, response, sizeof(response));
    
    if (status == 200) {
        ESP_LOGI(TAG, "Upload OK: %s (%d samples)", label, num_samples);
        return true;
    }
    ESP_LOGW(TAG, "Upload FAILED HTTP %d — %s", status, response);
    return false;
}

// ── Poll server for status ────────────────────────────────────────────────────
bool pollStatus(char* instruction, int instr_max) {
    char url[256];
    snprintf(url, sizeof(url), "%s?client_id=%s", serverUrl("/status"), CLIENT_ID);
    
    if (!httpGet(url, http_response, sizeof(http_response))) {
        strncpy(instruction, "wait", instr_max);
        return false;
    }
    
    const char* ii = strstr(http_response, "\"instruction\":\"");
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
    
    return true;
}

// ── Download TFLite model from local server ───────────────────────────────────
bool downloadModelFromServer() {
    ESP_LOGI(TAG, "Downloading TFLite model from server...");
    
    esp_http_client_config_t config = {};
    config.url = serverUrl("/get_model");
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
    
    if (read_len < 2000 || tflite_model_buffer[0] != 0x1c) {
        ESP_LOGE(TAG, "Invalid model: %d bytes, first byte=0x%02x", read_len, tflite_model_buffer[0]);
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

// ── TFLite forward pass ───────────────────────────────────────────────────────
void forward_tflite(const float* x, float* out) {
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
}

// ═══════════════════════════════════════════════════════════════════════════════
// NETWORK TASK — Core 0, Priority 5
// Polls server for instructions, triggers collection/inference
// ═══════════════════════════════════════════════════════════════════════════════
void network_task(void *pvParameters) {
    ESP_LOGI(TAG, "Network task started");
    
    bool in_inference = false;
    
    while (1) {
        // Log stack and heap status
        UBaseType_t stack_remaining = uxTaskGetStackHighWaterMark(NULL);
        ESP_LOGI(TAG, "Network stack: %lu/%d bytes (%lu%% free) | Heap: %lu bytes",
                 stack_remaining * sizeof(StackType_t), 8192,
                 (stack_remaining * sizeof(StackType_t) * 100) / 8192,
                 esp_get_free_heap_size());
        
        // Poll server for status
        char instruction[32];
        pollStatus(instruction, sizeof(instruction));
        ESP_LOGI(TAG, "Network: %s", instruction);
        
        if (strcmp(instruction, "train") == 0) {
            in_inference = false;
            
            // Send collect command to collect task
            Command cmd;
            cmd.type = CMD_COLLECT;
            if (xQueueSend(command_queue, &cmd, pdMS_TO_TICKS(1000)) == pdTRUE) {
                ESP_LOGI(TAG, "Network: Sent collect command");
                
                // Wait for collection to complete
                CollectResult result;
                if (xQueueReceive(result_queue, &result, pdMS_TO_TICKS(300000))) {
                    ESP_LOGI(TAG, "Network: Collection complete (%d samples)", result.n_samples);
                } else {
                    ESP_LOGW(TAG, "Network: Timeout waiting for collection");
                }
            }
            
        } else if (strcmp(instruction, "inference") == 0) {
            if (!in_inference) {
                ESP_LOGI(TAG, "Network: Entering inference mode");
                
                // Download TFLite model
                if (downloadModelFromServer()) {
                    initTFLite();
                }
                
                // Notify inference task to start
                xTaskNotify(inference_task_handle, 0, eNoAction);
                in_inference = true;
            }
        }
        
        vTaskDelay(pdMS_TO_TICKS(POLL_INTERVAL_MS));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COLLECT TASK — Core 1, Priority 3
// Collects sensor data per class and uploads to local server
// ═══════════════════════════════════════════════════════════════════════════════
void collect_task(void *pvParameters) {
    ESP_LOGI(TAG, "Collect task started");
    
    while (1) {
        // Wait for collect command
        Command cmd;
        if (xQueueReceive(command_queue, &cmd, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        
        int64_t collect_start = esp_timer_get_time();
        ESP_LOGI(TAG, "Collect task: Starting data collection");
        
        // Log task metrics
        UBaseType_t stack_start = uxTaskGetStackHighWaterMark(NULL);
        ESP_LOGI(TAG, "Collect stack: %lu/%d bytes (%lu%% free) | Heap: %lu bytes",
                 stack_start * sizeof(StackType_t), 8192,
                 (stack_start * sizeof(StackType_t) * 100) / 8192,
                 esp_get_free_heap_size());
        
        // Collect samples per class
        int total_collected = 0;
        bool all_success = true;
        
        for (int cls = 0; cls < ACTIVE_CLASSES; cls++) {
            ESP_LOGI(TAG, "Collect task: Class %d/%d: %s", 
                     cls + 1, ACTIVE_CLASSES, CLASS_NAMES[cls]);
            ESP_LOGI(TAG, "Perform %s now! Collecting %d samples...", 
                     CLASS_NAMES[cls], LOCAL_SAMPLES_PER_CLASS);
            
            // Collect samples for this class
            float class_samples[LOCAL_SAMPLES_PER_CLASS][INPUT_DIM];
            int collected = 0;
            
            while (collected < LOCAL_SAMPLES_PER_CLASS) {
                float s[INPUT_DIM];
                if (read_sensors(s)) {
                    memcpy(class_samples[collected], s, sizeof(s));
                    collected++;
                    if (collected % 10 == 0)
                        ESP_LOGI(TAG, "  %d / %d", collected, LOCAL_SAMPLES_PER_CLASS);
                }
                vTaskDelay(pdMS_TO_TICKS(SAMPLE_INTERVAL_MS));
            }
            
            ESP_LOGI(TAG, "Collected %d samples for %s", collected, CLASS_NAMES[cls]);
            
            // Upload to local server immediately
            if (!uploadSensorData(CLASS_NAMES[cls], class_samples, collected)) {
                ESP_LOGE(TAG, "Failed to upload %s data", CLASS_NAMES[cls]);
                all_success = false;
            }
            
            total_collected += collected;
            
            // Pause between classes (except last)
            if (cls < ACTIVE_CLASSES - 1) {
                ESP_LOGI(TAG, "Next class in %d s...", CLASS_TRANSITION_DELAY_MS / 1000);
                vTaskDelay(pdMS_TO_TICKS(CLASS_TRANSITION_DELAY_MS));
            }
        }
        
        // Send result to network task
        CollectResult result;
        result.n_samples = total_collected;
        result.success = all_success;
        xQueueSend(result_queue, &result, pdMS_TO_TICKS(1000));
        
        // Log final metrics
        int64_t collect_end = esp_timer_get_time();
        UBaseType_t stack_end = uxTaskGetStackHighWaterMark(NULL);
        ESP_LOGI(TAG, "Collection complete | Total: %lld ms | Samples: %d | Stack: %lu/%d bytes (%lu%% free) | Heap: %lu bytes",
                 (long long)(collect_end - collect_start) / 1000,
                 total_collected,
                 stack_end * sizeof(StackType_t), 8192,
                 (stack_end * sizeof(StackType_t) * 100) / 8192,
                 esp_get_free_heap_size());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INFERENCE TASK — Core 1, Priority 4
// Runs TFLite inference every 500ms
// ═══════════════════════════════════════════════════════════════════════════════
void inference_task(void *pvParameters) {
    // Wait for notification from network task
    ESP_LOGI(TAG, "Inference task: Waiting for model...");
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    ESP_LOGI(TAG, "Inference task: Starting inference");
    
    while (1) {
        // Try to download TFLite model if not done yet
        if (!tflite_model_downloaded) {
            ESP_LOGI(TAG, "Inference task: Downloading TFLite model...");
            if (downloadModelFromServer()) {
                initTFLite();
            }
        }
        
        // Read sensor
        float x[INPUT_DIM];
        if (!read_sensors(x)) {
            vTaskDelay(pdMS_TO_TICKS(INFERENCE_INTERVAL_MS));
            continue;
        }
        
        // TFLite inference with full metrics
        xSemaphoreTake(tflite_mutex, 0);
        if (tflite_initialized) {
            float out[OUTPUT_DIM];
            
            uint32_t heap_before = esp_get_free_heap_size();
            int64_t t0 = esp_timer_get_time();
            
            forward_tflite(x, out);
            
            int64_t t1 = esp_timer_get_time();
            uint32_t heap_after = esp_get_free_heap_size();
            
            int pred = 0;
            for (int j = 1; j < OUTPUT_DIM; j++)
                if (out[j] > out[pred]) pred = j;
            
            ESP_LOGI(TAG, "[TFLite] %s (%.1f%%) [%.3f %.3f %.3f %.3f] | %lld us | heap=%lu bytes",
                     CLASS_NAMES[pred], 100.0f * out[pred],
                     out[0], out[1], out[2], out[3],
                     (long long)(t1 - t0), (unsigned long)heap_after);
        }
        xSemaphoreGive(tflite_mutex);
        
        // Log inference task stack and system heap
        UBaseType_t stack_remaining = uxTaskGetStackHighWaterMark(NULL);
        ESP_LOGI(TAG, "Inference stack: %lu/%d bytes (%lu%% free) | Heap: %lu bytes",
                 stack_remaining * sizeof(StackType_t), 4096,
                 (stack_remaining * sizeof(StackType_t) * 100) / 4096,
                 esp_get_free_heap_size());
        
        // Check for notification (new model?)
        uint32_t notified = 0;
        xTaskNotifyWait(0, 0, &notified, 0);
        if (notified == 1) {
            ESP_LOGI(TAG, "Inference task: New model notification, resetting");
            tflite_model_downloaded = false;
            tflite_initialized = false;
            ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        }
        
        vTaskDelay(pdMS_TO_TICKS(INFERENCE_INTERVAL_MS));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN — Initialization and task creation
// ═══════════════════════════════════════════════════════════════════════════════
extern "C" void app_main(void) {
    // Suppress verbose logs
    esp_log_level_set("*", ESP_LOG_WARN);
    esp_log_level_set("EI_CLIENT", ESP_LOG_INFO);
    
    ESP_LOGI(TAG, "FL Client — ESP32 (ESP-IDF + TFLite + Edge Impulse)");
    
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
    
    // Create mutexes
    tflite_mutex = xSemaphoreCreateMutex();
    if (tflite_mutex == NULL) {
        ESP_LOGE(TAG, "Failed to create mutex");
        return;
    }
    
    // Create queues
    command_queue = xQueueCreate(1, sizeof(Command));
    result_queue = xQueueCreate(1, sizeof(CollectResult));
    
    if (command_queue == NULL || result_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create queues");
        return;
    }
    
    ESP_LOGI(TAG, "FreeRTOS primitives created");
    
    // Initialize WiFi
    wifi_init_sta();
    ESP_LOGI(TAG, "WiFi initialized");
    
    // Create tasks
    xTaskCreatePinnedToCore(network_task, "network", 8192, NULL, 5, &network_task_handle, 0);
    xTaskCreatePinnedToCore(collect_task, "collect", 8192, NULL, 3, &collect_task_handle, 1);
    xTaskCreatePinnedToCore(inference_task, "inference", 4096, NULL, 4, &inference_task_handle, 1);
    
    ESP_LOGI(TAG, "All tasks created");
    
    // Block forever
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
