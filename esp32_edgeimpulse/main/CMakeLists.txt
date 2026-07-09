idf_component_register(SRCS "main.cpp"
                      INCLUDE_DIRS "."
                      REQUIRES esp_wifi esp_event nvs_flash esp_http_client esp-tflite-micro driver esp_timer)
