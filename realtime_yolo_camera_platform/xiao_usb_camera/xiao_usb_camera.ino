/*
 * XIAO ESP32S3 Sense
 * 開機先用 USB 送 JPEG。網頁送來 WIFI 指令後才加入 2.4GHz，並提供 :81/stream。
 * 開發板：XIAO_ESP32S3，PSRAM：OPI PSRAM，USB CDC On Boot：Enabled
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_camera.h>
#include <esp_http_server.h>

#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 10
#define SIOD_GPIO_NUM 40
#define SIOC_GPIO_NUM 39
#define Y9_GPIO_NUM 48
#define Y8_GPIO_NUM 11
#define Y7_GPIO_NUM 12
#define Y6_GPIO_NUM 14
#define Y5_GPIO_NUM 16
#define Y4_GPIO_NUM 18
#define Y3_GPIO_NUM 17
#define Y2_GPIO_NUM 15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM 47
#define PCLK_GPIO_NUM 13

static bool cameraReady = false;
static bool serverStarted = false;
static bool joining = false;
static uint32_t joinStart = 0;
static char line[120];
static size_t lineLen = 0;

bool startCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 16;
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;
  return esp_camera_init(&config) == ESP_OK;
}

static const char *STREAM_TYPE = "multipart/x-mixed-replace;boundary=frame";
static const char *STREAM_BOUNDARY = "\r\n--frame\r\n";
static const char *STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t streamHandler(httpd_req_t *req) {
  char part[64];
  httpd_resp_set_type(req, STREAM_TYPE);
  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      delay(20);
      continue;
    }
    const size_t len = snprintf(part, sizeof(part), STREAM_PART, fb->len);
    esp_err_t res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, part, len);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    }
    esp_camera_fb_return(fb);
    if (res != ESP_OK) {
      break;
    }
    delay(30);
  }
  return ESP_OK;
}

void startStreamServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;
  config.ctrl_port = 32769;
  config.max_open_sockets = 3;
  config.stack_size = 8192;
  httpd_handle_t server = nullptr;
  if (httpd_start(&server, &config) != ESP_OK) {
    Serial.println("WIFI fail server");
    return;
  }
  httpd_uri_t uri = {};
  uri.uri = "/stream";
  uri.method = HTTP_GET;
  uri.handler = streamHandler;
  httpd_register_uri_handler(server, &uri);
  serverStarted = true;
}

void beginJoin(const char *ssid, const char *password) {
  joining = false;
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.disconnect(false, false);
  delay(50);
  const int count = WiFi.scanNetworks(false, true, false, 250);
  int channel = 0;
  uint8_t bssid[6] = {};
  bool seen = false;
  int best = -127;
  for (int i = 0; i < count; i++) {
    if (WiFi.SSID(i) == ssid && WiFi.RSSI(i) > best) {
      best = WiFi.RSSI(i);
      channel = WiFi.channel(i);
      memcpy(bssid, WiFi.BSSID(i), 6);
      seen = true;
    }
  }
  if (seen) {
    Serial.printf("WIFI seen rssi %d ch %d\n", best, channel);
    WiFi.begin(ssid, password, channel, bssid);
  } else {
    Serial.println("WIFI ssid not in scan");
    WiFi.begin(ssid, password);
  }
  WiFi.scanDelete();
  joining = true;
  joinStart = millis();
  Serial.printf("WIFI joining %s\n", ssid);
}

void handleLine(char *text) {
  if (strncmp(text, "WIFI\t", 5) != 0) {
    return;
  }
  char *ssid = text + 5;
  char *password = strchr(ssid, '\t');
  if (password == nullptr) {
    Serial.println("WIFI fail empty");
    return;
  }
  *password = '\0';
  password++;
  if (ssid[0] == '\0') {
    Serial.println("WIFI fail empty");
    return;
  }
  beginJoin(ssid, password);
}

void pollCommand() {
  while (Serial.available() > 0) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\n') {
      line[lineLen] = '\0';
      handleLine(line);
      lineLen = 0;
    } else if (c != '\r' && lineLen + 1 < sizeof(line)) {
      line[lineLen++] = c;
    }
  }
}

void pollJoin() {
  if (!joining) {
    return;
  }
  if (WiFi.status() == WL_CONNECTED) {
    joining = false;
    if (!serverStarted) {
      startStreamServer();
    }
    if (serverStarted) {
      Serial.printf("MODE wifi %s\n", WiFi.localIP().toString().c_str());
      Serial.printf("STREAM http://%s:81/stream\n", WiFi.localIP().toString().c_str());
    }
    return;
  }
  if (millis() - joinStart > 20000) {
    joining = false;
    Serial.printf("WIFI fail %d\n", static_cast<int>(WiFi.status()));
    WiFi.disconnect(false, false);
  }
}

void sendUsbFrame(camera_fb_t *fb) {
  const uint32_t len = fb->len;
  Serial.write("JPGS", 4);
  Serial.write(reinterpret_cast<const uint8_t *>(&len), 4);
  Serial.write(fb->buf, fb->len);
}

void setup() {
  Serial.setRxBufferSize(1024);
  Serial.setTxBufferSize(32768);
  Serial.begin(115200);
  delay(200);
  Serial.println("XIAO USB camera");
  cameraReady = psramFound() && startCamera();
  if (!cameraReady) {
    Serial.println("CAMERA FAILED");
    return;
  }
  Serial.println("MODE usb");
}

void loop() {
  pollCommand();
  pollJoin();
  if (!cameraReady) {
    delay(200);
    return;
  }
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    delay(20);
    return;
  }
  sendUsbFrame(fb);
  esp_camera_fb_return(fb);
  delay(40);
}
