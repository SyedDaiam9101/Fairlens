/*
 * Detectify Sentinel - ESP32 Security Monitor Firmware
 * 
 * Target: ESP32 Dev Module
 * 
 * Hardware:
 *   - PIR1: GPIO 32
 *   - PIR2: GPIO 33
 *   - HC-SR04 Trig: GPIO 13
 *   - HC-SR04 Echo: GPIO 14
 *   - Buzzer: GPIO 12
 *   - External Webcam: Controlled via HTTP request to local server
 * 
 * Features:
 *   - Dual PIR motion detection with 10s trespassing watch
 *   - Ultrasonic distance monitoring (alerts at <30cm, alarm at <15cm)
 *   - Non-blocking timing for concurrent sensor polling
 *   - HTTPS POST to server with JSON payload
 *   - Email trigger support via server response
 */

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

// WiFi Credentials
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Server Configuration
const char* SERVER_HOST = "192.168.43.201";
const int SERVER_PORT = 8001;
const char* API_ALERTS_ENDPOINT = "/api/alerts";
const char* API_CAPTURE_ENDPOINT = "/api/capture";

// Email recipient (sent to server for relay)
const char* EMAIL_RECIPIENT = "smdaiam616@gmail.com";

// Pin Assignments
const int PIR1_PIN = 32;
const int PIR2_PIN = 33;
const int TRIG_PIN = 13;
const int ECHO_PIN = 14;
const int BUZZER_PIN = 12;

// Timing Constants (milliseconds)
const unsigned long MOTION_TIMEOUT_MS = 10000;       // 10 seconds before capture
const unsigned long DISTANCE_POLL_INTERVAL_MS = 100; // Poll ultrasonic every 100ms
const unsigned long BUZZER_DURATION_MS = 200;        // Buzzer on time
const unsigned long WIFI_RETRY_INTERVAL_MS = 5000;   // WiFi reconnect interval
const unsigned long SERIAL_BAUD = 115200;

// Distance Thresholds (cm)
const float ALERT_DISTANCE_CM = 30.0;
const float ALARM_DISTANCE_CM = 15.0;
const float MAX_DISTANCE_CM = 20.0;  // Max measurable distance

// ============================================================================
// GLOBAL STATE
// ============================================================================

// Motion detection state
volatile bool pir1Triggered = false;
volatile bool pir2Triggered = false;
bool motionActive = false;
unsigned long motionStartTime = 0;
bool captureRequested = false;
bool captureSent = false;

// Distance measurement
float currentDistanceCm = 0.0;
unsigned long lastDistancePoll = 0;
float distanceBuffer[5] = {0};
int distanceBufferIdx = 0;

// Buzzer state (non-blocking)
bool buzzerActive = false;
unsigned long buzzerStartTime = 0;

// Alert flags
bool alertFlag = false;
bool immediateAlarmFlag = false;

// Confidence score (baseline, updated by server)
int humanConfidenceScore = 50;

// WiFi state
unsigned long lastWiFiCheck = 0;
bool wifiConnected = false;

// ============================================================================
// INTERRUPT SERVICE ROUTINES
// ============================================================================

void IRAM_ATTR isr_pir1() {
    pir1Triggered = true;
}

void IRAM_ATTR isr_pir2() {
    pir2Triggered = true;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

void logMessage(const char* level, const char* message) {
    unsigned long ts = millis();
    Serial.printf("[%lu][%s] %s\n", ts, level, message);
}

void logMessageF(const char* level, const char* format, ...) {
    char buffer[256];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    logMessage(level, buffer);
}

// ============================================================================
// WIFI FUNCTIONS
// ============================================================================

void setupWiFi() {
    logMessage("INFO", "Connecting to WiFi...");
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        wifiConnected = true;
        logMessageF("INFO", "WiFi Connected! IP: %s, RSSI: %d dBm", 
                    WiFi.localIP().toString().c_str(), WiFi.RSSI());
    } else {
        wifiConnected = false;
        logMessage("ERROR", "WiFi connection failed!");
    }
}

void checkWiFiConnection() {
    unsigned long now = millis();
    if (now - lastWiFiCheck > WIFI_RETRY_INTERVAL_MS) {
        lastWiFiCheck = now;
        
        if (WiFi.status() != WL_CONNECTED) {
            if (wifiConnected) {
                logMessage("WARN", "WiFi disconnected! Attempting reconnection...");
                wifiConnected = false;
            }
            WiFi.disconnect();
            WiFi.reconnect();
            
            // Quick check
            delay(1000);
            if (WiFi.status() == WL_CONNECTED) {
                wifiConnected = true;
                logMessageF("INFO", "WiFi Reconnected! IP: %s", WiFi.localIP().toString().c_str());
            }
        } else if (!wifiConnected) {
            wifiConnected = true;
            logMessage("INFO", "WiFi connection restored");
        }
    }
}

// ============================================================================
// SENSOR FUNCTIONS
// ============================================================================

float measureDistance() {
    // Send trigger pulse
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
    
    // Measure echo duration (timeout 30ms = ~5m max)
    long duration = pulseIn(ECHO_PIN, HIGH, 30000);
    
    if (duration == 0) {
        return -1.0; // No echo received
    }
    
    // Calculate distance: speed of sound = 0.0343 cm/us
    float distance = (duration * 0.0343) / 2.0;
    
    // Validate range
    if (distance > 400.0 || distance < 2.0) {
        return -1.0; // Out of valid range
    }
    
    return distance;
}

float getFilteredDistance() {
    // Get raw reading
    float raw = measureDistance();
    
    if (raw < 0) {
        return currentDistanceCm; // Return last valid reading
    }
    
    // Add to circular buffer
    distanceBuffer[distanceBufferIdx] = raw;
    distanceBufferIdx = (distanceBufferIdx + 1) % 5;
    
    // Calculate moving average
    float sum = 0;
    int count = 0;
    for (int i = 0; i < 5; i++) {
        if (distanceBuffer[i] > 0) {
            sum += distanceBuffer[i];
            count++;
        }
    }
    
    return (count > 0) ? (sum / count) : raw;
}

void pollDistanceSensor() {
    unsigned long now = millis();
    if (now - lastDistancePoll >= DISTANCE_POLL_INTERVAL_MS) {
        lastDistancePoll = now;
        
        currentDistanceCm = getFilteredDistance();
        
        // Reset flags
        alertFlag = false;
        immediateAlarmFlag = false;
        
        // Check thresholds
        if (currentDistanceCm > 0 && currentDistanceCm < ALARM_DISTANCE_CM) {
            // Immediate alarm: distance < 15cm
            alertFlag = true;
            immediateAlarmFlag = true;
            activateBuzzer();
            logMessageF("ALARM", "IMMEDIATE ALARM! Distance: %.1f cm", currentDistanceCm);
        } else if (currentDistanceCm > 0 && currentDistanceCm < ALERT_DISTANCE_CM) {
            // Alert: distance < 30cm
            alertFlag = true;
            logMessageF("ALERT", "Proximity alert! Distance: %.1f cm", currentDistanceCm);
        }
    }
}

// ============================================================================
// BUZZER FUNCTIONS
// ============================================================================

void activateBuzzer() {
    if (!buzzerActive) {
        tone(BUZZER_PIN, 2000); // 2kHz tone
        buzzerStartTime = millis();
        buzzerActive = true;
    }
}

void handleBuzzer() {
    if (buzzerActive) {
        if (millis() - buzzerStartTime >= BUZZER_DURATION_MS) {
            noTone(BUZZER_PIN);
            digitalWrite(BUZZER_PIN, LOW);
            buzzerActive = false;
        }
    }
}

// ============================================================================
// MOTION DETECTION
// ============================================================================

void handleMotionDetection() {
    unsigned long now = millis();
    
    // Check for new motion triggers
    if (pir1Triggered || pir2Triggered) {
        if (!motionActive) {
            // Start new motion event
            motionActive = true;
            motionStartTime = now;
            captureRequested = false;
            captureSent = false;
            
            logMessageF("MOTION", "Motion detected! PIR1: %d, PIR2: %d - Starting 10s watch", 
                        pir1Triggered ? 1 : 0, pir2Triggered ? 1 : 0);
        } else {
            // Extend motion event
            motionStartTime = now;
        }
        
        // Clear interrupt flags
        pir1Triggered = false;
        pir2Triggered = false;
    }
    
    // Handle motion timeout and capture
    if (motionActive) {
        unsigned long elapsed = now - motionStartTime;
        
        // After 10 seconds of motion, request capture
        if (elapsed >= MOTION_TIMEOUT_MS && !captureRequested) {
            captureRequested = true;
            logMessage("INFO", "10 seconds elapsed - requesting image capture");
        }
        
        // Reset motion state if no new triggers for 5 seconds past capture
        if (elapsed > MOTION_TIMEOUT_MS + 5000 && captureSent) {
            motionActive = false;
            captureRequested = false;
            captureSent = false;
            logMessage("INFO", "Motion event ended");
        }
    }
}

// ============================================================================
// HTTP COMMUNICATION
// ============================================================================

bool requestServerCapture() {
    /*
     * Request the Python server to capture an image from connected USB webcam.
     * The server handles the actual capture and returns a reference or base64 data.
     */
    if (!wifiConnected) {
        logMessage("ERROR", "Cannot request capture - WiFi disconnected");
        return false;
    }
    
    HTTPClient http;
    WiFiClient client;
    
    char url[128];
    snprintf(url, sizeof(url), "http://%s:%d%s", SERVER_HOST, SERVER_PORT, API_CAPTURE_ENDPOINT);
    
    logMessageF("HTTP", "Requesting capture from: %s", url);
    
    if (!http.begin(client, url)) {
        logMessage("ERROR", "HTTP begin failed");
        return false;
    }
    
    http.addHeader("Content-Type", "application/json");
    
    // Build request JSON
    StaticJsonDocument<256> doc;
    doc["action"] = "capture";
    doc["camera_id"] = 0; // Request from camera 0
    doc["reason"] = "motion_timeout";
    
    String jsonPayload;
    serializeJson(doc, jsonPayload);
    
    int httpCode = http.POST(jsonPayload);
    
    if (httpCode > 0) {
        logMessageF("HTTP", "Capture request response: %d", httpCode);
        
        if (httpCode == HTTP_CODE_OK) {
            String response = http.getString();
            
            // Parse response for image reference
            StaticJsonDocument<512> respDoc;
            DeserializationError error = deserializeJson(respDoc, response);
            
            if (!error) {
                const char* status = respDoc["status"] | "unknown";
                logMessageF("INFO", "Capture status: %s", status);
            }
            
            http.end();
            return true;
        }
    } else {
        logMessageF("ERROR", "Capture request failed: %s", http.errorToString(httpCode).c_str());
    }
    
    http.end();
    return false;
}

bool sendAlertToServer(const char* imageBase64) {
    /*
     * Send alert data to the server including sensor readings and confidence score.
     * imageBase64 can be NULL if no image is available.
     */
    if (!wifiConnected) {
        logMessage("ERROR", "Cannot send alert - WiFi disconnected");
        return false;
    }
    
    HTTPClient http;
    WiFiClient client;
    
    char url[128];
    snprintf(url, sizeof(url), "http://%s:%d%s", SERVER_HOST, SERVER_PORT, API_ALERTS_ENDPOINT);
    
    logMessageF("HTTP", "POST to: %s", url);
    
    if (!http.begin(client, url)) {
        logMessage("ERROR", "HTTP begin failed");
        return false;
    }
    
    http.addHeader("Content-Type", "application/json");
    http.addHeader("User-Agent", "ESP32-Detectify-Sentinel");
    http.setTimeout(30000); // 30 second timeout for large payloads
    
    // Build JSON payload
    size_t capacity = 1024;
    if (imageBase64 != NULL) {
        capacity += strlen(imageBase64);
    }
    
    DynamicJsonDocument doc(capacity);
    
    doc["timestamp"] = millis();
    doc["pir1_triggered"] = digitalRead(PIR1_PIN) == HIGH;
    doc["pir2_triggered"] = digitalRead(PIR2_PIN) == HIGH;
    doc["distance_cm"] = serialized(String(currentDistanceCm, 2));
    doc["alert_flag"] = alertFlag;
    doc["immediate_alarm"] = immediateAlarmFlag;
    doc["confidence_score"] = humanConfidenceScore;
    doc["email_recipient"] = EMAIL_RECIPIENT;
    
    if (imageBase64 != NULL) {
        doc["image_data"] = imageBase64;
    } else {
        doc["image_data"] = nullptr;
    }
    
    // Determine alert level string
    const char* alertLevel = "none";
    if (immediateAlarmFlag) {
        alertLevel = "critical";
    } else if (alertFlag) {
        alertLevel = "warning";
    } else if (motionActive) {
        alertLevel = "motion";
    }
    doc["alert_level"] = alertLevel;
    
    String jsonPayload;
    serializeJson(doc, jsonPayload);
    
    logMessageF("HTTP", "Payload size: %d bytes", jsonPayload.length());
    
    unsigned long startTime = millis();
    int httpCode = http.POST(jsonPayload);
    unsigned long responseTime = millis() - startTime;
    
    bool success = false;
    
    if (httpCode > 0) {
        logMessageF("HTTP", "Response: %d, Time: %lu ms", httpCode, responseTime);
        
        if (httpCode == HTTP_CODE_OK) {
            String response = http.getString();
            
            // Parse server response
            StaticJsonDocument<256> respDoc;
            DeserializationError error = deserializeJson(respDoc, response);
            
            if (!error) {
                // Check for email trigger
                bool emailTrigger = respDoc["email"] | respDoc["email_trigger"] | false;
                if (emailTrigger) {
                    logMessage("INFO", "Server requested email notification");
                    // Email will be sent by server, just log it
                }
                
                // Update confidence score from server ML model
                if (respDoc.containsKey("confidence_score")) {
                    int newScore = respDoc["confidence_score"];
                    if (newScore >= 20 && newScore <= 100) {
                        humanConfidenceScore = newScore;
                        logMessageF("INFO", "Confidence score updated: %d%%", humanConfidenceScore);
                    }
                }
            }
            
            success = true;
        }
    } else {
        logMessageF("ERROR", "HTTP POST failed: %s", http.errorToString(httpCode).c_str());
    }
    
    http.end();
    return success;
}

// ============================================================================
// MANUAL SNAPSHOT (Serial 'S' key)
// ============================================================================

void handleSerialCommands() {
    if (Serial.available() > 0) {
        char cmd = Serial.read();
        
        if (cmd == 'S' || cmd == 's') {
            logMessage("USER", "Manual snapshot requested via Serial");
            
            // Request capture from server
            if (requestServerCapture()) {
                logMessage("INFO", "Manual snapshot captured successfully");
                
                // Send alert with current sensor data
                sendAlertToServer(NULL);
            } else {
                logMessage("ERROR", "Manual snapshot failed");
            }
        } else if (cmd == 'D' || cmd == 'd') {
            // Debug info
            logMessageF("DEBUG", "PIR1: %d, PIR2: %d, Distance: %.1f cm", 
                        digitalRead(PIR1_PIN), digitalRead(PIR2_PIN), currentDistanceCm);
            logMessageF("DEBUG", "Motion: %d, Alert: %d, Alarm: %d, Confidence: %d%%",
                        motionActive ? 1 : 0, alertFlag ? 1 : 0, 
                        immediateAlarmFlag ? 1 : 0, humanConfidenceScore);
            logMessageF("DEBUG", "Free heap: %d bytes", ESP.getFreeHeap());
        } else if (cmd == 'W' || cmd == 'w') {
            // WiFi status
            logMessageF("DEBUG", "WiFi Status: %s, RSSI: %d dBm", 
                        wifiConnected ? "Connected" : "Disconnected", WiFi.RSSI());
        }
    }
}

// ============================================================================
// MAIN SETUP
// ============================================================================

void setup() {
    // Initialize Serial
    Serial.begin(SERIAL_BAUD);
    while (!Serial && millis() < 3000) {
        ; // Wait for Serial (max 3 seconds)
    }
    
    Serial.println();
    Serial.println("========================================");
    Serial.println("  Detectify Sentinel - ESP32 Firmware");
    Serial.println("========================================");
    Serial.println();
    
    // Configure pins
    pinMode(PIR1_PIN, INPUT_PULLDOWN);
    pinMode(PIR2_PIN, INPUT_PULLDOWN);
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    
    digitalWrite(TRIG_PIN, LOW);
    digitalWrite(BUZZER_PIN, LOW);
    
    logMessage("INFO", "Pins configured");
    
    // Attach interrupts for PIR sensors
    attachInterrupt(digitalPinToInterrupt(PIR1_PIN), isr_pir1, RISING);
    attachInterrupt(digitalPinToInterrupt(PIR2_PIN), isr_pir2, RISING);
    
    logMessage("INFO", "Interrupts attached");
    
    // Initialize WiFi
    setupWiFi();
    
    // Initial distance reading
    for (int i = 0; i < 5; i++) {
        distanceBuffer[i] = measureDistance();
        delay(50);
    }
    currentDistanceCm = getFilteredDistance();
    
    logMessageF("INFO", "Initial distance: %.1f cm", currentDistanceCm);
    
    Serial.println();
    logMessage("INFO", "System ready! Press 'S' for manual snapshot, 'D' for debug info");
    Serial.println();
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
    unsigned long now = millis();
    
    // 1. Handle Serial commands
    handleSerialCommands();
    
    // 2. Check WiFi connection
    checkWiFiConnection();
    
    // 3. Poll distance sensor (non-blocking)
    pollDistanceSensor();
    
    // 4. Handle buzzer timing (non-blocking)
    handleBuzzer();
    
    // 5. Handle motion detection and capture timing
    handleMotionDetection();
    
    // 6. Process capture request
    if (captureRequested && !captureSent) {
        logMessage("INFO", "Processing capture request...");
        
        // Request server to capture from USB webcam
        if (requestServerCapture()) {
            // Send alert with sensor data
            if (sendAlertToServer(NULL)) {
                captureSent = true;
                logMessage("INFO", "Alert sent successfully");
            }
        }
    }
    
    // 7. Send alerts for distance-based events (debounced)
    static unsigned long lastAlertSend = 0;
    if ((alertFlag || immediateAlarmFlag) && (now - lastAlertSend > 5000)) {
        lastAlertSend = now;
        
        logMessage("INFO", "Sending distance-based alert...");
        sendAlertToServer(NULL);
    }
    
    // 8. Periodic sensor logging (every 5 seconds)
    static unsigned long lastLog = 0;
    if (now - lastLog > 5000) {
        lastLog = now;
        
        logMessageF("SENSOR", "PIR1: %d, PIR2: %d, Distance: %.1f cm, RSSI: %d dBm",
                    digitalRead(PIR1_PIN), digitalRead(PIR2_PIN), 
                    currentDistanceCm, WiFi.RSSI());
    }
    
    // Small yield for WiFi stack
    yield();
}
