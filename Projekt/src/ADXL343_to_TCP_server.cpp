#include "Particle.h"
#include "adxl343.h"

SYSTEM_THREAD(ENABLED);
SYSTEM_MODE(AUTOMATIC);

SerialLogHandler logHandler;

const unsigned long MAX_RECORDING_LENGTH_MS = 3000;
const int SAMPLING_INTERVAL_MS = 10;
const int BUFFER_SIZE = 25;
const int TRANSMIT_THRESHOLD = 20;

IPAddress serverAddr = IPAddress(172,20,10,3);
int serverPort = 7123;

TCPClient client;

struct Sample
{
  uint32_t timestamp;
  float xa, ya, za;
  float xl, yl, zl;
};

Sample buffer1[BUFFER_SIZE];
Sample buffer2[BUFFER_SIZE];

Sample *samplingBuffer = buffer1;
Sample *transmitBuffer = buffer2;

volatile int samplingIndex = 0;
volatile bool bufferReady = false;

ADXL343 accelerometer_arm(Wire, ADXL343_ADDRESS_ARM);
ADXL343 accelerometer_leg(Wire, ADXL343_ADDRESS_LEG);

bool armOk = false;
bool legOk = false;

void sampleAccelerometer();
void transmitBufferData(int samples);
void buttonHandler(system_event_t event, int data);

Timer samplingTimer(SAMPLING_INTERVAL_MS, sampleAccelerometer);

enum State
{
  STATE_WAITING,
  STATE_CONNECT,
  STATE_RUNNING,
  STATE_FINISH
};

State state = STATE_WAITING;

unsigned long recordingStart = 0;

#define LED_PIN D7

void setup()
{
  System.on(button_click, buttonHandler);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  delay(2000);

  armOk = accelerometer_arm.begin();
  legOk = accelerometer_leg.begin();

  Log.info("ARM sensor: %s", armOk ? "OK" : "FAIL");
  Log.info("LEG sensor: %s", legOk ? "OK" : "FAIL");

  Log.info("Waiting for button press...");
}

void loop()
{
  switch (state)
  {
  case STATE_WAITING:
    break;

  case STATE_CONNECT:
    if (client.connect(serverAddr, serverPort))
    {
      Log.info("Connected to server. Starting data collection...");
      Log.info("Sample rate: %d Hz", 1000 / SAMPLING_INTERVAL_MS);

      recordingStart = millis();
      samplingIndex = 0;
      bufferReady = false;

      samplingTimer.start();
      digitalWrite(LED_PIN, HIGH);

      state = STATE_RUNNING;
    }
    else
    {
      Log.error("Failed to connect to server.");
      state = STATE_WAITING;
    }
    break;

  case STATE_RUNNING:
    if (bufferReady)
    {
      samplingTimer.stop();

      Sample *temp = samplingBuffer;
      samplingBuffer = transmitBuffer;
      transmitBuffer = temp;

      int samplesToTransmit = samplingIndex;
      samplingIndex = 0;
      bufferReady = false;

      samplingTimer.start();

      transmitBufferData(samplesToTransmit);
    }

    if (millis() - recordingStart >= MAX_RECORDING_LENGTH_MS)
    {
      state = STATE_FINISH;
    }
    break;

  case STATE_FINISH:
    samplingTimer.stop();

    if (samplingIndex > 0)
    {
      Sample *temp = samplingBuffer;
      samplingBuffer = transmitBuffer;
      transmitBuffer = temp;

      int samplesToTransmit = samplingIndex;
      samplingIndex = 0;

      transmitBufferData(samplesToTransmit);
    }

    client.stop();
    digitalWrite(LED_PIN, LOW);

    Log.info("Data collection complete.");
    state = STATE_WAITING;
    break;
  }
}

void sampleAccelerometer()
{
  if (samplingIndex >= BUFFER_SIZE)
  {
    bufferReady = true;
    return;
  }

  float xa = 0, ya = 0, za = 0;
  float xl = 0, yl = 0, zl = 0;

  if (armOk)
  {
    accelerometer_arm.readAccelerationG(&xa, &ya, &za);
  }

  if (legOk)
  {
    accelerometer_leg.readAccelerationG(&xl, &yl, &zl);
  }

  samplingBuffer[samplingIndex].timestamp = micros();

  samplingBuffer[samplingIndex].xa = xa;
  samplingBuffer[samplingIndex].ya = ya;
  samplingBuffer[samplingIndex].za = za;

  samplingBuffer[samplingIndex].xl = xl;
  samplingBuffer[samplingIndex].yl = yl;
  samplingBuffer[samplingIndex].zl = zl;

  samplingIndex++;

  if (samplingIndex >= TRANSMIT_THRESHOLD)
  {
    bufferReady = true;
  }
}

void transmitBufferData(int samples)
{
  if (samples == 0)
  {
    Log.warn("Transmit called with empty buffer. Skipping transmission.");
    return;
  }

  Log.info("Transmitting %d samples...", samples);

  for (int i = 0; i < samples; i++)
  {
    String data = String::format(
        "%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
        transmitBuffer[i].timestamp,
        transmitBuffer[i].xa,
        transmitBuffer[i].ya,
        transmitBuffer[i].za,
        transmitBuffer[i].xl,
        transmitBuffer[i].yl,
        transmitBuffer[i].zl);

    client.write((const uint8_t *)data.c_str(), data.length());
  }
}

void buttonHandler(system_event_t event, int data)
{
  switch (state)
  {
  case STATE_WAITING:
    if (WiFi.ready())
    {
      state = STATE_CONNECT;
    }
    else
    {
      Log.warn("Wi-Fi not ready.");
    }
    break;

  case STATE_RUNNING:
    state = STATE_FINISH;
    break;

  default:
    break;
  }
}