#include <Wire.h>
#include <hd44780.h>
#include <hd44780ioClass/hd44780_I2Cexp.h>

hd44780_I2Cexp lcd;

int whiteLED  = 10;
int yellowLED = 11;

enum BrainState { NON_ICTAL, PRE_ICTAL, ICTAL };
BrainState brainState = NON_ICTAL;

void setup() {
  lcd.begin(16, 2);
  lcd.backlight();
  lcd.clear();

  pinMode(whiteLED, OUTPUT);
  pinMode(yellowLED, OUTPUT);

  Serial.begin(9600);
  Serial.println("Ready to receive brain state + confidence.");  // ✅ Fixed typo
  setBrainState(NON_ICTAL, 0.0);
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    Serial.print("Received: ");
    Serial.println(input);

    // ✅ Input validation
    if (input.length() > 2 && input.indexOf(':') > 0) {
      char stateChar = input.charAt(0);
      float confidence = input.substring(2).toFloat();

      if (stateChar == 'n' || stateChar == 'N') {
        setBrainState(NON_ICTAL, confidence);
      } 
      else if (stateChar == 'p' || stateChar == 'P') {
        setBrainState(PRE_ICTAL, confidence);
      }
      else if (stateChar == 'i' || stateChar == 'I') {
        setBrainState(ICTAL, confidence);
      }
    }
  }
}

void setBrainState(BrainState state, float confidence) {
  brainState = state;
  lcd.clear();

  // Confidence bar
  int barLength = (int)(confidence * 16);
  lcd.setCursor(0, 1);
  for (int i = 0; i < barLength; i++) {
    lcd.print("|");
  }

  lcd.setCursor(0, 0);
  switch (state) {
    case NON_ICTAL:
      analogWrite(whiteLED, 255);
      analogWrite(yellowLED, 0);
      lcd.print("State: Non-Ictal");
      Serial.println("Switched to NON-ICTAL");
      break;

    case PRE_ICTAL:
      analogWrite(whiteLED, 50);
      analogWrite(yellowLED, 255);
      lcd.print("State: Pre-Ictal");
      Serial.println("Switched to PRE-ICTAL");
      break;

    case ICTAL:
      analogWrite(whiteLED, 0);
      analogWrite(yellowLED, 50);
      lcd.print("State: Ictal");
      Serial.println("Switched to ICTAL");
      break;
  }
}