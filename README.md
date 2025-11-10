# 🧠 MigraineMinder

<p align="center">
  <img src="./images/migrainelogo.png" alt="MigraineMinder Logo" width="400"/>
</p>

---

## 📖 About  
**MigraineMinder** is an intelligent migraine management assistant that bridges neurotechnology and environmental control. Using real-time EEG signals from the **Muse 2** headband, the system detects early neurological signatures of migraine onset and automatically adjusts ambient lighting or screen brightness via an **Arduino-controlled light module**.  

When a potential migraine episode is detected, MigraineMinder gently dims the environment, logs the event, and prompts the user to record contextual causes — stress, noise, caffeine, or the inevitable group project panic.  

As neurologist **Oliver Sacks** wrote in *Migraine*,  
> “The migraine is not a disease of the brain, but a disorder of energy, of control, of balance.”  
MigraineMinder aims to restore that balance — one photon at a time.  

---

## ⚙️ Software & Hardware Used  

### 🧩 Software Stack  
- **Flask** – for the dashboard and migraine event logging  
- **Python (Muse SDK)** – for real-time EEG signal acquisition and anomaly detection  
- **Arduino (C / C++)** – to control LED brightness and communicate with sensors  
- **Machine Learning Models** – Deep learning for migraine prediction  
- **CSS / JS** – for the interactive migraine diary interface  

### 🔧 Hardware Components  
- **Muse 2 Headband** – EEG, heart rate, motion tracking  
- **Arduino Board** – bridges software with the environment  
- **LDR Sensor** – monitors ambient light  
- **OLED Display** – shows live brain and light status  
- **Vibrating Motor** – gives subtle tactile feedback during pre-migraine warnings  
- **DHT11 Sensor** – records temperature and humidity (potential triggers)  

---

## 💡 Inspiration  
The spark came from one teammate’s grandmother, who endures frequent migraines that make everyday light feel like a flashbang. We wanted to make something that reacts faster than a person can — a little guardian that listens to your brain before the pain hits.  

It also doubles as a savior for university students who can finally say,  
> “Sorry, Professor — my brainwaves literally refused to let me finish the assignment.”  

---

## 🚀 What It Does  
- 🧠 Detects migraine patterns from EEG and physiological data (stress, HRV, focus drop).  
- 💡 Adapts the environment: dims lights automatically or prompts users to reduce screen brightness.  
- 🗒️ Logs triggers: prompts users to tag the cause (noise, fatigue, weather, coursework).  
- 📊 Visualizes data: shows EEG trends, trigger frequency, and environment correlation.  
- 🤖 Personalizes recommendations: over time, MigraineMinder learns which conditions precede migraines.  

---

## 🧩 Challenges We Ran Into  
- Distinguishing genuine migraine signatures from everyday stress signals (turns out, debugging at 3 a.m. looks a lot like a migraine).  
- Synchronizing Muse 2’s Bluetooth data stream with Arduino’s serial communication.  
- Managing false positives — we accidentally dimmed the lab lights every time someone yawned.  
- Balancing data privacy with usability for medical-adjacent data.  

---

## 🏆 Accomplishments  
- Successfully built an **EEG-to-light feedback loop** using Muse 2 + Arduino.  
- Developed a clean **Flask-based dashboard** to record and visualize migraine episodes.  
- Created a system that detects early migraine trends with promising accuracy in pilot testing.  
- Built a user-friendly front-end interface that looks cool even at 20 % brightness.  

---

## 🧠 What We Learned  
- The human brain is the ultimate noisy dataset — and patience is a debugging skill.  
- Integrating biosensors with real-world IoT control is both thrilling and humbling.  
- Empathy can inspire powerful design — when you build for one person’s pain, you often help many.  

---

## 🔮 What’s Next  
- Expanding detection accuracy with **machine learning** trained on diverse EEG data.  
- Integrating with **smart-home ecosystems** (Alexa, Philips Hue) for seamless light control.  
- Building a **mobile companion app** for quick trigger logging and real-time alerts.  
- Collaborating with neurologists to test MigraineMinder as a clinical support tool.  

---

## 🌙 Closing Note  
In the words of **Oliver Sacks**,  
> “Migraine is a kind of electrical storm of the nervous system.”  

**MigraineMinder** doesn't try to stop the storm — it just knows when to turn down the lights.  
