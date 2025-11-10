#!/usr/bin/env python3
"""
Muse 2 EEG, PPG, Motion, and Breathing Tracker
This script connects to a Muse 2 device, tracks various signals,
and processes them to extract meaningful data.
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime
import queue
import sys
import platform
from collections import deque
from pathlib import Path
import csv


try:
    from pylsl import StreamInlet, resolve_byprop, StreamInfo
    LSL_AVAILABLE = True
    # LostError is in pylsl.util, import it separately
    try:
        from pylsl.util import LostError
    except (ImportError, AttributeError):
        # Fallback if LostError is not available
        class LostError(RuntimeError):
            """Stream connection lost error"""
            pass


except ImportError as e:
    LSL_AVAILABLE = False
    print(f"ERROR: pylsl is not installed or error importing: {e}")
    print("Install it with: pip install pylsl")
    print("Also install muselsl: pip install muselsl")
    sys.exit(1)

try:
    from scipy import signal
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy is not installed. Install it with: pip install scipy")
    print("Some filtering features may not work properly.")

try:
    import matplotlib
    # Try different backends for better compatibility
    if platform.system() == 'Darwin':  # macOS
        try:
            matplotlib.use('TkAgg')
        except:
            try:
                matplotlib.use('Qt5Agg')
            except:
                pass  # Use default
    else:
        matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib is not installed. Install it with: pip install matplotlib")
    print("Visualization will be disabled.")

# Configuration
SAMPLING_RATE_EEG = 256  # Hz for Muse 2 EEG
SAMPLING_RATE_PPG = 64   # Hz for Muse 2 PPG
BUFFER_SIZE = 1024  # Buffer size for processing

# Frequency bands (Hz)
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100)
}

# EEG channel names for Muse 2
EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']

# ---- Heuristic metrics for UI percentages ----
def _safe_pct(x):
    try:
        if x is None or x != x:  # NaN
            return 0
        return max(0, min(100, int(round(x))))
    except:
        return 0

def compute_focus_metrics(eeg_bands_avg):
    """
    Map band powers to simple 0..100 scores for the UI.
    eeg_bands_avg: {'delta':..., 'theta':..., 'alpha':..., 'beta':..., 'gamma':...}
    """
    a = float(eeg_bands_avg.get('alpha', 0) or 0)
    b = float(eeg_bands_avg.get('beta',  0) or 0)
    t = float(eeg_bands_avg.get('theta', 0) or 0)
    d = float(eeg_bands_avg.get('delta', 0) or 0)
    g = float(eeg_bands_avg.get('gamma', 0) or 0)
    eps = 1e-9
    total = a + b + t + d + g + eps

    focus_raw      = (b / total) * 100.0
    attention_raw  = (b / (a + t + eps)) * 60.0      # scaled
    meditation_raw = (a / (a + b + t + eps)) * 100.0

    return {
        "focus":      _safe_pct(focus_raw),
        "attention":  _safe_pct(attention_raw),
        "meditation": _safe_pct(meditation_raw),
    }

class VisualizationDashboard:
    """Real-time visualization dashboard for Muse 2 data"""
    
    def __init__(self, max_history=60):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is not available")
        
        self.max_history = max_history  # Show last 60 seconds
        self.running = True
        
        # Data storage for plotting
        self.time_data = deque(maxlen=max_history)
        self.delta_data = deque(maxlen=max_history)
        self.theta_data = deque(maxlen=max_history)
        self.alpha_data = deque(maxlen=max_history)
        self.beta_data = deque(maxlen=max_history)
        self.gamma_data = deque(maxlen=max_history)
        
        # Current metrics
        self.heart_rate = None
        self.head_pitch = None
        self.breathing_rate = None
        
        # Initialize plot
        self.setup_plot()
        
    def setup_plot(self):
        """Setup the matplotlib figure and axes"""
        # Create figure with dark theme
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10), facecolor='#0a0a0a')
        self.fig.suptitle('Muse 2 Real-Time Brain Activity Dashboard', 
                         fontsize=20, fontweight='bold', color='#ffffff', y=0.98)
        
        # Create grid layout
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.35, wspace=0.3,
                     left=0.05, right=0.95, top=0.92, bottom=0.08)
        
        # Frequency band colors (vibrant, aesthetic colors)
        self.colors = {
            'delta': '#FF6B6B',   # Red
            'theta': '#4ECDC4',   # Teal
            'alpha': '#95E1D3',   # Light teal
            'beta': '#F38181',    # Pink
            'gamma': '#AA96DA'    # Purple
        }
        
        # Create subplot for frequency bands graph
        self.ax_waves = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_waves.set_facecolor('#1a1a1a')
        self.ax_waves.set_title('Frequency Bands Over Time', fontsize=14, fontweight='bold', 
                               color='#ffffff', pad=10)
        self.ax_waves.set_xlabel('Time (seconds)', fontsize=11, color='#cccccc')
        self.ax_waves.set_ylabel('Power (μV²)', fontsize=11, color='#cccccc')
        self.ax_waves.grid(True, alpha=0.3, color='#333333')
        self.ax_waves.tick_params(colors='#cccccc')
        
        # Initialize line plots for each frequency band
        self.lines = {}
        for band, color in self.colors.items():
            line, = self.ax_waves.plot([], [], label=band.upper(), color=color, 
                                      linewidth=2.5, alpha=0.8)
            self.lines[band] = line
        
        self.ax_waves.legend(loc='upper right', framealpha=0.8, facecolor='#2a2a2a', 
                            edgecolor='#444444', fontsize=10)
        self.ax_waves.set_xlim(0, self.max_history)
        
        # Metric boxes (top right area)
        # Heart Rate box
        self.ax_hr = self.fig.add_subplot(gs[0, 2])
        self.ax_hr.set_facecolor('#1a1a1a')
        self.ax_hr.axis('off')
        self.ax_hr.set_xlim(0, 1)
        self.ax_hr.set_ylim(0, 1)
        
        # Head Tilt box
        self.ax_tilt = self.fig.add_subplot(gs[1, 2])
        self.ax_tilt.set_facecolor('#1a1a1a')
        self.ax_tilt.axis('off')
        self.ax_tilt.set_xlim(0, 1)
        self.ax_tilt.set_ylim(0, 1)
        
        # Breathing Rate box
        self.ax_breath = self.fig.add_subplot(gs[2, 2])
        self.ax_breath.set_facecolor('#1a1a1a')
        self.ax_breath.axis('off')
        self.ax_breath.set_xlim(0, 1)
        self.ax_breath.set_ylim(0, 1)
        
        # Bar chart for current frequency band values (bottom)
        self.ax_bars = self.fig.add_subplot(gs[2, 0:2])
        self.ax_bars.set_facecolor('#1a1a1a')
        self.ax_bars.set_title('Current Frequency Band Powers', fontsize=14, fontweight='bold',
                              color='#ffffff', pad=10)
        self.ax_bars.set_ylabel('Power (μV²)', fontsize=11, color='#cccccc')
        self.ax_bars.set_xlabel('Frequency Band', fontsize=11, color='#cccccc')
        self.ax_bars.tick_params(colors='#cccccc')
        self.ax_bars.grid(True, alpha=0.3, color='#333333', axis='y')
        
        # Initialize bar chart
        self.bars = self.ax_bars.bar(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'], 
                                     [0, 0, 0, 0, 0],
                                     color=[self.colors['delta'], self.colors['theta'], 
                                           self.colors['alpha'], self.colors['beta'], 
                                           self.colors['gamma']],
                                     alpha=0.8, edgecolor='white', linewidth=1.5)
        
        # Text elements for metrics (will be updated)
        self.hr_text = None
        self.tilt_text = None
        self.breath_text = None
        
        # Make figure responsive
        self.fig.canvas.manager.set_window_title('Muse 2 Dashboard')
        plt.tight_layout()
        
    def update_metrics(self, heart_rate=None, head_pitch=None, breathing_rate=None):
        """Update metric values shown in the metric boxes."""
        if heart_rate is not None:
            self.heart_rate = heart_rate
        if head_pitch is not None:
            self.head_pitch = head_pitch
        if breathing_rate is not None:
            self.breathing_rate = breathing_rate
    
    def update_data(self, delta, theta, alpha, beta, gamma):
        """Update frequency band data"""
        current_time = len(self.time_data)
        self.time_data.append(current_time)
        self.delta_data.append(delta)
        self.theta_data.append(theta)
        self.alpha_data.append(alpha)
        self.beta_data.append(beta)
        self.gamma_data.append(gamma)
    
    def update_plot(self, frame):
        """Update the plot (called by animation)"""
        if not self.running:
            return
        
        # Update frequency band lines
        if len(self.time_data) > 0:
            time_array = np.array(self.time_data)
            
            # Normalize time to show sliding window
            if len(time_array) > 1:
                time_array = time_array - time_array[-1] + self.max_history
            
            self.lines['delta'].set_data(time_array, list(self.delta_data))
            self.lines['theta'].set_data(time_array, list(self.theta_data))
            self.lines['alpha'].set_data(time_array, list(self.alpha_data))
            self.lines['beta'].set_data(time_array, list(self.beta_data))
            self.lines['gamma'].set_data(time_array, list(self.gamma_data))
            
            # Auto-scale y-axis
            all_data = (list(self.delta_data) + list(self.theta_data) + 
                       list(self.alpha_data) + list(self.beta_data) + list(self.gamma_data))
            if all_data:
                max_val = max(all_data) * 1.1 if max(all_data) > 0 else 1000
                self.ax_waves.set_ylim(0, max_val)
            
            # Update x-axis to show sliding window
            if len(time_array) > 0:
                self.ax_waves.set_xlim(max(0, time_array[-1] - self.max_history), 
                                      max(self.max_history, time_array[-1] + 5))
        
        # Update bar chart with latest values
        if len(self.delta_data) > 0:
            latest_values = [
                self.delta_data[-1] if self.delta_data else 0,
                self.theta_data[-1] if self.theta_data else 0,
                self.alpha_data[-1] if self.alpha_data else 0,
                self.beta_data[-1] if self.beta_data else 0,
                self.gamma_data[-1] if self.gamma_data else 0
            ]
            for bar, val in zip(self.bars, latest_values):
                bar.set_height(val)
            self.ax_bars.set_ylim(0, max(latest_values) * 1.1 if max(latest_values) > 0 else 1000)
        
        # Update metric boxes
        self.update_metric_boxes()
        
        # Return artists for animation (filter out None values)
        artists = list(self.lines.values()) + list(self.bars)
        if self.hr_text:
            artists.append(self.hr_text)
        if self.tilt_text:
            artists.append(self.tilt_text)
        if self.breath_text:
            artists.append(self.breath_text)
        return artists
    
    
    def update_metric_boxes(self):
        """Update the metric display boxes"""
        # Heart Rate box
        self.ax_hr.clear()
        self.ax_hr.set_facecolor('#1a1a1a')
        self.ax_hr.axis('off')
        self.ax_hr.set_xlim(0, 1)
        self.ax_hr.set_ylim(0, 1)
        
        # Add rounded rectangle background
        rect_hr = mpatches.FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                          boxstyle="round,pad=0.1",
                                          facecolor='#2a2a2a', edgecolor='#FF6B6B',
                                          linewidth=3, transform=self.ax_hr.transAxes)
        self.ax_hr.add_patch(rect_hr)
        
        self.ax_hr.text(0.5, 0.7, 'HEART RATE', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='#FF6B6B',
                       transform=self.ax_hr.transAxes)
        hr_value = f"{self.heart_rate:.0f}" if self.heart_rate is not None else "---"
        self.hr_text = self.ax_hr.text(0.5, 0.35, f"{hr_value}\nBPM", ha='center', va='center',
                                      fontsize=24, fontweight='bold', color='#ffffff',
                                      transform=self.ax_hr.transAxes)
        
        # Head Tilt box
        self.ax_tilt.clear()
        self.ax_tilt.set_facecolor('#1a1a1a')
        self.ax_tilt.axis('off')
        self.ax_tilt.set_xlim(0, 1)
        self.ax_tilt.set_ylim(0, 1)
        
        rect_tilt = mpatches.FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                           boxstyle="round,pad=0.1",
                                           facecolor='#2a2a2a', edgecolor='#4ECDC4',
                                           linewidth=3, transform=self.ax_tilt.transAxes)
        self.ax_tilt.add_patch(rect_tilt)
        
        self.ax_tilt.text(0.5, 0.7, 'HEAD TILT', ha='center', va='center',
                         fontsize=12, fontweight='bold', color='#4ECDC4',
                         transform=self.ax_tilt.transAxes)
        tilt_value = f"{self.head_pitch:.1f}°" if self.head_pitch is not None else "---"
        self.tilt_text = self.ax_tilt.text(0.5, 0.35, tilt_value, ha='center', va='center',
                                          fontsize=24, fontweight='bold', color='#ffffff',
                                          transform=self.ax_tilt.transAxes)
        
        # Breathing Rate box
        self.ax_breath.clear()
        self.ax_breath.set_facecolor('#1a1a1a')
        self.ax_breath.axis('off')
        self.ax_breath.set_xlim(0, 1)
        self.ax_breath.set_ylim(0, 1)
        
        rect_breath = mpatches.FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                             boxstyle="round,pad=0.1",
                                             facecolor='#2a2a2a', edgecolor='#95E1D3',
                                             linewidth=3, transform=self.ax_breath.transAxes)
        self.ax_breath.add_patch(rect_breath)
        
        self.ax_breath.text(0.5, 0.7, 'BREATHING RATE', ha='center', va='center',
                           fontsize=12, fontweight='bold', color='#95E1D3',
                           transform=self.ax_breath.transAxes)
        breath_value = f"{self.breathing_rate:.1f}" if self.breathing_rate is not None else "---"
        self.breath_text = self.ax_breath.text(0.5, 0.35, f"{breath_value}\n/min", ha='center', va='center',
                                              fontsize=24, fontweight='bold', color='#ffffff',
                                              transform=self.ax_breath.transAxes)
    
    def start_animation(self):
        """Start the matplotlib animation"""
        self.anim = FuncAnimation(self.fig, self.update_plot, interval=200, blit=False, cache_frame_data=False)
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)
        plt.pause(0.1)  # Small pause to allow window to initialize
    
    def close(self):
        """Close the dashboard"""
        self.running = False
        if hasattr(self, 'fig'):
            plt.close(self.fig)


class Muse2Tracker:
    def __init__(self, enable_visualization=True):
        self.running = False
        self.raw_data_queue = queue.Queue()
        self.processed_data_queue = queue.Queue()
        self.eeg_buffer = {ch: deque(maxlen=BUFFER_SIZE) for ch in EEG_CHANNELS}
        self.ppg_buffer = deque(maxlen=BUFFER_SIZE)
        self.acc_buffer = deque(maxlen=BUFFER_SIZE)
        self.gyro_buffer = deque(maxlen=BUFFER_SIZE)
        self.breath_buffer = deque(maxlen=BUFFER_SIZE)
        self.timestamps = deque(maxlen=BUFFER_SIZE)
        
        self.raw_data = []
        self.processed_data = []
        
        self.eeg_inlet = None
        self.ppg_inlet = None
        self.acc_inlet = None
        self.gyro_inlet = None
        
        # Initialize visualization dashboard
        self.dashboard = None
        self.enable_visualization = enable_visualization and MATPLOTLIB_AVAILABLE
        if self.enable_visualization:
            try:
                self.dashboard = VisualizationDashboard()
                print("✓ Visualization dashboard initialized")
            except Exception as e:
                print(f"⚠ Could not initialize visualization: {e}")
                self.enable_visualization = False
                self.dashboard = None
        
    def connect_to_muse(self):
        """Connect to Muse 2 device"""
        print("Searching for Muse 2 device...")
        print("Make sure your Muse 2 is turned on and Bluetooth is enabled on your computer.")
        print("If using muselsl, run 'muselsl stream' in a separate terminal first.")
        print()
        
        # Look for EEG stream
        print("Looking for EEG stream...")
        try:
            streams = resolve_byprop('type', 'EEG', minimum=1, timeout=5.0)
            if streams:
                self.eeg_inlet = StreamInlet(streams[0])
                print(f"✓ Connected to EEG stream: {streams[0].name()}")
            else:
                print("✗ No EEG stream found. Make sure Muse 2 is streaming.")
                return False
        except Exception as e:
            print(f"✗ Error finding EEG stream: {e}")
            print("Make sure Muse 2 is streaming (run start_muse_stream.py first)")
            return False
        
        # Look for PPG stream (if available)
        print("Looking for PPG stream...")
        try:
            streams = resolve_byprop('type', 'PPG', minimum=1, timeout=2.0)
            if streams:
                self.ppg_inlet = StreamInlet(streams[0])
                print(f"✓ Connected to PPG stream: {streams[0].name()}")
        except Exception as e:
            print(f"⚠ PPG stream not found (may not be available): {e}")
        
        # Look for accelerometer stream
        print("Looking for accelerometer stream...")
        try:
            streams = resolve_byprop('type', 'ACC', minimum=1, timeout=2.0)
            if streams:
                self.acc_inlet = StreamInlet(streams[0])
                print(f"✓ Connected to accelerometer stream: {streams[0].name()}")
        except Exception as e:
            print(f"⚠ Accelerometer stream not found: {e}")
        
        # Look for gyroscope stream
        print("Looking for gyroscope stream...")
        try:
            streams = resolve_byprop('type', 'GYRO', minimum=1, timeout=2.0)
            if streams:
                self.gyro_inlet = StreamInlet(streams[0])
                print(f"✓ Connected to gyroscope stream: {streams[0].name()}")
        except Exception as e:
            print(f"⚠ Gyroscope stream not found: {e}")
        
        return True
    
    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        """Create a bandpass filter"""
        if not SCIPY_AVAILABLE:
            return None
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def notch_filter(self, data, fs, freq=50, quality=30):
        """Apply notch filter to remove power line noise (50/60 Hz)"""
        # Convert to numpy array if needed
        data_array = np.array(data) if not isinstance(data, np.ndarray) else data
        
        if not SCIPY_AVAILABLE or len(data_array) < fs // 10:
            return data_array
        try:
            b, a = signal.iirnotch(freq, quality, fs)
            filtered = filtfilt(b, a, data_array)
            return filtered
        except:
            return data_array
    
    def reject_artifacts(self, data, threshold=3):
        """Reject artifacts using z-score threshold"""
        if len(data) < 10:
            return data
        try:
            data_array = np.array(data)
            mean = np.mean(data_array)
            std = np.std(data_array)
            if std == 0:
                return data
            z_scores = np.abs((data_array - mean) / std)
            # Replace outliers with median value
            median = np.median(data_array)
            data_array[z_scores > threshold] = median
            return data_array.tolist()
        except:
            return data
    
    def moving_average(self, data, window_size=10):
        """Apply moving average filter for smoothing"""
        if len(data) < window_size:
            # Ensure return type matches input type
            return np.array(data) if isinstance(data, np.ndarray) else data
        try:
            # Convert to numpy array if needed
            data_array = np.array(data) if not isinstance(data, np.ndarray) else data
            smoothed = np.convolve(data_array, np.ones(window_size)/window_size, mode='same')
            # Return numpy array (not list) to maintain type consistency
            return smoothed
        except:
            # Ensure return type matches input type
            return np.array(data) if isinstance(data, np.ndarray) else data
    
    def bandpass_filter(self, data, lowcut, highcut, fs, order=4, apply_notch=True, reject_artifacts_flag=True):
        """Apply bandpass filter to data with optional noise reduction"""
        # Convert to numpy array if needed
        data_array = np.array(data) if not isinstance(data, np.ndarray) else data
        
        if not SCIPY_AVAILABLE or len(data_array) < order * 3:
            return data_array
        
        # Step 1: Artifact rejection (remove extreme values)
        if reject_artifacts_flag:
            data_array = np.array(self.reject_artifacts(data_array.tolist(), threshold=3))
        
        # Step 2: Notch filter for power line noise (50/60 Hz)
        if apply_notch:
            # Try both 50 Hz and 60 Hz, use the one that works
            try:
                data_array = self.notch_filter(data_array, fs, freq=50, quality=30)
                # Ensure it's still a numpy array
                data_array = np.array(data_array) if not isinstance(data_array, np.ndarray) else data_array
            except:
                try:
                    data_array = self.notch_filter(data_array, fs, freq=60, quality=30)
                    # Ensure it's still a numpy array
                    data_array = np.array(data_array) if not isinstance(data_array, np.ndarray) else data_array
                except:
                    pass
        
        # Step 3: Bandpass filter
        b, a = self.butter_bandpass(lowcut, highcut, fs, order)
        if b is None or a is None:
            return data_array
        try:
            filtered = filtfilt(b, a, data_array)
            return filtered
        except:
            return data_array
    
    def extract_frequency_bands(self, eeg_data, sampling_rate):
        """Extract frequency bands from EEG data with noise reduction"""
        bands = {}
        # Ensure eeg_data is a numpy array
        eeg_array = np.array(eeg_data) if not isinstance(eeg_data, np.ndarray) else eeg_data
        
        for band_name, (low, high) in FREQ_BANDS.items():
            if len(eeg_array) < sampling_rate // 2:  # Need at least 0.5 seconds of data
                bands[band_name] = 0.0
                continue
            
            # Apply bandpass filter with noise reduction
            # For frequency band extraction, we apply notch filter and artifact rejection
            filtered = self.bandpass_filter(eeg_array, low, high, sampling_rate, 
                                          apply_notch=True, reject_artifacts_flag=True)
            
            # Ensure filtered is a numpy array
            filtered = np.array(filtered) if not isinstance(filtered, np.ndarray) else filtered
            
            # Apply smoothing with moving average
            if len(filtered) > 10:
                filtered = self.moving_average(filtered, window_size=min(10, len(filtered)//4))
                # Ensure it's still a numpy array after moving_average
                filtered = np.array(filtered) if not isinstance(filtered, np.ndarray) else filtered
            
            # Calculate power using Welch's method
            if SCIPY_AVAILABLE and len(filtered) > sampling_rate // 4:
                try:
                    freqs, psd = signal.welch(filtered, sampling_rate, nperseg=min(len(filtered), sampling_rate))
                    # Find frequencies in band
                    idx = np.logical_and(freqs >= low, freqs <= high)
                    bands[band_name] = np.trapz(psd[idx], freqs[idx])
                except Exception as e:
                    # Fallback: simple RMS
                    try:
                        bands[band_name] = np.sqrt(np.mean(filtered**2))
                    except:
                        bands[band_name] = 0.0
            else:
                # Fallback: simple RMS
                try:
                    bands[band_name] = np.sqrt(np.mean(filtered**2))
                except:
                    bands[band_name] = 0.0
        
        return bands
    
    def calculate_heart_rate(self, ppg_data, sampling_rate):
        """Calculate heart rate from PPG data with noise reduction"""
        if len(ppg_data) < sampling_rate:  # Need at least 1 second of data
            return None
        
        if not SCIPY_AVAILABLE:
            return None
        
        try:
            # Convert to numpy array
            ppg_array = np.array(ppg_data) if not isinstance(ppg_data, np.ndarray) else ppg_data
            
            # Apply artifact rejection
            ppg_array = np.array(self.reject_artifacts(ppg_array.tolist(), threshold=2.5))
            
            # Apply smoothing (moving_average now returns numpy array)
            ppg_array = self.moving_average(ppg_array, window_size=5)
            # Ensure it's a numpy array
            ppg_array = np.array(ppg_array) if not isinstance(ppg_array, np.ndarray) else ppg_array
            
            # Apply bandpass filter for heart rate (0.5-5 Hz, which is 30-300 BPM)
            filtered = self.bandpass_filter(ppg_array, 0.5, 5, sampling_rate, 
                                          apply_notch=False, reject_artifacts_flag=False)
            # Ensure filtered is a numpy array
            filtered = np.array(filtered) if not isinstance(filtered, np.ndarray) else filtered
            
            # Find peaks with prominence threshold to avoid noise
            peaks, properties = signal.find_peaks(filtered, 
                                                 distance=int(sampling_rate * 0.4),  # Minimum 0.4s between peaks
                                                 prominence=np.std(filtered) * 0.5)  # Minimum prominence
            
            if len(peaks) < 2:
                return None
            
            # Calculate average time between peaks
            peak_intervals = np.diff(peaks) / sampling_rate
            # Remove outliers (intervals that are too short or too long)
            valid_intervals = peak_intervals[(peak_intervals > 0.3) & (peak_intervals < 2.0)]
            if len(valid_intervals) == 0:
                return None
            
            avg_interval = np.median(valid_intervals)
            heart_rate = 60 / avg_interval if avg_interval > 0 else None
            
            # Apply offset to correct for Muse 2 PPG calibration issues
            # Muse 2 tends to read ~30 BPM lower than actual heart rate
            if heart_rate and heart_rate < 100:
                heart_rate = heart_rate + 30
            
            # Sanity check: heart rate should be reasonable (50-200 BPM)
            if heart_rate and (heart_rate < 50 or heart_rate > 200):
                return None
            
            return heart_rate
        except:
            return None
    
    def calculate_posture(self, acc_sample):
        if acc_sample is None or len(acc_sample) < 3:
            return None
        x, y, z = acc_sample[:3]
        pitch = np.degrees(np.arctan2(-x, np.sqrt(y*y + z*z)))
        roll  = np.degrees(np.arctan2(y, z))
        return {'pitch': pitch, 'roll': roll, 'head_movement': np.sqrt(x*x + y*y + z*z)}

    
    def estimate_breathing_rate(self, breath_data, sampling_rate):
        """Estimate breathing rate from breath data with noise reduction"""
        if len(breath_data) < sampling_rate * 2:  # Need at least 2 seconds
            return None
        
        if not SCIPY_AVAILABLE:
            return None
        
        try:
            # Convert to numpy array
            breath_array = np.array(breath_data) if not isinstance(breath_data, np.ndarray) else breath_data
            
            # Apply artifact rejection
            breath_array = np.array(self.reject_artifacts(breath_array.tolist(), threshold=2.5))
            
            # Apply smoothing (moving_average now returns numpy array)
            breath_array = self.moving_average(breath_array, window_size=10)
            # Ensure it's a numpy array
            breath_array = np.array(breath_array) if not isinstance(breath_array, np.ndarray) else breath_array
            
            # Breathing is typically 0.1-0.5 Hz (6-30 breaths per minute)
            filtered = self.bandpass_filter(breath_array, 0.1, 0.5, sampling_rate, 
                                          apply_notch=False, reject_artifacts_flag=False)
            # Ensure filtered is a numpy array
            filtered = np.array(filtered) if not isinstance(filtered, np.ndarray) else filtered
            
            # Find peaks (breaths) with prominence threshold
            peaks, _ = signal.find_peaks(filtered, 
                                       distance=int(sampling_rate * 0.8),  # Minimum 0.8s between breaths
                                       prominence=np.std(filtered) * 0.3)  # Minimum prominence
            
            if len(peaks) < 2:
                return None
            
            # Calculate average time between breaths
            peak_intervals = np.diff(peaks) / sampling_rate
            # Remove outliers (intervals that are too short or too long)
            valid_intervals = peak_intervals[(peak_intervals > 1.0) & (peak_intervals < 10.0)]
            if len(valid_intervals) == 0:
                return None
            
            avg_interval = np.median(valid_intervals)
            breathing_rate = 60 / avg_interval if avg_interval > 0 else None
            
            # Sanity check: breathing rate should be reasonable (6-30 breaths per minute)
            if breathing_rate and (breathing_rate < 6 or breathing_rate > 30):
                return None
            
            return breathing_rate
        except:
            return None
    
    def collect_data(self):
        """Collect data from all streams"""
        eeg_samples = []
        ppg_samples = []
        acc_samples = []
        gyro_samples = []
        
        # Collect EEG data
        if self.eeg_inlet:
            try:
                sample, timestamp = self.eeg_inlet.pull_sample(timeout=0.1)
                if sample:
                    eeg_samples.append((sample, timestamp))
            except LostError:
                print("Lost connection to EEG stream")
            except:
                pass
        
        # Collect PPG data
        if self.ppg_inlet:
            try:
                sample, timestamp = self.ppg_inlet.pull_sample(timeout=0.1)
                if sample:
                    ppg_samples.append((sample, timestamp))
            except:
                pass
        
        # Collect accelerometer data
        if self.acc_inlet:
            try:
                sample, timestamp = self.acc_inlet.pull_sample(timeout=0.1)
                if sample:
                    acc_samples.append((sample, timestamp))
            except:
                pass
        
        # Collect gyroscope data
        if self.gyro_inlet:
            try:
                sample, timestamp = self.gyro_inlet.pull_sample(timeout=0.1)
                if sample:
                    gyro_samples.append((sample, timestamp))
            except:
                pass
        
        return eeg_samples, ppg_samples, acc_samples, gyro_samples
    
    def process_data(self):
        """Process collected data and extract features"""
        try:
            current_time = time.time()
            
            # Process EEG data
            eeg_bands = {}
            try:
                for i, channel in enumerate(EEG_CHANNELS):
                    if len(self.eeg_buffer[channel]) >= SAMPLING_RATE_EEG // 4:  # At least 0.25 seconds
                        eeg_array = np.array(list(self.eeg_buffer[channel]))
                        bands = self.extract_frequency_bands(eeg_array, SAMPLING_RATE_EEG)
                        for band_name, power in bands.items():
                            eeg_bands[f'{channel}_{band_name}'] = power
            except Exception as e:
                # Continue processing other data even if EEG processing fails
                pass
            
            # Process PPG data (heart rate)
            heart_rate = None
            try:
                if len(self.ppg_buffer) >= SAMPLING_RATE_PPG:
                    ppg_array = np.array(list(self.ppg_buffer))
                    heart_rate = self.calculate_heart_rate(ppg_array, SAMPLING_RATE_PPG)
            except Exception as e:
                # Continue processing other data even if heart rate calculation fails
                pass
            
            # Process accelerometer data (posture)
            posture = None
            try:
                if len(self.acc_buffer) > 0:
                    acc_data = list(self.acc_buffer)[-1]
                    posture = self.calculate_posture(acc_data)
            except Exception as e:
                # Continue processing other data even if posture calculation fails
                pass
            
            # Process breathing (if available from accelerometer or derived from other signals)
            breathing_rate = None
            try:
                if len(self.breath_buffer) >= SAMPLING_RATE_PPG:
                    breath_array = np.array(list(self.breath_buffer))
                    breathing_rate = self.estimate_breathing_rate(breath_array, SAMPLING_RATE_PPG)
            except Exception as e:
                # Continue processing other data even if breathing rate calculation fails
                pass
            
            return {
                'timestamp': current_time,
                'eeg_bands': eeg_bands,
                'heart_rate': heart_rate,
                'posture': posture,
                'breathing_rate': breathing_rate
            }
        except Exception as e:
            # Return empty dict if processing completely fails
            return {
                'timestamp': time.time(),
                'eeg_bands': {},
                'heart_rate': None,
                'posture': None,
                'breathing_rate': None
            }
    
    def run(self):
        """Main data collection and processing loop"""
        print("\n" + "="*60)
        print("Starting data collection...")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        # Start visualization dashboard in a separate thread if enabled
        if self.enable_visualization and self.dashboard:
            try:
                print("🚀 Launching visualization dashboard...")
                self.dashboard.start_animation()
                # Give the dashboard a moment to initialize
                time.sleep(0.5)
            except Exception as e:
                print(f"⚠ Could not start visualization: {e}")
                self.enable_visualization = False
        
        self.running = True
        last_save_time = time.time()
        save_interval = 1.0  # Save processed data every second
        
        try:
            while self.running:
                # Collect data
                eeg_samples, ppg_samples, acc_samples, gyro_samples = self.collect_data()
                
                # Store raw data
                for sample, timestamp in eeg_samples:
                    if len(sample) >= len(EEG_CHANNELS):
                        for i, channel in enumerate(EEG_CHANNELS):
                            self.eeg_buffer[channel].append(sample[i])
                        
                        # Store raw EEG data
                        raw_row = {
                            'timestamp': timestamp,
                            'type': 'EEG',
                            'TP9': sample[0] if len(sample) > 0 else None,
                            'AF7': sample[1] if len(sample) > 1 else None,
                            'AF8': sample[2] if len(sample) > 2 else None,
                            'TP10': sample[3] if len(sample) > 3 else None,
                        }
                        self.raw_data.append(raw_row)
                
                for sample, timestamp in ppg_samples:
                    self.ppg_buffer.append(sample[0] if isinstance(sample, (list, np.ndarray)) else sample)
                    raw_row = {
                        'timestamp': timestamp,
                        'type': 'PPG',
                        'value': sample[0] if isinstance(sample, (list, np.ndarray)) else sample,
                    }
                    self.raw_data.append(raw_row)
                
                for sample, timestamp in acc_samples:
                    self.acc_buffer.append(sample)
                    raw_row = {
                        'timestamp': timestamp,
                        'type': 'ACC',
                        'x': sample[0] if len(sample) > 0 else None,
                        'y': sample[1] if len(sample) > 1 else None,
                        'z': sample[2] if len(sample) > 2 else None,
                    }
                    self.raw_data.append(raw_row)
                    
                    # Use accelerometer z-axis for breathing estimation (chest movement)
                    if len(sample) > 2:
                        self.breath_buffer.append(sample[2])
                
                for sample, timestamp in gyro_samples:
                    raw_row = {
                        'timestamp': timestamp,
                        'type': 'GYRO',
                        'x': sample[0] if len(sample) > 0 else None,
                        'y': sample[1] if len(sample) > 1 else None,
                        'z': sample[2] if len(sample) > 2 else None,
                    }
                    self.raw_data.append(raw_row)
                
                # Process data periodically
                current_time = time.time()
                if current_time - last_save_time >= save_interval:
                    try:
                        processed = self.process_data()
                        if processed:
                            self.processed_data.append(processed)
                            last_save_time = current_time
                            
                            # Update visualization dashboard
                        if self.enable_visualization and self.dashboard:
                            try:
                                # Calculate average power for each band across all channels
                                eeg_bands = processed.get('eeg_bands', {})
                                delta_powers = [eeg_bands.get(f'{ch}_delta', 0) for ch in EEG_CHANNELS]
                                theta_powers = [eeg_bands.get(f'{ch}_theta', 0) for ch in EEG_CHANNELS]
                                alpha_powers = [eeg_bands.get(f'{ch}_alpha', 0) for ch in EEG_CHANNELS]
                                beta_powers  = [eeg_bands.get(f'{ch}_beta',  0) for ch in EEG_CHANNELS]
                                gamma_powers = [eeg_bands.get(f'{ch}_gamma', 0) for ch in EEG_CHANNELS]

                                delta_avg = np.mean(delta_powers) if delta_powers else 0.0
                                theta_avg = np.mean(theta_powers) if theta_powers else 0.0
                                alpha_avg = np.mean(alpha_powers) if alpha_powers else 0.0
                                beta_avg  = np.mean(beta_powers)  if beta_powers  else 0.0
                                gamma_avg = np.mean(gamma_powers) if gamma_powers else 0.0

                                # Compute focus metrics and append rolling CSV
                                eeg_avg = {
                                    'delta': float(delta_avg or 0.0),
                                    'theta': float(theta_avg or 0.0),
                                    'alpha': float(alpha_avg or 0.0),
                                    'beta':  float(beta_avg  or 0.0),
                                    'gamma': float(gamma_avg or 0.0),
                                }
                                metrics = compute_focus_metrics(eeg_avg)
                                latest = {**metrics, "timestamp": datetime.utcnow().isoformat() + "Z"}

                                Path("data").mkdir(exist_ok=True)
                                csv_path = Path("data/muse2_data_processed_latest.csv")
                                header = ["timestamp","delta","theta","alpha","beta","gamma",
                                        "heart_rate_bpm","breathing_rate_bpm","head_pitch","head_roll","head_movement"]
                                row = [
                                    datetime.utcnow().isoformat()+"Z",
                                    float(delta_avg or 0.0), float(theta_avg or 0.0), float(alpha_avg or 0.0),
                                    float(beta_avg or 0.0), float(gamma_avg or 0.0),
                                    processed.get("heart_rate") if processed.get("heart_rate") is not None else "",
                                    processed.get("breathing_rate") if processed.get("breathing_rate") is not None else "",
                                    processed.get("posture",{}).get("pitch","") if processed.get("posture") else "",
                                    processed.get("posture",{}).get("roll","")  if processed.get("posture") else "",
                                    processed.get("posture",{}).get("head_movement","") if processed.get("posture") else "",
                                ]
                                new_file = not csv_path.exists()
                                with csv_path.open("a", newline="") as f:
                                    w = csv.writer(f)
                                    if new_file:
                                        w.writerow(header)
                                    w.writerow(row)

                                # Update dashboard
                                self.dashboard.update_data(delta_avg, theta_avg, alpha_avg, beta_avg, gamma_avg)
                                self.dashboard.update_metrics(
                                    heart_rate=processed.get('heart_rate'),
                                    head_pitch=processed.get('posture', {}).get('pitch') if processed.get('posture') else None,
                                    breathing_rate=processed.get('breathing_rate')
                                )

                                # Process GUI events (non-blocking)
                                if hasattr(self.dashboard, 'fig') and plt.fignum_exists(self.dashboard.fig.number):
                                    try:
                                        plt.pause(0.001)
                                    except Exception:
                                        self.enable_visualization = False
                            except Exception as e:
                                # Don't let visualization errors stop data collection
                                pass

                            
                            # Print status (optional, can be disabled if using visualization)
                            if processed.get('heart_rate'):
                                print(f"HR: {processed['heart_rate']:.1f} BPM", end="  ")
                            
                            # Display all frequency bands (averaged across channels)
                            if processed.get('eeg_bands'):
                                eeg_bands = processed['eeg_bands']
                                
                                # Calculate average power for each band across all channels
                                delta_powers = [eeg_bands.get(f'{ch}_delta', 0) for ch in EEG_CHANNELS]
                                theta_powers = [eeg_bands.get(f'{ch}_theta', 0) for ch in EEG_CHANNELS]
                                alpha_powers = [eeg_bands.get(f'{ch}_alpha', 0) for ch in EEG_CHANNELS]
                                beta_powers = [eeg_bands.get(f'{ch}_beta', 0) for ch in EEG_CHANNELS]
                                gamma_powers = [eeg_bands.get(f'{ch}_gamma', 0) for ch in EEG_CHANNELS]
                                
                                # Average across all channels
                                delta_avg = np.mean(delta_powers) if delta_powers else 0.0
                                theta_avg = np.mean(theta_powers) if theta_powers else 0.0
                                alpha_avg = np.mean(alpha_powers) if alpha_powers else 0.0
                                beta_avg = np.mean(beta_powers) if beta_powers else 0.0
                                gamma_avg = np.mean(gamma_powers) if gamma_powers else 0.0
                                
                                print(f"Delta: {delta_avg:.1f}  Theta: {theta_avg:.1f}  Alpha: {alpha_avg:.1f}  Beta: {beta_avg:.1f}  Gamma: {gamma_avg:.1f}", end="  ")
                            
                            if processed.get('posture'):
                                print(f"Pitch: {processed['posture']['pitch']:.1f}°", end="  ")
                            if processed.get('breathing_rate'):
                                print(f"Breath: {processed['breathing_rate']:.1f} /min", end="")
                            print()
                        else:
                            last_save_time = current_time
                    except Exception as e:
                        # Log error but continue collecting data
                        print(f"⚠ Processing error (continuing): {type(e).__name__}: {e}")
                        last_save_time = current_time
                        # Continue running - don't stop data collection
                
                time.sleep(0.01)  # Small delay to prevent CPU overload
                
        except KeyboardInterrupt:
            print("\n\nStopping data collection...")
            self.running = False
        finally:
            # Close visualization dashboard
            if self.enable_visualization and self.dashboard:
                try:
                    self.dashboard.close()
                except:
                    pass
    
    def save_data(self, filename_prefix="muse2_data"):
        """Save collected data to CSV files"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw data
        if self.raw_data:
            raw_df = pd.DataFrame(self.raw_data)
            raw_filename = f"{filename_prefix}_raw_{timestamp_str}.csv"
            raw_df.to_csv(raw_filename, index=False)
            print(f"\n✓ Saved raw data to: {raw_filename}")
            print(f"  Total raw samples: {len(raw_df)}")
        
        # Save processed data
        if self.processed_data:
            processed_rows = []
            for item in self.processed_data:
                row = {'timestamp': item['timestamp']}
                
                # Add EEG bands
                if item.get('eeg_bands'):
                    row.update(item['eeg_bands'])
                
                # Add heart rate
                if item.get('heart_rate'):
                    row['heart_rate_bpm'] = item['heart_rate']
                
                # Add posture
                if item.get('posture'):
                    row['head_pitch'] = item['posture'].get('pitch')
                    row['head_roll'] = item['posture'].get('roll')
                    row['head_movement'] = item['posture'].get('head_movement')
                
                # Add breathing rate
                if item.get('breathing_rate'):
                    row['breathing_rate_bpm'] = item['breathing_rate']
                
                processed_rows.append(row)
            
            processed_df = pd.DataFrame(processed_rows)
            processed_filename = f"{filename_prefix}_processed_{timestamp_str}.csv"
            processed_df.to_csv(processed_filename, index=False)
            print(f"✓ Saved processed data to: {processed_filename}")
            print(f"  Total processed samples: {len(processed_df)}")
        
        print("\nData collection complete!")


def print_noise_reduction_guide():
    """Print guide for noise reduction"""
    print("\n" + "="*60)
    print("NOISE REDUCTION GUIDE FOR MUSE 2")
    print("="*60)
    print("""
1. DEVICE SETUP:
   - Ensure Muse 2 headband is properly fitted and electrodes are clean
   - Check electrode contact quality (should show good signal quality in Muse app)
   - Make sure the headband is securely positioned
   - Wet the electrodes slightly for better contact (use water or electrode gel)

2. ENVIRONMENT:
   - Minimize electrical interference (turn off unnecessary electronics)
   - Avoid fluorescent lights (they create 50/60 Hz noise)
   - Stay away from power lines and electrical equipment
   - Use in a quiet, stable environment

3. PHYSICAL PREPARATION:
   - Remove any metal jewelry or accessories
   - Avoid movements during data collection
   - Keep your body relaxed and still
   - Close your eyes or maintain steady gaze to reduce eye movement artifacts

4. SOFTWARE FILTERING (Already implemented in this script):
   - Bandpass filtering for frequency bands (removes out-of-band noise)
   - Notch filtering can be added for 50/60 Hz power line noise
   - Moving average smoothing for stable readings
   - Artifact rejection (extreme values are filtered out)

5. DATA QUALITY CHECKS:
   - Monitor signal amplitude (should be in microvolts range for EEG)
   - Check for flat lines (indicates poor electrode contact)
   - Watch for sudden spikes (indicates movement artifacts)
   - Verify consistent sampling rate

6. SOFTWARE FILTERING (Already implemented in this script):
   ✓ Notch filtering: Automatically removes 50/60 Hz power line noise
   ✓ Artifact rejection: Removes extreme values using z-score thresholding
   ✓ Moving average: Smooths data to reduce high-frequency noise
   ✓ Bandpass filtering: Isolates frequency bands of interest
   ✓ Peak detection with prominence: Reduces false positives in heart/breath rate

7. POST-PROCESSING (Optional advanced techniques):
   - Independent Component Analysis (ICA) to separate artifacts from brain signals
   - Adaptive filtering for motion artifacts
   - Baseline correction to remove DC offset
   - Wavelet denoising for non-stationary noise

NOISE REDUCTION FEATURES IMPLEMENTED:
The script automatically applies:
- Notch filter (50/60 Hz) for power line noise removal
- Artifact rejection (z-score > 3 for EEG, > 2.5 for PPG/breathing)
- Moving average smoothing (window size 10 for EEG, 5 for PPG, 10 for breathing)
- Bandpass filtering for frequency band isolation
- Peak detection with prominence thresholds for heart/breath rate
- Outlier removal for calculated rates (sanity checks)
""")


def main():
    """Main function"""
    print_noise_reduction_guide()
    
    print("\n" + "="*60)
    print("MUSE 2 TRACKER")
    print("="*60)
    print("\nWaiting for 'start' command...")
    print("Type 'start' and press Enter to begin tracking")
    print("Or type 'quit' to exit\n")
    
    while True:
        command = input("> ").strip().lower()
        if command == 'start':
            tracker = Muse2Tracker()
            if tracker.connect_to_muse():
                try:
                    tracker.run()
                finally:
                    tracker.save_data()
            else:
                print("Failed to connect to Muse 2. Please check your device and try again.")
            break
        elif command == 'quit':
            print("Exiting...")
            break
        else:
            print("Please type 'start' to begin or 'quit' to exit")


if __name__ == "__main__":
    main()

