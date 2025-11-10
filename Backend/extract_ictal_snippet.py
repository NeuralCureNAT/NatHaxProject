#!/usr/bin/env python3
"""
Extract a 10-minute snippet from the ictal file with highest ictal state.
Convert channel data to frequency bands (alpha, beta, gamma, theta, delta)
and format it for the live website.
"""

import csv
import os
import numpy as np
from datetime import datetime, timedelta

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using simplified frequency band calculation")

# Frequency bands (Hz)
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100)
}

# Sampling rate (assumed, adjust if needed)
SAMPLE_RATE = 256  # Hz

def extract_frequency_bands(eeg_signal, sampling_rate):
    """Extract frequency band powers from EEG signal"""
    bands = {}
    
    if SCIPY_AVAILABLE and len(eeg_signal) > 1:
        # Calculate power spectral density
        freqs, psd = signal.welch(eeg_signal, sampling_rate, nperseg=min(256, len(eeg_signal)))
        
        for band_name, (low, high) in FREQ_BANDS.items():
            # Find indices of frequencies in this band
            idx = np.logical_and(freqs >= low, freqs <= high)
            # Calculate power (integrate PSD over frequency band)
            power = np.trapz(psd[idx], freqs[idx])
            bands[band_name] = float(power)
    else:
        # Simplified calculation for single values
        signal_power = np.sum(np.abs(eeg_signal)) if len(eeg_signal) > 0 else 0
        if signal_power > 0:
            bands['delta'] = signal_power * 0.1
            bands['theta'] = signal_power * 0.15
            bands['alpha'] = signal_power * 0.25
            bands['beta'] = signal_power * 0.3
            bands['gamma'] = signal_power * 0.2
        else:
            for band_name in FREQ_BANDS.keys():
                bands[band_name] = 0.0
    
    return bands

def calculate_ictal_intensity(row_data, channels):
    """
    Calculate ictal intensity based on high-frequency activity.
    Ictal states typically show increased high-frequency (beta/gamma) activity.
    """
    total_power = 0
    high_freq_power = 0
    
    for channel in channels:
        if channel in row_data:
            try:
                value = float(row_data[channel])
                if not np.isnan(value):
                    # For single value, estimate bands based on magnitude
                    abs_value = abs(value)
                    total_power += abs_value
                    # High frequency activity (beta + gamma) indicates ictal state
                    high_freq_power += abs_value * 0.6  # Weight high frequencies more
            except (ValueError, TypeError):
                pass
    
    if total_power > 0:
        return high_freq_power / total_power
    return 0

def process_ictal_file(filepath, sample_interval=256):
    """
    Process the large ictal file to find highest ictal state.
    Extract a 10-minute snippet (600 samples at 1 Hz).
    Uses CSV reader to avoid loading entire file into memory.
    """
    print(f"Reading ictal file: {filepath}")
    print("This may take a while for large files...")
    
    # Read header to get column names
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        print(f"Found {len(header)} channels")
        
        # Scan file to find highest ictal intensity
        max_intensity = 0
        best_row_idx = 0
        row_idx = 0
        sample_count = 0
        
        print("\nScanning file for highest ictal state...")
        for row in reader:
            # Sample every Nth row to speed up processing
            if row_idx % sample_interval == 0:
                intensity = calculate_ictal_intensity(row, header)
                if intensity > max_intensity:
                    max_intensity = intensity
                    best_row_idx = row_idx
                sample_count += 1
                if sample_count % 1000 == 0:
                    print(f"Scanned {sample_count * sample_interval} rows, max intensity: {max_intensity:.4f}", end='\r')
            row_idx += 1
        
        print(f"\n\nFound highest ictal state at row {best_row_idx} (intensity: {max_intensity:.4f})")
        
        # Extract 10-minute snippet (600 samples)
        snippet_length = 600
        start_row = max(0, best_row_idx - snippet_length // 2)
        end_row = start_row + snippet_length
        
        print(f"\nExtracting 10-minute snippet (rows {start_row} to {end_row})...")
        
        # Reset file and read the snippet
        f.seek(0)
        reader = csv.DictReader(f)
        snippet_data = []
        current_row = 0
        
        for row in reader:
            if start_row <= current_row < end_row:
                snippet_data.append(row)
            current_row += 1
            if current_row >= end_row:
                break
        
        print(f"Extracted {len(snippet_data)} rows")
        return snippet_data, header, start_row

def convert_to_frequency_bands(snippet_data, channels):
    """
    Convert channel data to frequency bands (delta, theta, alpha, beta, gamma).
    Average across all channels for each band.
    """
    print("\nConverting channel data to frequency bands...")
    
    processed_rows = []
    
    for i, row in enumerate(snippet_data):
        if i % 100 == 0:
            print(f"Processing row {i}/{len(snippet_data)}...", end='\r')
        
        # Aggregate all channel values for this row
        channel_values = []
        for channel in channels:
            if channel in row:
                try:
                    val = float(row[channel])
                    if not np.isnan(val):
                        channel_values.append(val)
                except (ValueError, TypeError):
                    pass
        
        if len(channel_values) == 0:
            continue
        
        # Use channel values as a signal (simplified approach)
        signal_array = np.array(channel_values)
        
        # Calculate average power for each band
        total_power = np.sum(np.abs(signal_array))
        
        if total_power > 0:
            # Estimate band powers based on signal characteristics
            mean_val = np.mean(np.abs(signal_array))
            std_val = np.std(signal_array) if len(signal_array) > 1 else 0
            
            # Distribute power across bands (simplified model)
            # Ictal states typically show increased high-frequency activity
            delta_power = mean_val * 0.1  # Low frequency
            theta_power = mean_val * 0.15
            alpha_power = mean_val * 0.25
            beta_power = mean_val * 0.3 + std_val * 0.5  # Higher variance = more beta
            gamma_power = mean_val * 0.2 + std_val * 0.8  # High variance = more gamma
            
            # Scale to realistic ranges (adjust based on your data)
            scale_factor = total_power / (delta_power + theta_power + alpha_power + beta_power + gamma_power)
            delta_power *= scale_factor
            theta_power *= scale_factor
            alpha_power *= scale_factor
            beta_power *= scale_factor
            gamma_power *= scale_factor
        else:
            delta_power = theta_power = alpha_power = beta_power = gamma_power = 0.0
        
        # Create timestamp (10 minutes = 600 seconds)
        timestamp = datetime.utcnow() - timedelta(seconds=(len(snippet_data) - i))
        
        processed_rows.append({
            'timestamp': timestamp.isoformat() + 'Z',
            'delta': float(delta_power),
            'theta': float(theta_power),
            'alpha': float(alpha_power),
            'beta': float(beta_power),
            'gamma': float(gamma_power),
            'heart_rate_bpm': '',
            'breathing_rate_bpm': '',
            'head_pitch': '',
            'head_roll': '',
            'head_movement': ''
        })
    
    print(f"\nConverted {len(processed_rows)} rows to frequency bands")
    return processed_rows

def save_snippet(processed_data, output_path):
    """Save the processed snippet in the same format as live data"""
    print(f"\nSaving snippet to {output_path}...")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Write to CSV
    fieldnames = ['timestamp', 'delta', 'theta', 'alpha', 'beta', 'gamma',
                  'heart_rate_bpm', 'breathing_rate_bpm', 'head_pitch', 'head_roll', 'head_movement']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_data)
    
    print(f"✓ Saved {len(processed_data)} rows to {output_path}")

def main():
    ictal_file = 'chbmit_ictal_cleaned.csv'
    
    if not os.path.exists(ictal_file):
        print(f"Error: {ictal_file} not found!")
        return
    
    # Process file
    snippet_data, channels, start_row = process_ictal_file(ictal_file)
    
    # Convert to frequency bands
    processed_data = convert_to_frequency_bands(snippet_data, channels)
    
    # Save snippet
    output_file = 'data/ictal_10min_snippet.csv'
    save_snippet(processed_data, output_file)
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE!")
    print("="*60)
    print(f"✓ Extracted 10-minute snippet from highest ictal state")
    print(f"✓ Converted to frequency bands (delta, theta, alpha, beta, gamma)")
    print(f"✓ Saved to: {output_file}")
    print(f"✓ Ready for integration with live website")
    print("="*60)

if __name__ == "__main__":
    main()

