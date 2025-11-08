#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Muse 2 Stream Starter
Connects to MUSE-F7DD (or first available Muse device) and starts streaming
"""

import sys
import signal
from muselsl import stream, list_muses


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n" + "="*60)
    print("✅ Streaming stopped by user.")
    print("="*60)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main function to start Muse streaming"""
    print("="*60)
    print("Muse 2 Stream Starter")
    print("="*60)
    print("\nLooking for Muse devices...")
    print("Make sure your Muse 2 is turned on and Bluetooth is enabled.")
    print("Press Ctrl+C to stop streaming.\n")
    
    # Scan for devices
    print("Scanning for Muse devices (this may take up to 10 seconds)...")
    try:
        devices = list_muses()
    except Exception as e:
        print(f"Error scanning for devices: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your Muse 2 is turned on")
        print("2. Check Bluetooth is enabled on your Mac")
        print("3. Try turning the Muse 2 off and on again")
        sys.exit(1)
    
    if not devices:
        print("❌ No Muse devices found.")
        print("\nTroubleshooting:")
        print("1. Make sure your Muse 2 is turned on")
        print("2. Check Bluetooth is enabled on your Mac")
        print("3. Try turning the Muse 2 off and on again")
        print("4. Move closer to your computer")
        print("5. Make sure the Muse 2 is not connected to another device")
        sys.exit(1)
    
    # Show found devices
    print(f"\n✅ Found {len(devices)} Muse device(s):")
    for i, device in enumerate(devices):
        print(f"  {i + 1}. {device['name']} - {device['address']}")
    
    # Find MUSE-F7DD or use first device
    target_device = None
    target_name = None
    
    for device in devices:
        if "F7DD" in device['name'].upper() or device['name'] == "Muse-F7DD":
            target_device = device['address']
            target_name = device['name']
            print(f"\n✓ Found target device: {target_name}")
            break
    
    if not target_device:
        # Use first device if MUSE-F7DD not found
        target_device = devices[0]['address']
        target_name = devices[0]['name']
        print(f"\n⚠️  MUSE-F7DD not found, using: {target_name}")
    
    print(f"\n{'='*60}")
    print(f"Target Device: {target_name}")
    print(f"Address: {target_device}")
    print(f"{'='*60}")
    
    # Check if this is the right device
    if "F7DD" not in target_name.upper():
        print(f"\n⚠️  WARNING: MUSE-F7DD not found!")
        print(f"   Found device: {target_name}")
        print(f"   If this is not your device, please:")
        print(f"   1. Turn off {target_name}")
        print(f"   2. Turn on MUSE-F7DD")
        print(f"   3. Run this script again")
        response = input(f"\nContinue with {target_name}? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting...")
            sys.exit(0)
    
    # Important: Check if device might be connected elsewhere
    print("\n⚠️  IMPORTANT: Before connecting:")
    print("   1. ✗ Close the Muse app if it's open")
    print("   2. ✗ Make sure the device is NOT connected to another app")
    print("   3. ✓ The device should be within 1 meter of your computer")
    print("   4. ✓ Make sure the device is fully charged")
    print("   5. ✓ Pair the device in macOS Bluetooth settings first (if not already paired)")
    
    input("\nPress Enter to continue with connection...")
    
    print("\n🔌 Establishing Bluetooth connection...")
    print("⏳ This may take 15-30 seconds")
    print("   (The device will try to connect 3 times if needed)\n")
    
    # Small delay to ensure device is ready
    import time
    time.sleep(2)
    
    # IMPORTANT: stream() is a SYNCHRONOUS function, not async
    # It handles async operations internally via muselsl's _wait function
    try:
        # Start streaming with all sensors enabled
        # The stream() function will:
        # - Connect to the device
        # - Print "Connected." when successful
        # - Print "Streaming EEG PPG ACC GYRO..." when streaming starts
        # - Run continuously until Ctrl+C
        stream(
            address=target_device,
            ppg_enabled=True,
            acc_enabled=True,
            gyro_enabled=True,
            eeg_disabled=False,
            backend='auto',
            retries=3
        )
        
        # If we get here, streaming stopped (shouldn't happen normally)
        print("\n" + "="*60)
        print("⚠️  Streaming ended unexpectedly.")
        print("="*60)
    except KeyboardInterrupt:
        # User stopped it - handled by signal handler
        pass
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error: {error_msg}")
        
        if "not found" in error_msg.lower() or "device" in error_msg.lower():
            print("\n" + "="*60)
            print("❌ CONNECTION FAILED - DEVICE NOT FOUND")
            print("="*60)
            print("\nThe device was found during scanning but connection failed.")
            print("This usually means the device is connected elsewhere or needs pairing.")
            print("\n🔧 FIX STEPS (try in order):")
            print("\n1. CLOSE THE MUSE APP")
            print("   - Make sure the official Muse app is completely closed")
            print("   - Check if it's running in the background")
            print("\n2. DISCONNECT FROM OTHER APPS")
            print("   - The device can only connect to ONE app at a time")
            print("   - Close any other apps using the Muse device")
            print("\n3. PAIR THE DEVICE IN macOS")
            print("   - Open System Settings > Bluetooth")
            print("   - Find your Muse device (MuseS-79A4 or MUSE-F7DD)")
            print("   - Click 'Connect' to pair it")
            print("   - Wait for it to show 'Connected'")
            print("   - Then disconnect it from System Settings")
            print("   - Now run this script again")
            print("\n4. RESTART THE DEVICE")
            print("   - Turn off your Muse 2 completely")
            print("   - Wait 10 seconds")
            print("   - Turn it back on")
            print("   - Wait for it to fully start (LED indicators)")
            print("\n5. TRY AGAIN")
            print("   - Move closer to your computer (within 1 meter)")
            print("   - Make sure device is fully charged")
            print("   - Run this script again")
            print("\n6. CHECK DEVICE NAME")
            print(f"   - Found device: {target_name}")
            print("   - If this is not your device, turn on the correct one")
            print("   - Make sure MUSE-F7DD is turned on if that's what you want")
        elif "timeout" in error_msg.lower():
            print("\nConnection timed out. The device may be:")
            print("- Too far away")
            print("- Low on battery")
            print("- Connected to another device")
            print("- Not properly powered on")
        else:
            import traceback
            print("\nFull error details:")
            traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    main()
