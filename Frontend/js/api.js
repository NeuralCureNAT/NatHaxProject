/**
 * API Module
 * Handles communication with backend/Flask API
 * Falls back to mock data if API is unavailable
 */

class API {
  constructor() {
    // Backend runs on port 5050 by default (see Backend/app.py)
    this.baseURL = 'http://localhost:5050/api';
    this.init();
  }

  init() {
    this.checkConnection();
  }

  // Small helper to avoid repeating ok checks
  async fetchJSON(path, options = {}) {
    // Add cache-busting timestamp for real-time data endpoints
    const cacheBuster = path.includes('/eeg/current') || path.includes('/prediction/current') || path.includes('/muse/status')
      ? `?t=${Date.now()}`
      : '';
    
    const res = await fetch(`${this.baseURL}${path}${cacheBuster}`, {
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      cache: 'no-cache', // Prevent browser caching for real-time data
      ...options,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status} ${res.statusText} — ${text}`);
    }
    return res.json();
  }

  async checkConnection() {
    try {
      await this.fetchJSON('/health');
      console.log('API connection successful');
      return true;
    } catch (error) {
      console.warn('API not available, using mock data:', error);
      return false;
    }
  }

  // Check Muse 2 connection status
  async getMuseStatus() {
    try {
      return await this.fetchJSON('/muse/status');
    } catch (error) {
      console.error('Error checking Muse status:', error);
      return {
        connected: false,
        has_data: false,
        message: 'Backend not available'
      };
    }
  }

  // Get real-time EEG data from Muse 2
  async getEEGData() {
    try {
      const data = await this.fetchJSON('/eeg/current');
      // Don't return mock data - return actual data or null
      return data;
    } catch (error) {
      console.error('Error fetching EEG data:', error);
      // Return null data instead of mock
      return {
        connected: false,
        timestamp: null,
        delta: null,
        theta: null,
        alpha: null,
        beta: null,
        gamma: null
      };
    }
  }

  // Get current migraine prediction
  async getPrediction() {
    try {
      const data = await this.fetchJSON('/prediction/current');
      return data;
    } catch (error) {
      console.error('Error fetching prediction:', error);
      return {
        connected: false,
        migraine_severity: null,
        migraine_stage: null,
        migraine_interpretation: 'Connect Muse 2 to see predictions',
        migraine_risk_level: null
      };
    }
  }

  // Get migraine event history
  async getMigraineHistory(limit = 10) {
    try {
      return await this.fetchJSON(`/migraines?limit=${encodeURIComponent(limit)}`);
    } catch (error) {
      console.error('Error fetching migraine history:', error);
      return this.getMockMigraineHistory();
    }
  }

  // Log a new migraine event
  async logMigraineEvent(eventData) {
    try {
      return await this.fetchJSON('/migraines', {
        method: 'POST',
        body: JSON.stringify(eventData),
      });
    } catch (error) {
      console.error('Error logging migraine event:', error);
      return { success: false, error: error.message };
    }
  }

  // Get environment sensor data (Arduino)
  async getEnvironmentData() {
    try {
      return await this.fetchJSON('/environment/current');
    } catch (error) {
      console.error('Error fetching environment data:', error);
      return this.getMockEnvironmentData();
    }
  }

  // Control Arduino light module
  async controlLight(brightness) {
    try {
      return await this.fetchJSON('/arduino/light', {
        method: 'POST',
        body: JSON.stringify({ brightness }),
      });
    } catch (error) {
      console.error('Error controlling light:', error);
      return { success: false, error: error.message };
    }
  }

  // Log user profile from onboarding
  async logUserProfile(userData) {
    try {
      return await this.fetchJSON('/user/profile', {
        method: 'POST',
        body: JSON.stringify(userData),
      });
    } catch (error) {
      console.error('Error logging user profile:', error);
      return { success: false, error: error.message };
    }
  }

  // -------- Removed mock data - only show real data --------

  getMockMigraineHistory() {
    return [
      {
        id: 1,
        timestamp: new Date(Date.now() - 86400000).toISOString(),
        severity: 'moderate',
        triggers: ['stress', 'screen_time'],
        duration: 120,
      },
      {
        id: 2,
        timestamp: new Date(Date.now() - 172800000).toISOString(),
        severity: 'mild',
        triggers: ['noise'],
        duration: 45,
      },
    ];
  }

  getMockEnvironmentData() {
    return {
      light: Math.floor(Math.random() * 100),
      temperature: Math.floor(Math.random() * 10) + 20,
      humidity: Math.floor(Math.random() * 30) + 40,
      timestamp: new Date().toISOString(),
    };
  }
}

// Export for use in other modules
window.API = new API();
