/**
 * API Module
 * Handles communication with backend/Flask API
 * Placeholder for real API integration
 */

class API {
    constructor() {
        this.baseURL = 'http://localhost:5000/api'; // Update with your Flask backend URL
        this.init();
    }

    init() {
        // Check API connection on load
        this.checkConnection();
    }

    async checkConnection() {
        try {
            const response = await fetch(`${this.baseURL}/health`);
            if (response.ok) {
                console.log('API connection successful');
                return true;
            }
        } catch (error) {
            console.warn('API not available, using mock data:', error);
            return false;
        }
    }

    // Get real-time EEG data from Muse 2
    async getEEGData() {
        try {
            const response = await fetch(`${this.baseURL}/eeg/current`);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching EEG data:', error);
            return this.getMockEEGData();
        }
    }

    // Get migraine event history
    async getMigraineHistory(limit = 10) {
        try {
            const response = await fetch(`${this.baseURL}/migraines?limit=${limit}`);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching migraine history:', error);
            return this.getMockMigraineHistory();
        }
    }

    // Log a new migraine event
    async logMigraineEvent(eventData) {
        try {
            const response = await fetch(`${this.baseURL}/migraines`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(eventData),
            });
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error logging migraine event:', error);
            return { success: false, error: error.message };
        }
    }

    // Get environment sensor data (Arduino)
    async getEnvironmentData() {
        try {
            const response = await fetch(`${this.baseURL}/environment/current`);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching environment data:', error);
            return this.getMockEnvironmentData();
        }
    }

    // Control Arduino light module
    async controlLight(brightness) {
        try {
            const response = await fetch(`${this.baseURL}/arduino/light`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ brightness }),
            });
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error controlling light:', error);
            return { success: false, error: error.message };
        }
    }

    // Log user profile from onboarding
    async logUserProfile(userData) {
        try {
            const response = await fetch(`${this.baseURL}/user/profile`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(userData),
            });
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error logging user profile:', error);
            return { success: false, error: error.message };
        }
    }

    // Mock data for development/testing
    getMockEEGData() {
        return {
            focus: Math.floor(Math.random() * 30) + 70,
            attention: Math.floor(Math.random() * 20) + 60,
            meditation: Math.floor(Math.random() * 25) + 50,
            timestamp: new Date().toISOString()
        };
    }

    getMockMigraineHistory() {
        return [
            {
                id: 1,
                timestamp: new Date(Date.now() - 86400000).toISOString(),
                severity: 'moderate',
                triggers: ['stress', 'screen_time'],
                duration: 120
            },
            {
                id: 2,
                timestamp: new Date(Date.now() - 172800000).toISOString(),
                severity: 'mild',
                triggers: ['noise'],
                duration: 45
            }
        ];
    }

    getMockEnvironmentData() {
        return {
            light: Math.floor(Math.random() * 100),
            temperature: Math.floor(Math.random() * 10) + 20,
            humidity: Math.floor(Math.random() * 30) + 40,
            timestamp: new Date().toISOString()
        };
    }
}

// Export for use in other modules
window.API = new API();

