/**
 * Dashboard Module
 * Handles dashboard interactions, data visualization, and real-time updates
 */

class Dashboard {
    constructor() {
        this.focusMeter = document.querySelector('.meter-circle');
        this.meterValue = document.querySelector('.meter-value');
        this.meterLabel = document.querySelector('.meter-label');
        this.isConnected = false;
        this.hasData = false;
        this.init();
    }

    init() {
        this.setupFocusMeter();
        this.setupDashboardCards();
        this.checkConnection();
        this.simulateDataUpdates();
    }

    async checkConnection() {
        try {
            const status = await window.API.getMuseStatus();
            this.isConnected = status.connected || false;
            this.hasData = status.has_data || false;
            this.updateConnectionStatus(status);
        } catch (error) {
            console.error('Error checking connection:', error);
            this.isConnected = false;
            this.hasData = false;
            this.updateConnectionStatus({ connected: false, message: 'Backend not available' });
        }
    }

    updateConnectionStatus(status) {
        // Show connection status indicator
        let statusIndicator = document.getElementById('muse-connection-status');
        if (!statusIndicator) {
            // Create status indicator if it doesn't exist
            statusIndicator = document.createElement('div');
            statusIndicator.id = 'muse-connection-status';
            statusIndicator.className = 'muse-status-indicator';
            const dashboardSection = document.querySelector('#dashboard .container');
            if (dashboardSection) {
                dashboardSection.insertBefore(statusIndicator, dashboardSection.firstChild);
            }
        }

        if (status.connected) {
            statusIndicator.className = 'muse-status-indicator connected';
            statusIndicator.innerHTML = '🟢 Muse 2 Connected';
        } else {
            statusIndicator.className = 'muse-status-indicator disconnected';
            statusIndicator.innerHTML = '🔴 Muse 2 Not Connected - ' + (status.message || 'Please connect your Muse 2 headband');
        }
    }

    setupFocusMeter() {
        if (!this.focusMeter || !this.meterValue) return;

        // Initialize with "No Data" state
        this.updateFocusMeter(null);
    }

    updateFocusMeter(value) {
        if (!this.meterValue || !this.focusMeter) return;

        if (value === null || value === undefined || !this.hasData) {
            // Show "No Data" state
            this.meterValue.textContent = '--';
            this.focusMeter.style.background = `conic-gradient(
                var(--bg-light-beige) 0% 100%
            )`;
            if (this.meterLabel) {
                this.meterLabel.textContent = 'No Data';
            }
            return;
        }

        const percentage = Math.max(0, Math.min(100, value));
        this.meterValue.textContent = `${percentage}%`;

        // Update conic gradient
        this.focusMeter.style.background = `
            conic-gradient(
                var(--accent-yellow) 0% ${percentage}%,
                var(--bg-light-beige) ${percentage}% 100%
            )
        `;

        // Update label
        if (this.meterLabel) {
            if (percentage >= 80) {
                this.meterLabel.textContent = 'High Focus';
            } else if (percentage >= 60) {
                this.meterLabel.textContent = 'Moderate Focus';
            } else if (percentage >= 40) {
                this.meterLabel.textContent = 'Low Focus';
            } else {
                this.meterLabel.textContent = 'Very Low';
            }
        }
    }

    setupDashboardCards() {
        const cards = document.querySelectorAll('.dashboard-card');
        
        cards.forEach((card, index) => {
            card.style.animationDelay = `${index * 0.1}s`;
            card.classList.add('fade-in');

            // Add click interaction
            card.addEventListener('click', () => {
                this.handleCardClick(card);
            });
        });
    }

    handleCardClick(card) {
        const cardType = card.classList.contains('focus-meter') ? 'focus' :
                        card.classList.contains('session-info') ? 'session' :
                        card.classList.contains('environment') ? 'environment' :
                        'recommendations';

        console.log(`Dashboard card clicked: ${cardType}`);
        // Add detailed view or modal functionality here
    }

    simulateDataUpdates() {
        // Check connection status periodically
        setInterval(() => {
            this.checkConnection();
        }, 5000); // Check every 5 seconds
        
        // Try to use real-time data first
        this.updateRealTimeData();
        
        // Set up periodic updates every 2 seconds
        setInterval(() => {
            this.updateRealTimeData();
        }, 2000);
    }

    updateEnvironmentData() {
        const envMetrics = document.querySelectorAll('.env-metric strong');
        
        if (envMetrics.length > 0) {
            setInterval(() => {
                // Simulate environment changes
                // In production, this will use real sensor data
                const statuses = ['Optimal', 'Good', 'Fair', 'Needs Attention'];
                envMetrics.forEach(metric => {
                    if (Math.random() > 0.9) { // 10% chance to update
                        metric.textContent = statuses[Math.floor(Math.random() * statuses.length)];
                    }
                });
            }, 5000);
        }
    }

    // Method to update dashboard with real data (for API integration)
    updateWithRealData(data) {
        if (data.focusLevel !== undefined) {
            this.updateFocusMeter(data.focusLevel);
        }

        // Update other dashboard elements with real data
        // This will be called from the API module
    }

    // Update dashboard with real-time EEG data and predictions
    async updateRealTimeData() {
        try {
            const eegData = await window.API.getEEGData();
            
            // Check if we have real data
            this.hasData = eegData.connected === true && (
                (eegData.delta !== null && eegData.delta !== undefined && eegData.delta > 0) ||
                (eegData.theta !== null && eegData.theta !== undefined && eegData.theta > 0) ||
                (eegData.alpha !== null && eegData.alpha !== undefined && eegData.alpha > 0) ||
                (eegData.beta !== null && eegData.beta !== undefined && eegData.beta > 0) ||
                (eegData.gamma !== null && eegData.gamma !== undefined && eegData.gamma > 0)
            );
            
            // Only update if we have real data
            if (this.hasData) {
                // Update EEG frequency bands display
                this.updateEEGBands(eegData);
                
                // Update focus meter based on EEG data
                if (eegData.focus !== undefined && eegData.focus !== null) {
                    this.updateFocusMeter(eegData.focus);
                } else if (eegData.alpha !== undefined && eegData.alpha !== null && 
                          eegData.beta !== undefined && eegData.beta !== null) {
                    // Calculate focus from alpha/beta ratio
                    const focus = Math.min(100, Math.max(0, (eegData.beta / (eegData.alpha + eegData.beta + 0.001)) * 100));
                    this.updateFocusMeter(focus);
                } else {
                    this.updateFocusMeter(null);
                }
                
                // Update migraine prediction
                const prediction = await window.API.getPrediction();
                this.updateMigrainePrediction(prediction);
            } else {
                // Show "No Data" state
                this.updateEEGBands({ delta: null, theta: null, alpha: null, beta: null, gamma: null });
                this.updateFocusMeter(null);
                this.updateMigrainePrediction({
                    connected: false,
                    migraine_severity: null,
                    migraine_stage: null,
                    migraine_interpretation: 'Connect Muse 2 to see predictions',
                    migraine_risk_level: null
                });
            }
            
            // Update environment data (this can work independently)
            const envData = await window.API.getEnvironmentData();
            this.updateEnvironmentData(envData);
        } catch (error) {
            console.error('Error updating real-time data:', error);
            this.hasData = false;
            this.updateFocusMeter(null);
        }
    }

    updateEEGBands(data) {
        // Update frequency band displays if they exist
        const bands = ['delta', 'theta', 'alpha', 'beta', 'gamma'];
        bands.forEach(band => {
            const element = document.querySelector(`[data-band="${band}"]`);
            if (element) {
                if (data[band] !== undefined && data[band] !== null && data[band] > 0) {
                    element.textContent = parseFloat(data[band]).toFixed(2);
                } else {
                    element.textContent = '--';
                }
            }
        });
    }

    updateMigrainePrediction(prediction) {
        if (!prediction) {
            // Show "No Data" state
            const severityEl = document.getElementById('migraine-severity');
            if (severityEl) severityEl.textContent = '--';
            
            const stageEl = document.getElementById('migraine-stage');
            if (stageEl) {
                stageEl.textContent = 'No Data';
                stageEl.className = 'migraine-stage unknown';
            }
            
            const interpretationEl = document.getElementById('migraine-interpretation');
            if (interpretationEl) {
                interpretationEl.textContent = 'Connect Muse 2 to see predictions';
            }
            
            const riskEl = document.getElementById('migraine-risk');
            if (riskEl) {
                riskEl.className = 'risk-indicator unknown';
                riskEl.textContent = 'NO DATA';
            }
            return;
        }
        
        const hasData = prediction.connected === true && 
                       prediction.migraine_severity !== null && 
                       prediction.migraine_severity !== undefined;
        
        // Update severity display
        const severityEl = document.getElementById('migraine-severity');
        if (severityEl) {
            if (hasData) {
                severityEl.textContent = `${(prediction.migraine_severity * 100).toFixed(1)}%`;
            } else {
                severityEl.textContent = '--';
            }
        }
        
        // Update stage display
        const stageEl = document.getElementById('migraine-stage');
        if (stageEl) {
            if (hasData) {
                stageEl.textContent = prediction.migraine_stage || 'Unknown';
                stageEl.className = `migraine-stage ${prediction.migraine_risk_level || 'unknown'}`;
            } else {
                stageEl.textContent = 'No Data';
                stageEl.className = 'migraine-stage unknown';
            }
        }
        
        // Update interpretation
        const interpretationEl = document.getElementById('migraine-interpretation');
        if (interpretationEl) {
            interpretationEl.textContent = prediction.migraine_interpretation || 'Connect Muse 2 to see predictions';
        }
        
        // Update risk indicator
        const riskEl = document.getElementById('migraine-risk');
        if (riskEl) {
            if (hasData && prediction.migraine_risk_level) {
                riskEl.className = `risk-indicator ${prediction.migraine_risk_level}`;
                riskEl.textContent = prediction.migraine_risk_level.toUpperCase();
            } else {
                riskEl.className = 'risk-indicator unknown';
                riskEl.textContent = 'NO DATA';
            }
        }
    }

    updateEnvironmentData(envData) {
        if (!envData) return;
        
        const lightEl = document.querySelector('.env-metric:nth-child(1) strong');
        if (lightEl && envData.light !== undefined) {
            lightEl.textContent = `${envData.light}%`;
        }
        
        const tempEl = document.querySelector('.env-metric:nth-child(2) strong');
        if (tempEl && envData.temperature !== undefined) {
            tempEl.textContent = `${envData.temperature}°C`;
        }
        
        const humidityEl = document.querySelector('.env-metric:nth-child(3) strong');
        if (humidityEl && envData.humidity !== undefined) {
            humidityEl.textContent = `${envData.humidity}%`;
        }
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new Dashboard();
});

