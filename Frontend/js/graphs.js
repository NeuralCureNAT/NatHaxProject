/**
 * Real-time Graphs Module
 * Displays real-time graphs for EEG, heart rate, breathing, head tilt, and migraine severity
 */

class RealTimeGraphs {
    constructor() {
        this.maxDataPoints = 60; // Show last 60 data points (2 minutes at 2s intervals)
        this.dataHistory = {
            timestamps: [],
            delta: [],
            theta: [],
            alpha: [],
            beta: [],
            gamma: [],
            heartRate: [],
            breathingRate: [],
            headPitch: []
        };
        
        this.charts = {};
        this.init();
    }

    init() {
        this.setupEEGGraph();
    }

    setupEEGGraph() {
        const ctx = document.getElementById('eegGraph');
        if (!ctx) return;

        this.charts.eeg = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Delta',
                        data: [],
                        borderColor: '#FF6B6B',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Theta',
                        data: [],
                        borderColor: '#4ECDC4',
                        backgroundColor: 'rgba(78, 205, 196, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Alpha',
                        data: [],
                        borderColor: '#95E1D3',
                        backgroundColor: 'rgba(149, 225, 211, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Beta',
                        data: [],
                        borderColor: '#F38181',
                        backgroundColor: 'rgba(243, 129, 129, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Gamma',
                        data: [],
                        borderColor: '#AA96DA',
                        backgroundColor: 'rgba(170, 150, 218, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Power (μV²)'
                        },
                        beginAtZero: true
                    }
                },
                animation: {
                    duration: 0
                }
            }
        });
    }


    addDataPoint(data) {
        if (!data || !data.connected) {
            return; // Don't add data if not connected
        }

        const timestamp = data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
        
        // Add to history
        this.dataHistory.timestamps.push(timestamp);
        this.dataHistory.delta.push(data.delta || 0);
        this.dataHistory.theta.push(data.theta || 0);
        this.dataHistory.alpha.push(data.alpha || 0);
        this.dataHistory.beta.push(data.beta || 0);
        this.dataHistory.gamma.push(data.gamma || 0);
        this.dataHistory.heartRate.push(data.heart_rate_bpm || 0);
        this.dataHistory.breathingRate.push(data.breathing_rate_bpm || 0);
        this.dataHistory.headPitch.push(data.head_pitch || 0);
        
        // Update numerical displays
        this.updateNumericalDisplays(data);

        // Limit history size
        if (this.dataHistory.timestamps.length > this.maxDataPoints) {
            this.dataHistory.timestamps.shift();
            this.dataHistory.delta.shift();
            this.dataHistory.theta.shift();
            this.dataHistory.alpha.shift();
            this.dataHistory.beta.shift();
            this.dataHistory.gamma.shift();
            this.dataHistory.heartRate.shift();
            this.dataHistory.breathingRate.shift();
            this.dataHistory.headPitch.shift();
        }

        // Update all charts
        this.updateCharts();
    }

    updateCharts() {
        const labels = this.dataHistory.timestamps;

        // Update EEG graph
        if (this.charts.eeg) {
            this.charts.eeg.data.labels = labels;
            this.charts.eeg.data.datasets[0].data = this.dataHistory.delta;
            this.charts.eeg.data.datasets[1].data = this.dataHistory.theta;
            this.charts.eeg.data.datasets[2].data = this.dataHistory.alpha;
            this.charts.eeg.data.datasets[3].data = this.dataHistory.beta;
            this.charts.eeg.data.datasets[4].data = this.dataHistory.gamma;
            this.charts.eeg.update('none');
        }
    }

    updateNumericalDisplays(data) {
        // Update current values
        const heartRateEl = document.getElementById('heart-rate-display');
        if (heartRateEl) {
            if (data.heart_rate_bpm && data.heart_rate_bpm > 0) {
                heartRateEl.textContent = Math.round(data.heart_rate_bpm);
            } else {
                heartRateEl.textContent = '--';
            }
        }

        const breathingRateEl = document.getElementById('breathing-rate-display');
        if (breathingRateEl) {
            if (data.breathing_rate_bpm && data.breathing_rate_bpm > 0) {
                breathingRateEl.textContent = data.breathing_rate_bpm.toFixed(1);
            } else {
                breathingRateEl.textContent = '--';
            }
        }

        const headPitchEl = document.getElementById('head-pitch-display');
        if (headPitchEl) {
            if (data.head_pitch !== null && data.head_pitch !== undefined) {
                headPitchEl.textContent = data.head_pitch.toFixed(1);
            } else {
                headPitchEl.textContent = '--';
            }
        }

        const headRollEl = document.getElementById('head-roll-display');
        if (headRollEl) {
            if (data.head_roll !== null && data.head_roll !== undefined) {
                headRollEl.textContent = data.head_roll.toFixed(1);
            } else {
                headRollEl.textContent = '--';
            }
        }

        // Update history list (show last 10 values)
        this.updateHistoryList(data);
    }

    updateHistoryList(data) {
        const historyList = document.getElementById('metrics-history-list');
        if (!historyList) return;

        // Get last 10 entries
        const recentCount = Math.min(10, this.dataHistory.timestamps.length);
        if (recentCount === 0) {
            historyList.innerHTML = '<p class="no-data-text">No data recorded yet</p>';
            return;
        }

        // Build history HTML
        let historyHTML = '';
        for (let i = this.dataHistory.timestamps.length - recentCount; i < this.dataHistory.timestamps.length; i++) {
            const timestamp = this.dataHistory.timestamps[i];
            const hr = this.dataHistory.heartRate[i] > 0 ? Math.round(this.dataHistory.heartRate[i]) : '--';
            const br = this.dataHistory.breathingRate[i] > 0 ? this.dataHistory.breathingRate[i].toFixed(1) : '--';
            const pitch = this.dataHistory.headPitch[i] !== 0 ? this.dataHistory.headPitch[i].toFixed(1) : '--';
            
            historyHTML += `
                <div class="history-item">
                    <span class="history-time">${timestamp}</span>
                    <span class="history-values">
                        HR: ${hr} | BR: ${br} | Pitch: ${pitch}°
                    </span>
                </div>
            `;
        }

        historyList.innerHTML = historyHTML;
    }
}

// Initialize graphs when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.realTimeGraphs = new RealTimeGraphs();
});

