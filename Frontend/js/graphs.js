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
            headPitch: [],
            severity: [],
            stressIndex: [],
            arousalIndex: []
        };
        
        this.charts = {};
        this.init();
    }

    init() {
        this.setupEEGGraph();
        this.setupHeartRateGraph();
        this.setupBreathingGraph();
        this.setupHeadTiltGraph();
        this.setupSeverityGraph();
        this.setupStressGraph();
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

    setupHeartRateGraph() {
        const ctx = document.getElementById('heartRateGraph');
        if (!ctx) return;

        this.charts.heartRate = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Heart Rate (BPM)',
                    data: [],
                    borderColor: '#E53E3E',
                    backgroundColor: 'rgba(229, 62, 62, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
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
                            text: 'BPM'
                        },
                        beginAtZero: false,
                        suggestedMin: 50,
                        suggestedMax: 120
                    }
                },
                animation: {
                    duration: 0
                }
            }
        });
    }

    setupBreathingGraph() {
        const ctx = document.getElementById('breathingGraph');
        if (!ctx) return;

        this.charts.breathing = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Breathing Rate (per min)',
                    data: [],
                    borderColor: '#3182CE',
                    backgroundColor: 'rgba(49, 130, 206, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
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
                            text: 'Breaths/min'
                        },
                        beginAtZero: false,
                        suggestedMin: 10,
                        suggestedMax: 25
                    }
                },
                animation: {
                    duration: 0
                }
            }
        });
    }

    setupHeadTiltGraph() {
        const ctx = document.getElementById('headTiltGraph');
        if (!ctx) return;

        this.charts.headTilt = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Head Pitch (degrees)',
                    data: [],
                    borderColor: '#38A169',
                    backgroundColor: 'rgba(56, 161, 105, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
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
                            text: 'Degrees'
                        },
                        beginAtZero: false,
                        suggestedMin: -30,
                        suggestedMax: 30
                    }
                },
                animation: {
                    duration: 0
                }
            }
        });
    }

    setupSeverityGraph() {
        const ctx = document.getElementById('severityGraph');
        if (!ctx) return;

        this.charts.severity = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Migraine Severity',
                    data: [],
                    borderColor: '#FFC107',
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
                    tension: 0.4,
                    fill: true,
                    borderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    annotation: {
                        annotations: {
                            lowThreshold: {
                                type: 'line',
                                yMin: 0.25,
                                yMax: 0.25,
                                borderColor: '#38A169',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {
                                    content: 'Low Risk',
                                    enabled: true
                                }
                            },
                            mediumThreshold: {
                                type: 'line',
                                yMin: 0.75,
                                yMax: 0.75,
                                borderColor: '#FFC107',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {
                                    content: 'Medium Risk',
                                    enabled: true
                                }
                            }
                        }
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
                            text: 'Severity (0.0 - 1.0)'
                        },
                        min: 0,
                        max: 1
                    }
                },
                animation: {
                    duration: 0
                }
            }
        });
    }

    setupStressGraph() {
        const ctx = document.getElementById('stressGraph');
        if (!ctx) return;

        this.charts.stress = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Stress Index',
                        data: [],
                        borderColor: '#E53E3E',
                        backgroundColor: 'rgba(229, 62, 62, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Arousal Index',
                        data: [],
                        borderColor: '#FFC107',
                        backgroundColor: 'rgba(255, 193, 7, 0.1)',
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
                            text: 'Index Value'
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
        this.dataHistory.severity.push(data.migraine_severity || 0);
        this.dataHistory.stressIndex.push(data.stress_index || 0);
        this.dataHistory.arousalIndex.push(data.arousal_index || 0);

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
            this.dataHistory.severity.shift();
            this.dataHistory.stressIndex.shift();
            this.dataHistory.arousalIndex.shift();
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

        // Update Heart Rate graph
        if (this.charts.heartRate) {
            this.charts.heartRate.data.labels = labels;
            this.charts.heartRate.data.datasets[0].data = this.dataHistory.heartRate;
            this.charts.heartRate.update('none');
        }

        // Update Breathing graph
        if (this.charts.breathing) {
            this.charts.breathing.data.labels = labels;
            this.charts.breathing.data.datasets[0].data = this.dataHistory.breathingRate;
            this.charts.breathing.update('none');
        }

        // Update Head Tilt graph
        if (this.charts.headTilt) {
            this.charts.headTilt.data.labels = labels;
            this.charts.headTilt.data.datasets[0].data = this.dataHistory.headPitch;
            this.charts.headTilt.update('none');
        }

        // Update Severity graph
        if (this.charts.severity) {
            this.charts.severity.data.labels = labels;
            this.charts.severity.data.datasets[0].data = this.dataHistory.severity;
            this.charts.severity.update('none');
        }

        // Update Stress graph
        if (this.charts.stress) {
            this.charts.stress.data.labels = labels;
            this.charts.stress.data.datasets[0].data = this.dataHistory.stressIndex;
            this.charts.stress.data.datasets[1].data = this.dataHistory.arousalIndex;
            this.charts.stress.update('none');
        }
    }
}

// Initialize graphs when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.realTimeGraphs = new RealTimeGraphs();
});

