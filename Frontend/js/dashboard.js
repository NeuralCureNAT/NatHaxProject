/**
 * Dashboard Module
 * Handles dashboard interactions, data visualization, and real-time updates
 */

class Dashboard {
    constructor() {
        this.focusMeter = document.querySelector('.meter-circle');
        this.meterValue = document.querySelector('.meter-value');
        this.meterLabel = document.querySelector('.meter-label');
        this.init();
    }

    init() {
        this.setupFocusMeter();
        this.setupDashboardCards();
        this.simulateDataUpdates();
    }

    setupFocusMeter() {
        if (!this.focusMeter || !this.meterValue) return;

        // Initialize meter animation
        this.updateFocusMeter(78);
        
        // Animate meter on load
        setTimeout(() => {
            this.focusMeter.style.animation = 'pulse 2s ease-in-out infinite';
        }, 500);
    }

    updateFocusMeter(value) {
        if (!this.meterValue || !this.focusMeter) return;

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
        // Simulate real-time focus updates
        // In production, this will connect to Muse 2 EEG data stream
        if (this.meterValue) {
            setInterval(() => {
                const currentValue = parseInt(this.meterValue.textContent);
                const change = Math.floor(Math.random() * 6) - 3; // -3 to +3
                const newValue = Math.max(60, Math.min(95, currentValue + change));
                this.updateFocusMeter(newValue);
            }, 3000);
        }

        // Simulate environment data updates
        this.updateEnvironmentData();
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
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new Dashboard();
});

