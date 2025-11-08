/**
 * Main Application Entry Point
 * Initializes all modules and handles app-wide functionality
 */

class MigroMinderApp {
    constructor() {
        this.navigation = null;
        this.dashboard = null;
        this.animations = null;
        this.api = null;
        this.dataUpdateInterval = null;
        this.updateInterval = null;
        this.init();
    }

    async init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.start());
        } else {
            this.start();
        }
    }

    start() {
        console.log('🧠 MigroMinder App Initializing...');
        
        // Initialize modules (they auto-initialize, but we can reference them)
        this.api = window.API || null;

        // Setup event listeners
        this.setupEventListeners();
        
        // Initialize real-time data updates
        this.startDataUpdates();

        console.log('✅ MigroMinder App Ready');
    }

    setupEventListeners() {
        // Start Session button
        const startSessionBtn = document.querySelector('.btn-primary, .cta-button.primary');
        if (startSessionBtn) {
            startSessionBtn.addEventListener('click', () => {
                this.handleStartSession();
            });
        }

        // View Dashboard button
        const viewDashboardBtn = document.querySelector('.btn-secondary, .cta-button.secondary');
        if (viewDashboardBtn) {
            viewDashboardBtn.addEventListener('click', () => {
                this.scrollToSection('dashboard');
            });
        }

        // Window resize handler
        window.addEventListener('resize', () => {
            this.handleResize();
        });

        // Page visibility handler
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseUpdates();
            } else {
                this.resumeUpdates();
            }
        });
    }

    async handleStartSession() {
        console.log('Starting MigroMinder session...');
        
        const btn = document.querySelector('.btn-primary, .cta-button.primary');
        if (!btn) return;
        
        const originalText = btn.textContent;
        btn.textContent = 'Connecting...';
        btn.disabled = true;

        try {
            if (this.api) {
                const connected = await this.api.checkConnection();
                if (connected) {
                    this.startRealTimeMonitoring();
                    btn.textContent = 'Session Active';
                    btn.classList.add('btn-accent');
                } else {
                    btn.textContent = 'Session Active (Mock Mode)';
                }
            } else {
                btn.textContent = 'Session Active (Mock Mode)';
            }
        } catch (error) {
            console.error('Error starting session:', error);
            btn.textContent = originalText;
            btn.disabled = false;
        }
    }

    startRealTimeMonitoring() {
        if (this.dataUpdateInterval) clearInterval(this.dataUpdateInterval);
        
        this.dataUpdateInterval = setInterval(async () => {
            if (this.api) {
                const eegData = await this.api.getEEGData();
                this.checkMigrainePatterns(eegData);
            }
        }, 3000);
    }

    checkMigrainePatterns(eegData) {
        if (eegData.attention < 40 && eegData.meditation < 30) {
            console.warn('⚠️ Potential migraine pattern detected');
            this.handleMigraineWarning();
        }
    }

    handleMigraineWarning() {
        if (this.api) {
            this.api.controlLight(20);
        }
        this.showNotification('Migraine pattern detected. Lights dimmed automatically.');
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            background: var(--accent-yellow);
            color: var(--text-dark);
            padding: 1rem 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }

    scrollToSection(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) {
            const headerOffset = 80;
            const elementPosition = section.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

            window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
            });
        }
    }

    startDataUpdates() {
        if (this.updateInterval) clearInterval(this.updateInterval);
        
        this.updateInterval = setInterval(async () => {
            if (this.api) {
                const envData = await this.api.getEnvironmentData();
                this.updateEnvironmentDisplay(envData);
            }
        }, 5000);
    }

    updateEnvironmentDisplay(data) {
        // Update environment metrics
        const envMetrics = document.querySelectorAll('.env-metric strong');
        // Implementation for updating display
    }

    pauseUpdates() {
        if (this.dataUpdateInterval) clearInterval(this.dataUpdateInterval);
        if (this.updateInterval) clearInterval(this.updateInterval);
    }

    resumeUpdates() {
        this.startDataUpdates();
    }

    handleResize() {
        if (window.innerWidth > 768 && window.Navigation) {
            // Close mobile menu if open
        }
    }
}

// Initialize app
const app = new MigroMinderApp();
window.MigroMinderApp = app;
