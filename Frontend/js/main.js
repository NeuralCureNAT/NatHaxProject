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
        const startSessionBtn = document.querySelector('.btn.btn-primary');
        if (startSessionBtn) {
            startSessionBtn.addEventListener('click', () => {
                this.handleStartSession();
            });
        }

        // View Dashboard button
        const viewDashboardBtn = document.querySelector('.btn.btn-secondary');
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
        const btn = document.querySelector('.btn.btn-primary');
        if (!btn) return;

        if (btn.dataset.active === '1') {
            this.showNotification('Session is already active.');
            return;
        }   

        const originalText = btn.textContent;
        btn.textContent = 'Connecting...';
        btn.disabled = true;

        try {
            const connected = this.api ? await this.api.checkConnection() : false;

            this.startRealTimeMonitoring();
            btn.textContent = connected ? 'Session Active' : 'Session Active (Mock Mode)';
            btn.classList.add('btn-accent');
            btn.dataset.active = '1';
            btn.disabled = false;

        } catch (e) {
            console.error(e);
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
    // Expects: { light: 0-100, temperature: C, humidity: %, timestamp }
    const env = {
        light: document.querySelector('.env-metric:nth-child(1) strong'),
        temperature: document.querySelector('.env-metric:nth-child(2) strong'),
        humidity: document.querySelector('.env-metric:nth-child(3) strong'),
    };
    if (env.light && typeof data.light === 'number') env.light.textContent = `${data.light}%`;
    if (env.temperature && typeof data.temperature === 'number') env.temperature.textContent = `${data.temperature}°C`;
    if (env.humidity && typeof data.humidity === 'number') env.humidity.textContent = `${data.humidity}%`;
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
