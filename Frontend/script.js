// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Start Session button handler
const startSessionBtn = document.querySelector('.cta-button.primary');
if (startSessionBtn) {
    startSessionBtn.addEventListener('click', function() {
        // This will connect to your backend/Arduino integration
        console.log('Starting study session...');
        alert('Connecting to Muse 2 and sensors...\nSession will begin shortly!');
        // Add your session start logic here
    });
}

// View Dashboard button handler
const viewDashboardBtn = document.querySelector('.cta-button.secondary');
if (viewDashboardBtn) {
    viewDashboardBtn.addEventListener('click', function() {
        document.querySelector('#dashboard').scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    });
}

// Add scroll effect to navbar
window.addEventListener('scroll', function() {
    const header = document.querySelector('header');
    if (window.scrollY > 100) {
        header.style.boxShadow = '0 4px 30px rgba(0, 212, 255, 0.3)';
        header.style.background = 'rgba(15, 23, 42, 0.95)';
    } else {
        header.style.boxShadow = '0 4px 20px rgba(0, 212, 255, 0.1)';
        header.style.background = 'rgba(15, 23, 42, 0.9)';
    }
});

// Simulate real-time focus meter updates (for demo purposes)
// In production, this will connect to your EEG data stream
function updateFocusMeter() {
    const meterValue = document.querySelector('.meter-value');
    const meterCircle = document.querySelector('.meter-circle');
    
    if (meterValue && meterCircle) {
        // Simulate focus level changes (replace with real EEG data)
        const currentFocus = parseInt(meterValue.textContent);
        const newFocus = Math.max(60, Math.min(95, currentFocus + Math.floor(Math.random() * 6) - 3));
        
        meterValue.textContent = newFocus + '%';
        
        // Update the conic gradient
        const percentage = newFocus;
        meterCircle.style.background = `conic-gradient(
            var(--primary-color) 0% ${percentage}%,
            var(--bg-lighter) ${percentage}% 100%
        )`;
        
        // Update label based on focus level
        const meterLabel = document.querySelector('.meter-label');
        if (meterLabel) {
            if (newFocus >= 80) {
                meterLabel.textContent = 'High Focus';
            } else if (newFocus >= 60) {
                meterLabel.textContent = 'Moderate Focus';
            } else {
                meterLabel.textContent = 'Low Focus';
            }
        }
    }
}

// Update focus meter every 3 seconds (simulation)
// Replace this with real-time EEG data streaming
if (document.querySelector('.meter-value')) {
    setInterval(updateFocusMeter, 3000);
}

// Animate cards on scroll
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all cards and steps
document.querySelectorAll('.card, .step').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
});

// Add hover effects to feature cards
document.querySelectorAll('.feature-card').forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-8px) scale(1.02)';
    });
    
    card.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0) scale(1)';
    });
});

// Simulate environment data updates
function updateEnvironmentData() {
    const tempElement = document.querySelector('.env-metric strong');
    if (tempElement && tempElement.textContent.includes('°C')) {
        // Simulate temperature changes (replace with real sensor data)
        const currentTemp = parseInt(tempElement.textContent);
        const newTemp = Math.max(18, Math.min(26, currentTemp + Math.floor(Math.random() * 3) - 1));
        tempElement.textContent = newTemp + '°C';
    }
}

// Update environment data every 5 seconds (simulation)
if (document.querySelector('.env-metric')) {
    setInterval(updateEnvironmentData, 5000);
}

// Add click handlers for dashboard cards (for future interactivity)
document.querySelectorAll('.dashboard-card').forEach(card => {
    card.addEventListener('click', function() {
        // Add detailed view or modal functionality here
        console.log('Dashboard card clicked:', this.querySelector('h3').textContent);
    });
});

// Console log for debugging
console.log('StudyAmp - NeuralCare Frontend Loaded');
console.log('Ready to connect to Muse 2 EEG and Arduino sensors');
