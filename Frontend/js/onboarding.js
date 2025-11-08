/**
 * Onboarding Flow Handler
 * Manages multi-step questionnaire and user data collection
 */

class OnboardingFlow {
    constructor() {
        this.currentStep = 1;
        this.totalSteps = 7;
        this.userData = {
            frequency: null,
            severity: null,
            triggers: [],
            goal: null,
            hardware: null
        };
        this.init();
    }

    init() {
        this.updateProgress();
        this.loadSavedData();
        this.setupTriggerListeners();
    }

    setupTriggerListeners() {
        // Use event delegation for trigger buttons - only when step 4 is active
        document.addEventListener('click', (e) => {
            // Check if step 4 is active
            const step4 = document.getElementById('step4');
            if (!step4 || !step4.classList.contains('active')) {
                return;
            }

            // Find the trigger button that was clicked
            const triggerButton = e.target.closest('.option-card.selectable[data-trigger]');
            if (triggerButton) {
                e.preventDefault();
                e.stopPropagation();
                const triggerValue = triggerButton.getAttribute('data-trigger');
                console.log('Trigger button clicked:', triggerValue, triggerButton);
                if (triggerValue) {
                    this.toggleOption('triggers', triggerValue, triggerButton);
                }
            }
        });
    }

    updateProgress() {
        const progress = (this.currentStep / this.totalSteps) * 100;
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
    }

    showStep(stepNumber) {
        // Hide all steps
        document.querySelectorAll('.onboarding-step').forEach(step => {
            step.classList.remove('active');
        });

        // Show current step
        const currentStepElement = document.getElementById(`step${stepNumber}`);
        if (currentStepElement) {
            currentStepElement.classList.add('active');
            this.currentStep = stepNumber;
            this.updateProgress();
        }
    }

    nextStep() {
        if (this.currentStep < this.totalSteps) {
            // Validate current step before proceeding
            if (this.validateStep(this.currentStep)) {
                this.showStep(this.currentStep + 1);
                this.updateSummary();
            } else {
                this.showValidationMessage();
            }
        }
    }

    prevStep() {
        if (this.currentStep > 1) {
            this.showStep(this.currentStep - 1);
        }
    }

    validateStep(step) {
        switch(step) {
            case 2:
                return this.userData.frequency !== null;
            case 3:
                return this.userData.severity !== null;
            case 4:
                return this.userData.triggers.length > 0;
            case 5:
                return this.userData.goal !== null;
            case 6:
                return this.userData.hardware !== null;
            default:
                return true;
        }
    }

    showValidationMessage() {
        // Simple alert for now - can be replaced with better UI
        alert('Please select an option to continue');
    }

    selectOption(key, value) {
        this.userData[key] = value;
        
        // Visual feedback
        event.target.closest('.options-grid').querySelectorAll('.option-card').forEach(card => {
            card.classList.remove('selected');
        });
        event.target.closest('.option-card').classList.add('selected');

        // Auto-advance for single-select questions
        if (key !== 'triggers') {
            setTimeout(() => this.nextStep(), 300);
        } else {
            // Enable continue button for multi-select
            const continueBtn = document.getElementById('step4Next');
            if (continueBtn && this.userData.triggers.length > 0) {
                continueBtn.disabled = false;
            }
        }
    }

    toggleOption(key, value, element) {
        // Ensure we have the correct key (triggers, not trigger)
        const dataKey = key === 'trigger' ? 'triggers' : key;
        
        // Get the card element - use the passed element directly
        const card = element && (element.classList.contains('option-card') || element.closest('.option-card'))
            ? (element.classList.contains('option-card') ? element : element.closest('.option-card'))
            : null;

        if (!card) {
            console.error('Could not find option card element', { element, key, value });
            return;
        }

        // Ensure the array exists
        if (!this.userData[dataKey]) {
            this.userData[dataKey] = [];
        }

        const index = this.userData[dataKey].indexOf(value);

        if (index > -1) {
            // Deselect
            this.userData[dataKey].splice(index, 1);
            card.classList.remove('selected');
        } else {
            // Select
            this.userData[dataKey].push(value);
            card.classList.add('selected');
        }

        // Update continue button state
        const continueBtn = document.getElementById('step4Next');
        if (continueBtn) {
            continueBtn.disabled = this.userData.triggers.length === 0;
        }

        // Log for debugging
        console.log('Selected triggers:', this.userData.triggers);
    }

    updateSummary() {
        if (this.currentStep === 7) {
            // Update summary display
            const frequencyMap = {
                'daily': 'Daily',
                'weekly': 'Weekly',
                'monthly': 'Monthly',
                'occasional': 'Occasional'
            };

            const severityMap = {
                'mild': 'Mild',
                'moderate': 'Moderate',
                'severe': 'Severe',
                'debilitating': 'Debilitating'
            };

            const goalMap = {
                'prevent': 'Prevent Episodes',
                'reduce': 'Reduce Frequency',
                'manage': 'Better Management',
                'work': 'Stay Productive'
            };

            const frequencyEl = document.getElementById('summaryFrequency');
            const severityEl = document.getElementById('summarySeverity');
            const goalEl = document.getElementById('summaryGoal');

            if (frequencyEl) frequencyEl.textContent = frequencyMap[this.userData.frequency] || '-';
            if (severityEl) severityEl.textContent = severityMap[this.userData.severity] || '-';
            if (goalEl) goalEl.textContent = goalMap[this.userData.goal] || '-';
        }
    }

    saveData() {
        // Save to localStorage
        localStorage.setItem('migrominder_user_data', JSON.stringify(this.userData));
        localStorage.setItem('migrominder_onboarding_complete', 'true');
    }

    loadSavedData() {
        // Load from localStorage if exists
        const saved = localStorage.getItem('migrominder_user_data');
        if (saved) {
            try {
                this.userData = { ...this.userData, ...JSON.parse(saved) };
            } catch (e) {
                console.error('Error loading saved data:', e);
            }
        }
    }

    completeOnboarding() {
        // Save data
        this.saveData();

        // Send to backend (if available)
        this.sendToBackend();

        // Redirect to main app
        window.location.href = 'index.html';
    }

    async sendToBackend() {
        // Send user data to backend API
        try {
            if (window.API && window.API.logUserProfile) {
                await window.API.logUserProfile(this.userData);
                console.log('User data sent to backend');
            } else {
                // Fallback: just log
                console.log('User data (saved locally):', this.userData);
            }
        } catch (error) {
            console.error('Error sending data to backend:', error);
            // Still continue even if backend fails
            console.log('User data saved to localStorage');
        }
    }
}

// Initialize onboarding flow
const onboarding = new OnboardingFlow();

// Global functions for button onclick handlers
function nextStep() {
    onboarding.nextStep();
}

function prevStep() {
    onboarding.prevStep();
}

function selectOption(key, value) {
    onboarding.selectOption(key, value);
}

function toggleOption(key, value, element) {
    // element is passed from onclick handler
    // This function is kept for backward compatibility but event delegation is preferred
    onboarding.toggleOption(key, value, element);
}

function completeOnboarding() {
    onboarding.completeOnboarding();
}

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === 'Enter') {
        if (onboarding.currentStep < onboarding.totalSteps) {
            nextStep();
        }
    } else if (e.key === 'ArrowLeft') {
        if (onboarding.currentStep > 1) {
            prevStep();
        }
    }
});

