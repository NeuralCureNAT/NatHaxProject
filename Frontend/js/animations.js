/**
 * Animations Module
 * Handles scroll animations, fade-ins, and interactive animations
 * Enhanced with Three.js-style 3D effects and smooth transitions
 */

class Animations {
    constructor() {
        this.observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        this.mouseX = 0;
        this.mouseY = 0;
        this.init();
    }

    init() {
        this.setupScrollAnimations();
        this.setupCardAnimations();
        this.setupHoverEffects();
        this.setupParallax();
        this.setup3DEffects();
        this.setupMouseTracking();
        this.setupParticleEffects();
    }

    setupMouseTracking() {
        document.addEventListener('mousemove', (e) => {
            this.mouseX = (e.clientX / window.innerWidth) * 2 - 1;
            this.mouseY = (e.clientY / window.innerHeight) * 2 - 1;
            this.update3DTransforms();
        });
    }

    setup3DEffects() {
        const cards = document.querySelectorAll('.card, .feature-card, .dashboard-card, .contributor-card');
        
        cards.forEach(card => {
            card.style.transition = 'transform 0.3s cubic-bezier(0.23, 1, 0.32, 1), box-shadow 0.3s ease';
            
            card.addEventListener('mouseenter', () => {
                card.style.willChange = 'transform';
            });
            
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateZ(0)';
                card.style.willChange = 'auto';
            });
        });
    }

    update3DTransforms() {
        const cards = document.querySelectorAll('.card:hover, .feature-card:hover, .dashboard-card:hover');
        
        cards.forEach(card => {
            const rect = card.getBoundingClientRect();
            const cardCenterX = rect.left + rect.width / 2;
            const cardCenterY = rect.top + rect.height / 2;
            
            const deltaX = (this.mouseX * window.innerWidth / 2 - cardCenterX) / window.innerWidth;
            const deltaY = (this.mouseY * window.innerHeight / 2 - cardCenterY) / window.innerHeight;
            
            const rotateX = deltaY * 5;
            const rotateY = -deltaX * 5;
            const translateZ = Math.abs(deltaX) * 10 + Math.abs(deltaY) * 10;
            
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(${translateZ}px)`;
        });
    }

    setupParticleEffects() {
        // Add floating particles to hero section
        const hero = document.querySelector('.hero');
        if (!hero) return;

        // Create particle container
        const particleContainer = document.createElement('div');
        particleContainer.className = 'particles-container';
        particleContainer.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            overflow: hidden;
            z-index: 0;
        `;
        hero.appendChild(particleContainer);

        // Create particles
        for (let i = 0; i < 20; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.cssText = `
                position: absolute;
                width: ${Math.random() * 4 + 2}px;
                height: ${Math.random() * 4 + 2}px;
                background: var(--accent-yellow);
                border-radius: 50%;
                opacity: ${Math.random() * 0.5 + 0.2};
                left: ${Math.random() * 100}%;
                top: ${Math.random() * 100}%;
                animation: float ${Math.random() * 10 + 10}s infinite ease-in-out;
                animation-delay: ${Math.random() * 5}s;
            `;
            particleContainer.appendChild(particle);
        }

        // Add CSS animation
        if (!document.getElementById('particle-animations')) {
            const style = document.createElement('style');
            style.id = 'particle-animations';
            style.textContent = `
                @keyframes float {
                    0%, 100% {
                        transform: translate(0, 0) scale(1);
                        opacity: 0.3;
                    }
                    50% {
                        transform: translate(${Math.random() * 100 - 50}px, ${Math.random() * 100 - 50}px) scale(1.5);
                        opacity: 0.6;
                    }
                }
            `;
            document.head.appendChild(style);
        }
    }

    setupScrollAnimations() {
        const animatedElements = document.querySelectorAll(
            '.card, .step, .feature-card, .dashboard-card, .graph-card, .metric-item'
        );

        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry, index) => {
                if (entry.isIntersecting) {
                    setTimeout(() => {
                        entry.target.style.opacity = '1';
                        entry.target.style.transform = 'translateY(0) rotateX(0deg)';
                        entry.target.style.filter = 'blur(0px)';
                    }, index * 50);
                    observer.unobserve(entry.target);
                }
            });
        }, this.observerOptions);

        animatedElements.forEach((el, index) => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(40px) rotateX(10deg)';
            el.style.filter = 'blur(5px)';
            el.style.transition = 'all 0.8s cubic-bezier(0.23, 1, 0.32, 1)';
            el.style.transitionDelay = `${index * 0.05}s`;
            observer.observe(el);
        });
    }

    setupCardAnimations() {
        const cards = document.querySelectorAll('.card, .feature-card, .dashboard-card');
        
        cards.forEach((card, index) => {
            // Stagger animation on load
            card.style.opacity = '0';
            card.style.transform = 'translateY(30px) rotateX(10deg)';
            card.style.transition = 'all 0.6s cubic-bezier(0.23, 1, 0.32, 1)';
            
            setTimeout(() => {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0) rotateX(0deg)';
            }, index * 100);
            
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-12px) scale(1.03) rotateX(-2deg)';
                card.style.boxShadow = '0 20px 40px rgba(26, 31, 58, 0.2)';
            });

            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(0) scale(1) rotateX(0deg)';
                card.style.boxShadow = '';
            });
        });
    }

    setupHoverEffects() {
        const buttons = document.querySelectorAll('.btn, .cta-button');
        
        buttons.forEach(button => {
            button.style.transition = 'all 0.3s cubic-bezier(0.23, 1, 0.32, 1)';
            
            button.addEventListener('mouseenter', () => {
                button.style.transform = 'translateY(-4px) scale(1.05)';
                button.style.boxShadow = '0 10px 25px rgba(255, 193, 7, 0.3)';
            });

            button.addEventListener('mouseleave', () => {
                button.style.transform = 'translateY(0) scale(1)';
                button.style.boxShadow = '';
            });
        });
    }

    setupParallax() {
        const hero = document.querySelector('.hero');
        if (!hero) return;

        let ticking = false;
        
        window.addEventListener('scroll', () => {
            if (!ticking) {
                window.requestAnimationFrame(() => {
                    const scrolled = window.pageYOffset;
                    const heroContent = document.querySelector('.hero-content');
                    const particles = document.querySelectorAll('.particle');
                    
                    if (heroContent && scrolled < hero.offsetHeight) {
                        const parallaxSpeed = 0.5;
                        const opacity = Math.max(0.5, 1 - (scrolled / hero.offsetHeight) * 0.5);
                        
                        heroContent.style.transform = `translateY(${scrolled * parallaxSpeed}px)`;
                        heroContent.style.opacity = opacity;
                        
                        // Parallax particles
                        particles.forEach((particle, index) => {
                            const speed = (index % 3 + 1) * 0.2;
                            particle.style.transform = `translateY(${scrolled * speed}px)`;
                        });
                    }
                    
                    ticking = false;
                });
                ticking = true;
            }
        });
    }

    // Animate numbers counting up with easing
    animateNumber(element, target, duration = 2000) {
        const start = 0;
        const startTime = performance.now();
        const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = easeOutCubic(progress);
            const current = Math.floor(start + (target - start) * eased);
            
            element.textContent = current;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                element.textContent = target;
            }
        };
        
        requestAnimationFrame(animate);
    }

    // Animate stat values on scroll
    setupStatAnimations() {
        const statValues = document.querySelectorAll('.stat-value');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const value = entry.target.textContent;
                    const numericValue = parseInt(value.replace(/\D/g, ''));
                    
                    if (numericValue) {
                        entry.target.textContent = '0';
                        this.animateNumber(entry.target, numericValue);
                        observer.unobserve(entry.target);
                    }
                }
            });
        }, this.observerOptions);

        statValues.forEach(stat => observer.observe(stat));
    }
}

// Initialize animations when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const animations = new Animations();
    animations.setupStatAnimations();
});

