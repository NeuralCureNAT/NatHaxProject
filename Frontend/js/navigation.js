/**
 * Navigation Module
 * Handles navigation menu interactions and mobile menu toggle
 */

class Navigation {
    constructor() {
        this.menuToggle = document.querySelector('.menu-toggle');
        this.navMenu = document.querySelector('.nav-menu');
        this.navLinks = document.querySelectorAll('.nav-menu a');
        this.init();
    }

    init() {
        this.setupMenuToggle();
        this.setupSmoothScroll();
        this.setupActiveLink();
        this.setupScrollEffect();
    }

    setupMenuToggle() {
        if (this.menuToggle) {
            this.menuToggle.addEventListener('click', () => {
                this.toggleMobileMenu();
            });
        }

        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.navbar') && this.navMenu.classList.contains('active')) {
                this.closeMobileMenu();
            }
        });

        // Close menu when clicking a link
        this.navLinks.forEach(link => {
            link.addEventListener('click', () => {
                if (window.innerWidth <= 768) {
                    this.closeMobileMenu();
                }
            });
        });
    }

    toggleMobileMenu() {
        this.menuToggle?.classList.toggle('active');
        this.navMenu?.classList.toggle('active');
    }

    closeMobileMenu() {
        this.menuToggle?.classList.remove('active');
        this.navMenu?.classList.remove('active');
    }

    setupSmoothScroll() {
        this.navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                const href = link.getAttribute('href');
                
                if (href.startsWith('#')) {
                    e.preventDefault();
                    const targetId = href.substring(1);
                    const targetElement = document.getElementById(targetId);
                    
                    if (targetElement) {
                        const headerOffset = 80;
                        const elementPosition = targetElement.getBoundingClientRect().top;
                        const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

                        window.scrollTo({
                            top: offsetPosition,
                            behavior: 'smooth'
                        });
                    }
                }
            });
        });
    }

    setupActiveLink() {
        const sections = document.querySelectorAll('section[id]');
        
        const observerOptions = {
            threshold: 0.3,
            rootMargin: '-80px 0px -80% 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.getAttribute('id');
                    this.navLinks.forEach(link => {
                        link.classList.remove('active');
                        if (link.getAttribute('href') === `#${id}`) {
                            link.classList.add('active');
                        }
                    });
                }
            });
        }, observerOptions);

        sections.forEach(section => observer.observe(section));
    }

    setupScrollEffect() {
        const header = document.querySelector('header');
        let lastScroll = 0;

        window.addEventListener('scroll', () => {
            const currentScroll = window.pageYOffset;

            if (currentScroll > 100) {
                header.style.boxShadow = '0 4px 12px rgba(26, 31, 58, 0.15)';
            } else {
                header.style.boxShadow = '0 1px 3px rgba(26, 31, 58, 0.1)';
            }

            lastScroll = currentScroll;
        });
    }
}

// Initialize navigation when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new Navigation();
});

