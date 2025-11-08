# MigroMinder Frontend

## 📁 Project Structure

```
Frontend/
├── index.html              # Main HTML file
├── onboarding.html         # Onboarding questionnaire flow
├── css/                    # Stylesheets (modular)
│   ├── main.css           # Main stylesheet (imports all)
│   ├── variables.css      # CSS variables & theme
│   ├── reset.css          # CSS reset & base styles
│   ├── layout.css         # Layout utilities
│   ├── navigation.css     # Navigation styles
│   ├── hero.css           # Hero section styles
│   ├── components.css     # Reusable components
│   ├── dashboard.css      # Dashboard styles
│   ├── footer.css         # Footer styles
│   ├── onboarding.css     # Onboarding flow styles
│   └── responsive.css    # Media queries
├── js/                     # JavaScript modules
│   ├── main.js           # Main app entry point
│   ├── navigation.js     # Navigation functionality
│   ├── dashboard.js     # Dashboard interactions
│   ├── animations.js    # Scroll & animation effects
│   ├── onboarding.js    # Onboarding flow handler
│   └── api.js           # API communication (Flask backend)
└── README.md             # This file
```

## 🎨 Color Theme

Based on MigroMinder logo:
- **Light Beige**: `#F5F1E8` (background)
- **Dark Blue/Black**: `#1A1F3A` (primary text, accents)
- **Yellow/Gold**: `#FFC107` (accent color, highlights)

## 🚀 Getting Started

1. **Open `onboarding.html`** in a web browser (first-time users)
   - New users will be automatically redirected to onboarding
   - Complete the questionnaire to personalize your experience
2. **Main app** (`index.html`) loads after onboarding completion
3. All CSS files are imported through `css/main.css`
4. JavaScript files load in order (api.js → navigation.js → dashboard.js → animations.js → main.js)

## 📝 File Descriptions

### CSS Files

- **variables.css**: All CSS custom properties (colors, spacing, shadows, etc.)
- **reset.css**: CSS reset and base typography
- **layout.css**: Container, grid, and flex utilities
- **navigation.css**: Header and navigation menu styles
- **hero.css**: Hero section with gradient background
- **components.css**: Buttons, cards, badges, and reusable components
- **dashboard.css**: Dashboard cards, meters, and data displays
- **footer.css**: Footer styles
- **responsive.css**: Media queries for mobile/tablet
- **main.css**: Imports all CSS modules

### JavaScript Files

- **api.js**: Handles communication with Flask backend
  - `getEEGData()` - Fetch real-time EEG data
  - `getMigraineHistory()` - Get logged migraine events
  - `logMigraineEvent()` - Log new migraine episode
  - `getEnvironmentData()` - Get sensor data (Arduino)
  - `controlLight()` - Control Arduino LED module
  - `logUserProfile()` - Save user onboarding data
  - Includes mock data for development

- **onboarding.js**: Manages onboarding questionnaire flow
  - Multi-step question flow (7 steps)
  - Collects: frequency, severity, triggers, goals, hardware
  - Saves to localStorage and sends to backend
  - Progress bar and smooth transitions
  - Keyboard navigation support

- **navigation.js**: Navigation menu functionality
  - Mobile menu toggle
  - Smooth scrolling
  - Active link highlighting
  - Scroll effects on header

- **dashboard.js**: Dashboard interactions
  - Focus meter updates
  - Real-time data visualization
  - Card interactions
  - Simulated data updates (for demo)

- **animations.js**: Animation effects
  - Scroll-triggered animations
  - Card hover effects
  - Parallax effects
  - Number counting animations

- **main.js**: Main application logic
  - Initializes all modules
  - Handles session start/stop
  - Migraine pattern detection
  - Notification system
  - Real-time monitoring

## 🔧 Customization

### Changing Colors

Edit `css/variables.css`:
```css
:root {
    --primary-dark: #1A1F3A;    /* Change primary color */
    --accent-yellow: #FFC107;   /* Change accent color */
    --bg-beige: #F5F1E8;        /* Change background */
}
```

### Adding New Components

1. Add styles to appropriate CSS file (or create new one)
2. Import in `css/main.css` if new file
3. Add HTML in `index.html`

### Backend Integration

Update API base URL in `js/api.js`:
```javascript
this.baseURL = 'http://your-flask-backend:5000/api';
```

## 📱 Responsive Breakpoints

- **Desktop**: > 1024px
- **Tablet**: 768px - 1024px
- **Mobile**: < 768px
- **Small Mobile**: < 480px

## 🛠️ Development

### Browser Support
- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

### Testing
- Test with mock data (API not required)
- Check responsive design on different screen sizes
- Verify accessibility (keyboard navigation, screen readers)

## 📚 Team Collaboration

### For Frontend Developers
- Each CSS file is self-contained - edit the relevant file
- JavaScript modules are independent - modify one without affecting others
- Onboarding flow can be customized in `onboarding.html` and `onboarding.js`
- User data is stored in localStorage - can be cleared to retake onboarding
- Follow existing naming conventions
- Comment your code

### For Backend Developers
- API endpoints expected in `js/api.js`
- Mock data available for frontend-only development
- Update `api.js` when endpoints change

### For Designers
- Color variables in `css/variables.css`
- Component styles in `css/components.css`
- Easy to update theme colors

## 🐛 Troubleshooting

**Styles not loading?**
- Check that `css/main.css` imports all files
- Verify file paths are correct

**JavaScript errors?**
- Check browser console
- Ensure files load in correct order
- Verify API base URL if using backend

**Mobile menu not working?**
- Check `navigation.js` is loaded
- Verify menu toggle button exists

## 📄 License

Part of NatHax Project - EdTech Innovation

---

**Built with ❤️ for those who understand the storm**

