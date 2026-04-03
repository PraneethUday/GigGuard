// GigGuard Design System — Dark Navy / Indigo / Amber
// ui branch — new design

export const COLORS = {
  // Core palette
  primary: '#6C63FF',           // Indigo-Violet (CTAs, active states)
  primaryDim: '#8B84FF',        // Lighter indigo for highlights
  primaryContainer: '#1D1B45',  // Dark indigo background for cards

  amber: '#FF8C42',             // Electric Amber (alerts, urgency, highlights)
  amberDim: '#FFB380',          // Soft amber
  amberContainer: '#2A1A0A',    // Dark amber card bg

  success: '#22C55E',           // Active/paid/green states
  successContainer: '#0A2E18',  // Dark green card bg
  error: '#EF4444',             // Rejected/danger
  errorContainer: '#2E0A0A',    // Dark red card bg
  info: '#3B82F6',              // Info banners

  // Surfaces (dark layered system)
  background: '#0C0E18',        // Deepest base (surface_container_lowest)
  surface: '#11131E',           // Main page background
  surfaceLow: '#191B26',        // Section backgrounds
  surfaceContainer: '#1D1F2B',  // Card backgrounds
  surfaceHigh: '#272935',       // Elevated cards
  surfaceHighest: '#323440',    // Interactive / input backgrounds

  // Text
  white: '#E1E1F2',             // Primary text (on dark)
  textMuted: '#C7C4D8',         // Secondary text
  textFaint: '#918FA1',         // Placeholder / disabled text

  // Borders (ghost borders — low opacity)
  border: 'rgba(70, 69, 85, 0.6)',      // Default border
  borderActive: 'rgba(108, 99, 255, 0.5)', // Focus / active border

  // Legacy aliases (keeps old references from breaking during transition)
  secondary: '#1D1F2B',
  accent: '#FF8C42',
  text: '#C7C4D8',
  gray: '#464555',
  lightGray: '#191B26',
};

export const FONTS = {
  bold: 'Inter_700Bold',
  semiBold: 'Inter_600SemiBold',
  medium: 'Inter_500Medium',
  regular: 'Inter_400Regular',
};

export const SIZES = {
  base: 8,
  font: 14,
  radius: 12,
  radiusFull: 100,
  padding: 20,

  h1: 28,
  h2: 22,
  h3: 18,
  body: 15,
  small: 13,
  tiny: 11,
};

export const GRADIENTS = {
  primary: ['#6C63FF', '#8B84FF'],
  amber: ['#FF8C42', '#FFB380'],
  surface: ['#1D1F2B', '#272935'],
};

export const SHADOWS = {
  card: {
    shadowColor: '#6C63FF',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.08,
    shadowRadius: 24,
    elevation: 4,
  },
  button: {
    shadowColor: '#6C63FF',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.35,
    shadowRadius: 16,
    elevation: 8,
  },
};

export const THEME = {
  COLORS,
  FONTS,
  SIZES,
  GRADIENTS,
  SHADOWS,
};

export default THEME;
