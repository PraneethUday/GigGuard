// Shared design tokens for GigGuard — used by both mobile/ and web/

export const COLORS = {
  primary: '#A51C30',       // Ruby Red (CTAs / Highlights)
  secondary: '#A01E22',     // Deep Red (Header / Structural)
  accent: '#04052E',        // Prussian Blue (Deep Accents)
  background: '#FFFDFB',    // Ivory (Page Background)
  surface: '#FFFFFF',       // Pure White (Cards)
  text: '#1B1B3A',          // Deep Navy Text
  rubyRed: '#A51C30',
  deepRed: '#A01E22',
  prussianBlue: '#04052E',
  ivory: '#FFFDFB',
  white: '#FFFFFF',
  gray: '#E2E2E2',
  error: '#D32F2F',
  success: '#22C55E',
  info: '#3B82F6',
  lightGray: '#F9FAFB',
};

export const FONT_FAMILIES = {
  bold: "'EB Garamond', serif",
  medium: "'EB Garamond', serif",
  regular: "'EB Garamond', serif",
};

export const FONT_WEIGHTS = {
  bold: 700,
  medium: 500,
  regular: 400,
};

export const SIZES = {
  base: 8,
  font: 14,
  radius: 12,
  padding: 24,
  h1: 32,
  h2: 24,
  h3: 18,
  body: 16,
};

export default { COLORS, FONT_FAMILIES, FONT_WEIGHTS, SIZES };
