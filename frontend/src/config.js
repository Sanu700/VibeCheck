/**
 * Centralized API Configuration for VibeCheck Frontend.
 * Reads base URL from import.meta.env.VITE_API_URL or VITE_BACKEND_URL.
 */

const rawApiUrl = import.meta.env.VITE_API_URL || import.meta.env.VITE_BACKEND_URL || '';
export const API_BASE_URL = rawApiUrl.replace(/\/$/, '');

/**
 * Helper to construct full API endpoint URL.
 * Example: getApiUrl('/api/health') -> 'https://toxicbuddy.onrender.com/api/health'
 */
export const getApiUrl = (path) => {
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${cleanPath}`;
};
