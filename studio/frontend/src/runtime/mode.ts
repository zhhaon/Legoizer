export const localMode = import.meta.env.VITE_STUDIO_MODE === 'local';
export const apiBase = (import.meta.env.VITE_STUDIO_API_URL || '/api').replace(/\/$/, '');
