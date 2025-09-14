import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests if available
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Handle token expiration
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 || error.response?.status === 403) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/';
    }
    return Promise.reject(error);
  }
);

export const authService = {
  // Register new user
  register: async (userData) => {
    try {
      const response = await api.post('/api/register', userData);
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Registration failed' };
    }
  },

  // Login user
  login: async (credentials) => {
    try {
      const response = await api.post('/api/login', credentials);
      const { token, user } = response.data;
      
      // Store token and user data
      localStorage.setItem('token', token);
      localStorage.setItem('user', JSON.stringify(user));
      
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Login failed' };
    }
  },

  // Get user profile
  getProfile: async () => {
    try {
      const response = await api.get('/api/profile');
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to get profile' };
    }
  },

  // Logout user
  logout: () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  },

  // Check if user is authenticated
  isAuthenticated: () => {
    return !!localStorage.getItem('token');
  },

  // Get current user from localStorage
  getCurrentUser: () => {
    const user = localStorage.getItem('user');
    return user ? JSON.parse(user) : null;
  },

  // Get cities for search
  getCities: async () => {
    try {
      const response = await api.get('/api/cities');
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to get cities' };
    }
  },

  // Get all cities (hotels and places separately)
  getAllCities: async () => {
    try {
      const response = await api.get('/api/cities/all');
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to get all cities' };
    }
  },

  // Get sustainable places in a city
  getSustainablePlaces: async (city) => {
    try {
      const response = await api.get(`/api/sustainable-places/${city}`);
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to get sustainable places' };
    }
  },

  // Get all places in a city
  getAllPlaces: async (city) => {
    try {
      const response = await api.get(`/api/places/${city}`);
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to get places' };
    }
  },

  // Calculate distance between two cities
  calculateDistance: async (source, destination, mode = 'driving', includeTraffic = false) => {
    try {
      const response = await api.post('/api/distance', {
        source,
        destination,
        mode,
        includeTraffic
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to calculate distance' };
    }
  },

  // Calculate multiple distances (batch processing)
  calculateMultipleDistances: async (origins, destinations, mode = 'driving') => {
    try {
      const response = await api.post('/api/distance/batch', {
        origins,
        destinations,
        mode
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to calculate distances' };
    }
  },

  // CO₂ Emission Services

  // Calculate CO₂ emissions for a specific distance and travel mode
  calculateCO2Emissions: async (distanceKm, travelMode, options = {}) => {
    try {
      const response = await api.post('/api/co2/calculate', {
        distanceKm,
        travelMode,
        options
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to calculate CO₂ emissions' };
    }
  },

  // Compare CO₂ emissions between different travel modes
  compareCO2Emissions: async (distanceKm, travelModes) => {
    try {
      const response = await api.post('/api/co2/compare', {
        distanceKm,
        travelModes
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to compare CO₂ emissions' };
    }
  },

  // Calculate CO₂ savings by switching travel modes
  calculateCO2Savings: async (distanceKm, fromMode, toMode) => {
    try {
      const response = await api.post('/api/co2/savings', {
        distanceKm,
        fromMode,
        toMode
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to calculate CO₂ savings' };
    }
  },

  // Get all available travel modes and emission factors
  getCO2EmissionFactors: async () => {
    try {
      const response = await api.get('/api/co2/modes');
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to get emission factors' };
    }
  },

  // Chat Services

  // Send message to travel assistant
  sendChatMessage: async (message, sessionId = null) => {
    try {
      const response = await api.post('/api/chat', {
        message,
        session_id: sessionId
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to send message' };
    }
  },

  // Send message to fine-tuned model
  sendFinetunedChatMessage: async (message, sessionId = null) => {
    try {
      const response = await api.post('/api/chat/finetuned', {
        message,
        session_id: sessionId
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to send message to fine-tuned model' };
    }
  },

  // Get fine-tuned model status
  getFinetunedModelStatus: async () => {
    try {
      const response = await api.get('/api/chat/finetuned/status');
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to get model status' };
    }
  },

  // Chat History Services

  // Get all chat sessions for the current user
  getChatSessions: async () => {
    try {
      const response = await api.get('/api/chat/sessions');
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to get chat sessions' };
    }
  },

  // Get messages for a specific chat session
  getChatSessionMessages: async (sessionId) => {
    try {
      const response = await api.get(`/api/chat/sessions/${sessionId}/messages`);
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to get chat messages' };
    }
  },

  // Delete a chat session
  deleteChatSession: async (sessionId) => {
    try {
      const response = await api.delete(`/api/chat/sessions/${sessionId}`);
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to delete chat session' };
    }
  },

  // Update chat session title
  updateChatSessionTitle: async (sessionId, newTitle) => {
    try {
      const response = await api.put(`/api/chat/sessions/${sessionId}/title`, {
        title: newTitle
      });
      return response.data;
    } catch (error) {
      throw error.response?.data || { error: 'Failed to update chat session title' };
    }
  }
};

export default api;
