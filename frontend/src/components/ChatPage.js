import React, { useState, useRef, useEffect } from 'react';
import { authService } from '../services/auth';
import ChatHistorySidebar from './ChatHistorySidebar';

const ChatPage = ({ onLogout, onNavigateToProfile, onNavigateToSearch, onNavigateToFinetunedChat }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content: 'Hello! I\'m your travel planning assistant. I can help you find the best travel options between cities, calculate CO₂ emissions, and suggest eco-friendly alternatives. How can I help you today?',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [historyRefreshTrigger, setHistoryRefreshTrigger] = useState(0);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleInputChange = (e) => {
    setInputMessage(e.target.value);
    if (error) setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!inputMessage.trim()) {
      setError('Please enter a message');
      return;
    }

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    setError('');

    try {
      const response = await authService.sendChatMessage(inputMessage.trim(), currentSessionId);
      
      // Update current session ID if this is a new session
      if (response.session_id && !currentSessionId) {
        setCurrentSessionId(response.session_id);
      }
      
      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: response.response,
        timestamp: new Date(),
        travelData: response.travel_data || null
      };

      setMessages(prev => [...prev, botMessage]);
      
      // Trigger history refresh to show the new session/message
      setHistoryRefreshTrigger(prev => prev + 1);
    } catch (err) {
      console.error('Chat error:', err);
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: 'Sorry, I encountered an error while processing your request. Please try again.',
        timestamp: new Date(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
      setError(err.error || err.message || 'Failed to send message');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleLogout = () => {
    authService.logout();
    onLogout();
  };

  const handleLoadSession = (sessionMessages, sessionId) => {
    // Convert session messages to the format expected by the component
    const formattedMessages = sessionMessages.map(msg => ({
      id: msg.id,
      type: msg.message_type,
      content: msg.content,
      timestamp: new Date(msg.created_at),
      travelData: msg.travel_data || null,
      isError: msg.is_error || false
    }));
    
    setMessages(formattedMessages);
    setCurrentSessionId(sessionId);
  };

  const handleNewChat = () => {
    setMessages([
      {
        id: 1,
        type: 'bot',
        content: 'Hello! I\'m your travel planning assistant. I can help you find the best travel options between cities, calculate CO₂ emissions, and suggest eco-friendly alternatives. How can I help you today?',
        timestamp: new Date()
      }
    ]);
    setCurrentSessionId(null);
  };

  const formatMessage = (content) => {
    // Simple formatting for better readability
    return content.split('\n').map((line, index) => (
      <span key={index}>
        {line}
        {index < content.split('\n').length - 1 && <br />}
      </span>
    ));
  };

  const renderTravelData = (travelData) => {
    if (!travelData) return null;

    return (
      <div className="travel-data-card">
        <h4>🌱 Travel Analysis</h4>
        <div className="travel-info">
          <p><strong>Route:</strong> {travelData.source} → {travelData.destination}</p>
          <p><strong>Distance:</strong> {travelData.distance}</p>
        </div>
        
        {travelData.options && travelData.options.length > 0 && (
          <div className="travel-options">
            <h5>Travel Options (sorted by CO₂ emissions):</h5>
            <div className="options-list">
              {travelData.options.slice(0, 5).map((option, index) => (
                <div key={index} className={`option-item ${index < 3 ? 'eco-friendly' : ''}`}>
                  <div className="option-header">
                    <span className="mode-name">{option.name}</span>
                    <span className="emissions">{option.co2Emissions.toFixed(3)} kg CO₂</span>
                  </div>
                  <div className="option-details">
                    <span className="category">{option.category}</span>
                    <span className="duration">{option.duration}</span>
                    <span className="trees">{option.treesNeeded} trees needed</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {travelData.places && travelData.places.length > 0 && (
          <div className="all-places">
            <h5>📍 Places to Visit in {travelData.destination}</h5>
            <div className="places-summary">
              <span className="summary-item sustainable-count">
                🌿 {travelData.sustainable_places ? travelData.sustainable_places.length : 0} Eco-Friendly
              </span>
              <span className="summary-item total-count">
                📍 {travelData.places.length} Total Places
              </span>
            </div>
            <div className="places-list">
              {travelData.places.slice(0, 8).map((place, index) => (
                <div key={index} className={`place-item ${(place.is_sustainable === 1 || place.is_sustainable === true) ? 'sustainable' : 'regular'}`}>
                  <div className="place-header">
                    <span className="place-name">{place.name}</span>
                    <div className="place-badges">
                      <span className="place-rating">
                        ⭐ {place.google_review_rating || 'N/A'}
                        {place.google_reviews_lakhs && place.google_reviews_lakhs > 0 && (
                          <span className="review-count"> ({place.google_reviews_lakhs}L reviews)</span>
                        )}
                      </span>
                      {(place.is_sustainable === 1 || place.is_sustainable === true) && <span className="eco-badge">🌿 Eco-Friendly</span>}
                    </div>
                  </div>
                  <div className="place-details">
                    <span className="place-type">{place.type || 'N/A'}</span>
                    {place.establishment_year && (
                      <span className="establishment-year">Est. {place.establishment_year}</span>
                    )}
                    {(place.is_sustainable === 1 || place.is_sustainable === true) && place.sustainability_reason && (
                      <span className="sustainability-reason">{place.sustainability_reason}</span>
                    )}
                  </div>
                  <div className="place-additional-info">
                    {place.time_needed_hrs && place.time_needed_hrs > 0 && (
                      <div className="info-item">
                        <span className="info-label">⏱️ Time Needed:</span>
                        <span className="info-value">{place.time_needed_hrs} hours</span>
                      </div>
                    )}
                    {place.entrance_fee_inr !== null && place.entrance_fee_inr !== undefined && (
                      <div className="info-item">
                        <span className="info-label">💰 Entrance Fee:</span>
                        <span className="info-value">
                          {place.entrance_fee_inr === 0 ? 'Free' : `₹${place.entrance_fee_inr}`}
                        </span>
                      </div>
                    )}
                    {place.significance && (
                      <div className="info-item">
                        <span className="info-label">🏛️ Significance:</span>
                        <span className="info-value">{place.significance}</span>
                      </div>
                    )}
                    {place.best_time_to_visit && (
                      <div className="info-item">
                        <span className="info-label">🌅 Best Time:</span>
                        <span className="info-value">{place.best_time_to_visit}</span>
                      </div>
                    )}
                    {place.weekly_off && (
                      <div className="info-item">
                        <span className="info-label">📅 Weekly Off:</span>
                        <span className="info-value">{place.weekly_off}</span>
                      </div>
                    )}
                    {place.dslr_allowed && (
                      <div className="info-item">
                        <span className="info-label">📸 Photography:</span>
                        <span className="info-value">{place.dslr_allowed === 'Yes' ? 'Allowed' : 'Not Allowed'}</span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
            {travelData.places.length > 8 && (
              <div className="places-footer">
                <p className="more-places-note">
                  Showing top 8 places by rating. {travelData.places.length - 8} more places available.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="container">
      <div className="chat-page">
        <div className="chat-header">
          <div className="header-left">
            <h2>Travel Assistant Chat</h2>
          </div>
          <div className="header-actions">
            <button onClick={onNavigateToSearch} className="search-btn">
              Travel Search
            </button>
            <button onClick={onNavigateToFinetunedChat} className="finetuned-chat-btn">
              Fine-Tuned LLM
            </button>
            <button onClick={onNavigateToProfile} className="profile-btn">
              Profile
            </button>
            <button onClick={handleLogout} className="logout-btn">
              Logout
            </button>
          </div>
        </div>

        <div className="chat-container">
          <div className="messages-container">
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.type} ${message.isError ? 'error' : ''}`}>
                <div className="message-content">
                  <div className="message-text">
                    {formatMessage(message.content)}
                  </div>
                  {message.travelData && renderTravelData(message.travelData)}
                  <div className="message-timestamp">
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="message bot">
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </div>

        <form onSubmit={handleSubmit} className="chat-input-form">
          {error && <div className="error">{error}</div>}
          
          <div className="input-container">
            <textarea
              value={inputMessage}
              onChange={handleInputChange}
              onKeyPress={handleKeyPress}
              placeholder="Ask me about travel options, CO₂ emissions, or eco-friendly alternatives..."
              disabled={isLoading}
              rows="2"
              className="chat-input"
            />
            <button 
              type="submit" 
              disabled={isLoading || !inputMessage.trim()} 
              className="send-btn"
            >
              {isLoading ? 'Sending...' : 'Send'}
            </button>
          </div>
        </form>

        <div className="chat-suggestions">
          <h4>Try asking:</h4>
          <div className="suggestion-chips">
            <button 
              className="suggestion-chip"
              onClick={() => setInputMessage("What's the best way to travel from Delhi to Mumbai?")}
              disabled={isLoading}
            >
              Best way from Delhi to Mumbai
            </button>
            <button 
              className="suggestion-chip"
              onClick={() => setInputMessage("Compare CO₂ emissions for different travel modes to Bangalore")}
              disabled={isLoading}
            >
              Compare CO₂ emissions to Bangalore
            </button>
            <button 
              className="suggestion-chip"
              onClick={() => setInputMessage("What are the most eco-friendly travel options?")}
              disabled={isLoading}
            >
              Most eco-friendly options
            </button>
            <button 
              className="suggestion-chip"
              onClick={() => setInputMessage("How many trees would I need to plant to offset my flight?")}
              disabled={isLoading}
            >
              Tree offset calculation
            </button>
          </div>
        </div>
      </div>
      
      <ChatHistorySidebar
        onLoadSession={handleLoadSession}
        onNewChat={handleNewChat}
        currentSessionId={currentSessionId}
        chatType="regular"
        refreshTrigger={historyRefreshTrigger}
      />
    </div>
  );
};

export default ChatPage;
