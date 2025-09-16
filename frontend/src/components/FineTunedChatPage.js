import React, { useState, useRef, useEffect } from 'react';
import { authService } from '../services/auth';
import ChatHistorySidebar from './ChatHistorySidebar';

const FineTunedChatPage = ({ onLogout, onNavigateToProfile, onNavigateToSearch, onNavigateToChat }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content: 'Hello! I\'m your fine-tuned travel sustainability assistant. I\'ve been specifically trained on sustainable travel data to provide you with expert advice on eco-friendly travel options, CO₂ emissions, and sustainable tourism. How can I help you plan your sustainable journey today?',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [modelStatus, setModelStatus] = useState(null);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [historyRefreshTrigger, setHistoryRefreshTrigger] = useState(0);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
    // Check model status on component mount
    checkModelStatus();
  }, [messages]);

  const checkModelStatus = async () => {
    try {
      const status = await authService.getFinetunedModelStatus();
      setModelStatus(status.data);
    } catch (err) {
      console.error('Failed to get model status:', err);
      setModelStatus({ is_loaded: false, error: 'Failed to check model status' });
    }
  };

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
      const response = await authService.sendFinetunedChatMessage(inputMessage.trim(), currentSessionId);
      
      // Update current session ID if this is a new session
      if (response.session_id && !currentSessionId) {
        setCurrentSessionId(response.session_id);
      }
      
      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: response.response,
        timestamp: new Date(),
        isError: !response.success
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
        content: 'Hello! I\'m your fine-tuned travel sustainability assistant. I\'ve been specifically trained on sustainable travel data to provide you with expert advice on eco-friendly travel options, CO₂ emissions, and sustainable tourism. How can I help you plan your sustainable journey today?',
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

  return (
    <div className="container">
      <div className="chat-page">
        <div className="chat-header">
          <div className="header-left">
            <div className="chat-title">
              <h2>Fine-Tuned Travel Assistant</h2>
              <div className="model-status">
                {modelStatus && !modelStatus.is_loaded && (
                  <span className="status-indicator not-loaded">
                    ⚠ Model Not Available
                  </span>
                )}
              </div>
            </div>
          </div>
          <div className="header-actions">
            <button onClick={onNavigateToSearch} className="search-btn">
              Home
            </button>
            <button onClick={onNavigateToChat} className="chat-btn">
              Chat Assistant
            </button>
            <button className="finetuned-chat-btn" disabled>
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
              placeholder="Ask me about sustainable travel options, CO₂ emissions, eco-friendly destinations..."
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
          <h4>Try asking about:</h4>
          <div className="suggestion-chips">
            <button 
              className="suggestion-chip"
              onClick={() => setInputMessage("What are the most sustainable ways to travel from Mumbai to Delhi?")}
              disabled={isLoading}
            >
              Sustainable Mumbai to Delhi
            </button>
            <button 
              className="suggestion-chip"
              onClick={() => setInputMessage("Compare CO₂ emissions for different transport modes to Bangalore")}
              disabled={isLoading}
            >
              CO₂ comparison to Bangalore
            </button>
            <button 
              className="suggestion-chip"
              onClick={() => setInputMessage("Suggest eco-friendly places to visit in Kerala")}
              disabled={isLoading}
            >
              Eco-friendly Kerala destinations
            </button>
            <button 
              className="suggestion-chip"
              onClick={() => setInputMessage("How can I offset my travel carbon footprint?")}
              disabled={isLoading}
            >
              Carbon offset strategies
            </button>
            <button 
              className="suggestion-chip"
              onClick={() => setInputMessage("What are green accommodation options in Goa?")}
              disabled={isLoading}
            >
              Green hotels in Goa
            </button>
            <button 
              className="suggestion-chip"
              onClick={() => setInputMessage("Plan a sustainable weekend trip from Pune")}
              disabled={isLoading}
            >
              Sustainable weekend from Pune
            </button>
          </div>
        </div>
      </div>
      
      <ChatHistorySidebar
        onLoadSession={handleLoadSession}
        onNewChat={handleNewChat}
        currentSessionId={currentSessionId}
        chatType="finetuned"
        refreshTrigger={historyRefreshTrigger}
      />
    </div>
  );
};

export default FineTunedChatPage;
