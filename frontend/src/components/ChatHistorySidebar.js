import React, { useState, useEffect } from 'react';
import { authService } from '../services/auth';

const ChatHistorySidebar = ({ 
  onLoadSession, 
  onNewChat,
  currentSessionId,
  chatType = 'regular',
  refreshTrigger = 0
}) => {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    loadChatSessions();
  }, [chatType, refreshTrigger]);

  const loadChatSessions = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await authService.getChatSessions();
      if (response.success) {
        // Filter sessions by chat type
        const filteredSessions = response.sessions.filter(
          session => session.chat_type === chatType
        );
        setSessions(filteredSessions);
      } else {
        setError('Failed to load chat sessions');
      }
    } catch (err) {
      setError(err.error || 'Failed to load chat sessions');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSession = async (sessionId) => {
    try {
      const response = await authService.getChatSessionMessages(sessionId);
      if (response.success) {
        onLoadSession(response.messages, sessionId);
      } else {
        setError('Failed to load chat session');
      }
    } catch (err) {
      setError(err.error || 'Failed to load chat session');
    }
  };

  const handleDeleteSession = async (sessionId, e) => {
    e.stopPropagation();
    if (window.confirm('Are you sure you want to delete this chat session?')) {
      try {
        const response = await authService.deleteChatSession(sessionId);
        if (response.success) {
          setSessions(sessions.filter(session => session.id !== sessionId));
          if (currentSessionId === sessionId) {
            onLoadSession([], null); // Clear current session
          }
        } else {
          setError('Failed to delete chat session');
        }
      } catch (err) {
        setError(err.error || 'Failed to delete chat session');
      }
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 1) {
      return 'Today';
    } else if (diffDays === 2) {
      return 'Yesterday';
    } else if (diffDays <= 7) {
      return `${diffDays - 1} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  return (
    <div className="chat-history-sidebar">
      <div className="sidebar-header">
        <div className="header-left">
          <button 
            className="new-chat-sidebar-btn"
            onClick={onNewChat}
            title="Start new chat"
          >
            ➕ New Chat
          </button>
          <h3>Chat History</h3>
        </div>
      </div>
      
      <div className="sidebar-content">
        {loading && (
          <div className="loading">Loading chat sessions...</div>
        )}
        
        {error && (
          <div className="error">{error}</div>
        )}
        
        {!loading && !error && sessions.length === 0 && (
          <div className="no-sessions">
            <p>No chat sessions found.</p>
            <p>Start a new conversation to see it here!</p>
          </div>
        )}
        
        {!loading && !error && sessions.length > 0 && (
          <div className="sessions-list">
            {sessions.map((session) => (
              <div
                key={session.id}
                className={`session-item ${currentSessionId === session.id ? 'active' : ''}`}
                onClick={() => handleLoadSession(session.id)}
              >
                <div className="session-content">
                  <div className="session-title">{session.title}</div>
                  <div className="session-meta">
                    <span className="message-count">{session.message_count} messages</span>
                    <span className="session-date">{formatDate(session.updated_at)}</span>
                  </div>
                </div>
                <button
                  className="delete-btn"
                  onClick={(e) => handleDeleteSession(session.id, e)}
                  title="Delete session"
                >
                  🗑️
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatHistorySidebar;
