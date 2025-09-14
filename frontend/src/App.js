import React, { useState, useEffect } from 'react';
import LoginForm from './components/LoginForm';
import RegisterForm from './components/RegisterForm';
import Profile from './components/Profile';
import SearchPage from './components/SearchPage';
import ChatPage from './components/ChatPage';
import { authService } from './services/auth';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [showRegister, setShowRegister] = useState(false);
  const [currentPage, setCurrentPage] = useState('search'); // 'search', 'chat', or 'profile'

  useEffect(() => {
    // Check if user is already logged in
    if (authService.isAuthenticated()) {
      const currentUser = authService.getCurrentUser();
      if (currentUser) {
        setUser(currentUser);
        setIsAuthenticated(true);
      }
    }
  }, []);

  const handleLogin = (userData) => {
    setUser(userData);
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    setUser(null);
    setIsAuthenticated(false);
    setCurrentPage('search');
  };

  const toggleForm = () => {
    setShowRegister(!showRegister);
  };

  const navigateToProfile = () => {
    setCurrentPage('profile');
  };

  const navigateToSearch = () => {
    setCurrentPage('search');
  };

  const navigateToChat = () => {
    setCurrentPage('chat');
  };

  if (isAuthenticated && user) {
    if (currentPage === 'profile') {
      return <Profile user={user} onLogout={handleLogout} onNavigateToSearch={navigateToSearch} onNavigateToChat={navigateToChat} />;
    } else if (currentPage === 'chat') {
      return <ChatPage onLogout={handleLogout} onNavigateToProfile={navigateToProfile} onNavigateToSearch={navigateToSearch} />;
    } else {
      return <SearchPage onLogout={handleLogout} onNavigateToProfile={navigateToProfile} onNavigateToChat={navigateToChat} />;
    }
  }

  return (
    <div className="App">
      {showRegister ? (
        <RegisterForm onRegister={handleLogin} onToggleForm={toggleForm} />
      ) : (
        <LoginForm onLogin={handleLogin} onToggleForm={toggleForm} />
      )}
    </div>
  );
}

export default App;
