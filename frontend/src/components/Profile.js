import React, { useState, useEffect } from 'react';
import { authService } from '../services/auth';

const Profile = ({ user, onLogout, onNavigateToSearch, onNavigateToChat, onNavigateToFinetunedChat }) => {
  const [profileData, setProfileData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchProfile();
  }, []);

  const fetchProfile = async () => {
    try {
      const response = await authService.getProfile();
      setProfileData(response.user);
    } catch (err) {
      setError(err.error || 'Failed to load profile');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    authService.logout();
    onLogout();
  };

  if (loading) {
    return (
      <div className="container">
        <div className="profile">
          <h2>Loading...</h2>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container">
        <div className="profile">
          <h2>Error</h2>
          <div className="error">{error}</div>
          <button onClick={handleLogout} className="logout-btn">
            Logout
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <div className="profile">
        <h2>Welcome, {user.username}!</h2>
        
        <div className="user-info">
          <h3>Profile Information</h3>
          <p><strong>User ID:</strong> {profileData?.id}</p>
          <p><strong>Username:</strong> {profileData?.username}</p>
          <p><strong>Email:</strong> {profileData?.email || 'Not provided'}</p>
          <p><strong>Member since:</strong> {new Date(profileData?.created_at).toLocaleDateString()}</p>
        </div>

        <div className="profile-actions">
          <button onClick={onNavigateToSearch} className="search-btn">
            Go to Travel Search
          </button>
          <button onClick={onNavigateToChat} className="chat-btn">
            Chat Assistant
          </button>
          <button onClick={onNavigateToFinetunedChat} className="finetuned-chat-btn">
            Fine-Tuned LLM
          </button>
          <button onClick={handleLogout} className="logout-btn">
            Logout
          </button>
        </div>
      </div>
    </div>
  );
};

export default Profile;
