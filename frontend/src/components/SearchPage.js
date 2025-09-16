import React, { useState, useEffect } from 'react';
import { authService } from '../services/auth';

const SearchPage = ({ onLogout, onNavigateToProfile, onNavigateToChat, onNavigateToFinetunedChat }) => {
  const [cities, setCities] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [searchData, setSearchData] = useState({
    source: '',
    destination: ''
  });
  const [searchResults, setSearchResults] = useState(null);
  const [searching, setSearching] = useState(false);
  const [includeTraffic, setIncludeTraffic] = useState(false);
  const [allTravelOptions, setAllTravelOptions] = useState([]);
  const [sustainablePlaces, setSustainablePlaces] = useState([]);
  const [allPlaces, setAllPlaces] = useState([]);

  useEffect(() => {
    fetchCities();
  }, []);

  const fetchCities = async () => {
    try {
      const response = await authService.getCities();
      setCities(response.cities);
    } catch (err) {
      setError(err.error || 'Failed to load cities');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setSearchData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear error when user starts typing
    if (error) setError('');
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    
    if (!searchData.source || !searchData.destination) {
      setError('Please select both source and destination');
      return;
    }

    if (searchData.source === searchData.destination) {
      setError('Source and destination cannot be the same');
      return;
    }

    setSearching(true);
    setError('');

    try {
      // Get distance first using Google Maps API (for road-based modes)
      const distanceResult = await authService.calculateDistance(
        searchData.source, 
        searchData.destination, 
        'driving', // Use driving mode to get the base distance
        includeTraffic
      );

      const distanceKm = distanceResult.data.distance.value / 1000;
      
      // Calculate CO₂ emissions for specified travel modes
      const travelModes = [
        { id: 'flight', name: 'Flight (average)', category: 'Air Travel' },
        { id: 'diesel_car', name: 'Diesel Car', category: 'Road Transport' },
        { id: 'petrol_car', name: 'Petrol Car', category: 'Road Transport' },
        { id: 'electric_car', name: 'Electric Car', category: 'Road Transport' },
        { id: 'train_diesel', name: 'Train (diesel)', category: 'Public Transport' },
        { id: 'bus_shared', name: 'Bus (shared)', category: 'Public Transport' },
        { id: 'train_electric', name: 'Train (electric)', category: 'Public Transport' },
        { id: 'bicycle', name: 'Bicycle', category: 'Active Transport' },
        { id: 'walking', name: 'Walking', category: 'Active Transport' }
      ];

      const travelOptions = [];
      
      for (const mode of travelModes) {
        try {
          const co2Result = await authService.calculateCO2Emissions(distanceKm, mode.id);
          
          // Calculate duration for all travel modes
          let duration = null;
          if (mode.id === 'bicycle') {
            // Average cycling speed: 15 km/h
            const hours = distanceKm / 15;
            duration = formatDuration(hours);
          } else if (mode.id === 'walking') {
            // Average walking speed: 5 km/h
            const hours = distanceKm / 5;
            duration = formatDuration(hours);
          } else if (mode.id === 'flight') {
            // Flight duration: includes check-in, boarding, flight time, and baggage claim
            // Base flight time: 500 km/h average speed + 2 hours for airport procedures
            const flightHours = (distanceKm / 500) + 2;
            duration = formatDuration(flightHours);
          } else if (mode.id === 'diesel_car' || mode.id === 'petrol_car' || mode.id === 'electric_car') {
            // Car travel: use Google Maps duration if available, otherwise estimate
            if (distanceResult.data.duration) {
              duration = distanceResult.data.duration.text;
            } else {
              // Estimate: 60 km/h average speed including stops
              const hours = distanceKm / 60;
              duration = formatDuration(hours);
            }
          } else if (mode.id === 'train_diesel' || mode.id === 'train_electric') {
            // Train travel: 80 km/h average speed including stops
            const hours = distanceKm / 80;
            duration = formatDuration(hours);
          } else if (mode.id === 'bus_shared') {
            // Bus travel: 50 km/h average speed including stops
            const hours = distanceKm / 50;
            duration = formatDuration(hours);
          }
          
          travelOptions.push({
            ...mode,
            distance: distanceKm,
            co2Emissions: co2Result.data.totalEmissions,
            emissionFactor: co2Result.data.emissionFactor,
            treesNeeded: co2Result.data.equivalentMetrics.treesNeeded,
            duration: duration
          });
        } catch (err) {
          console.error(`Error calculating CO₂ for ${mode.id}:`, err);
        }
      }

      // Sort by CO₂ emissions (lowest first)
      travelOptions.sort((a, b) => a.co2Emissions - b.co2Emissions);

      setAllTravelOptions(travelOptions);
      
      // Fetch all places for the destination
      try {
        const placesResponse = await authService.getAllPlaces(searchData.destination);
        setAllPlaces(placesResponse.places || []);
        setSustainablePlaces(placesResponse.sustainable_places || []);
      } catch (placesErr) {
        console.warn('Could not fetch places:', placesErr);
        setAllPlaces([]);
        setSustainablePlaces([]);
      }
      
      setSearchResults({
        source: searchData.source,
        destination: searchData.destination,
        origin: distanceResult.data.origin,
        destination: distanceResult.data.destination,
        distance: distanceResult.data.distance,
        duration: distanceResult.data.duration,
        durationInTraffic: distanceResult.data.durationInTraffic,
        timestamp: new Date().toLocaleString()
      });
    } catch (err) {
      setError(err.error || err.message || 'Failed to calculate distances. Please try again.');
    } finally {
      setSearching(false);
    }
  };

  const handleLogout = () => {
    authService.logout();
    onLogout();
  };

  const formatDuration = (hours) => {
    if (hours < 1) {
      const minutes = Math.round(hours * 60);
      return `${minutes} mins`;
    } else if (hours < 24) {
      const wholeHours = Math.floor(hours);
      const minutes = Math.round((hours - wholeHours) * 60);
      if (minutes === 0) {
        return `${wholeHours} hrs`;
      }
      return `${wholeHours} hrs ${minutes} mins`;
    } else {
      const days = Math.floor(hours / 24);
      const remainingHours = Math.floor(hours % 24);
      if (remainingHours === 0) {
        return `${days} days`;
      }
      return `${days} days ${remainingHours} hrs`;
    }
  };


  if (loading) {
    return (
      <div className="container">
        <div className="search-page">
          <h2>Loading cities...</h2>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <div className="search-page">
        <div className="search-header">
          <h2>EarthRoutes</h2>
          <div className="header-actions">
            <button className="search-btn" disabled>
              Home
            </button>
            <button onClick={onNavigateToChat} className="chat-btn">
              Chat Assistant
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

        <form onSubmit={handleSearch} className="search-form">
          <div className="form-group">
            <label htmlFor="source">Source City:</label>
            <select
              id="source"
              name="source"
              value={searchData.source}
              onChange={handleInputChange}
              required
              disabled={searching}
            >
              <option value="">Select source city</option>
              {cities.map((city, index) => (
                <option key={index} value={city}>
                  {city}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="destination">Destination City:</label>
            <select
              id="destination"
              name="destination"
              value={searchData.destination}
              onChange={handleInputChange}
              required
              disabled={searching}
            >
              <option value="">Select destination city</option>
              {cities.map((city, index) => (
                <option key={index} value={city}>
                  {city}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group checkbox-group">
            <label>
              <input
                type="checkbox"
                checked={includeTraffic}
                onChange={(e) => setIncludeTraffic(e.target.checked)}
                disabled={searching}
              />
              Include current traffic conditions (for road transport)
            </label>
          </div>

          {error && <div className="error">{error}</div>}

          <button type="submit" disabled={searching} className="search-btn">
            {searching ? 'Searching...' : 'Search'}
          </button>
        </form>

        {searchResults && (
          <div className="search-results">
            <h3>Travel Comparison</h3>
            <div className="result-card">
              <div className="route-info">
                <p><strong>From:</strong> {searchResults.origin}</p>
                <p><strong>To:</strong> {searchResults.destination}</p>
                <p><strong>Distance:</strong> {searchResults.distance.text}</p>
                {searchResults.durationInTraffic && (
                  <p><strong>Driving Duration (with traffic):</strong> {searchResults.durationInTraffic.text}</p>
                )}
              </div>
              
              <div className="travel-comparison-table">
                <h4>🌱 Environmental Impact Comparison</h4>
                <table className="comparison-table">
                  <thead>
                    <tr>
                      <th>Travel Mode</th>
                      <th>Category</th>
                      <th>CO₂ Emissions</th>
                      <th>Emission Factor</th>
                      <th>Trees Needed</th>
                      <th>Duration</th>
                    </tr>
                  </thead>
                  <tbody>
                    {allTravelOptions.map((option, index) => (
                      <tr key={option.id} className={index < 3 ? 'eco-friendly' : ''}>
                        <td className="mode-name">{option.name}</td>
                        <td className="category">{option.category}</td>
                        <td className="emissions">
                          <span className="emission-value">{option.co2Emissions.toFixed(3)} kg</span>
                        </td>
                        <td className="factor">{option.emissionFactor} kg/km</td>
                        <td className="trees">{option.treesNeeded}</td>
                        <td className="duration">
                          {option.duration ? option.duration : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              
              <div className="search-meta">
                <p><strong>Calculated at:</strong> {searchResults.timestamp}</p>
                <p><em>Sorted by CO₂ emissions (lowest to highest)</em></p>
              </div>
            </div>
            
            {allPlaces.length > 0 && (
              <div className="all-places">
                <h4>📍 Places to Visit in {searchResults.destination}</h4>
                <div className="places-summary">
                  <span className="summary-item sustainable-count">
                    🌿 {sustainablePlaces.length} Eco-Friendly
                  </span>
                  <span className="summary-item total-count">
                    📍 {allPlaces.length} Total Places
                  </span>
                </div>
                <div className="places-list">
                  {allPlaces.slice(0, 8).map((place, index) => (
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
                {allPlaces.length > 8 && (
                  <div className="places-footer">
                    <p className="more-places-note">
                      Showing top 8 places by rating. {allPlaces.length - 8} more places available.
                    </p>
                  </div>
                )}
              </div>
            )}
            <button 
              onClick={() => {
                setSearchResults(null);
                setAllTravelOptions([]);
                setSustainablePlaces([]);
                setAllPlaces([]);
              }} 
              className="clear-btn"
            >
              New Search
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default SearchPage;
