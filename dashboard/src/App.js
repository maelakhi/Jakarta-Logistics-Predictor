import React, { useState, useEffect } from 'react';
import { Truck, DollarSign, MapPin, Calendar, Droplets } from 'lucide-react';

// --- Styling Constants (Pure JavaScript Objects) ---
const styles = {
  container: {
    padding: '2rem',
    backgroundColor: '#f8f8f8',
    minHeight: '100vh',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    fontFamily: 'Inter, sans-serif',
  },
  card: {
    backgroundColor: '#ffffff',
    borderRadius: '1rem',
    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    padding: '2.5rem',
    maxWidth: '500px',
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
  },
  title: {
    fontSize: '1.75rem',
    fontWeight: 700,
    color: '#1e3a8a',
    borderBottom: '2px solid #e0e7ff',
    paddingBottom: '0.75rem',
    marginBottom: '0.5rem',
  },
  inputGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.3rem',
  },
  label: {
    fontSize: '0.875rem',
    fontWeight: 500,
    color: '#374151',
  },
  inputWrapper: {
    display: 'flex',
    alignItems: 'center',
    border: '1px solid #d1d5db',
    borderRadius: '0.5rem',
    padding: '0.5rem 0.75rem',
    backgroundColor: '#fff',
    transition: 'border-color 0.2s',
  },
  input: {
    flexGrow: 1,
    border: 'none',
    outline: 'none',
    fontSize: '1rem',
    color: '#1f2937',
    padding: 0,
    marginLeft: '0.5rem',
  },
  button: {
    padding: '0.75rem 1.5rem',
    borderRadius: '0.5rem',
    backgroundColor: '#4f46e5', // Indigo 600
    color: 'white',
    fontSize: '1rem',
    fontWeight: 600,
    border: 'none',
    cursor: 'pointer',
    transition: 'background-color 0.2s, transform 0.1s',
  },
  outputBox: {
    marginTop: '1.5rem',
    padding: '1.25rem',
    backgroundColor: '#ecfdf5', // Green 50
    borderRadius: '0.75rem',
    border: '1px solid #a7f3d0', // Green 200
    textAlign: 'center',
  },
  outputLabel: {
    fontSize: '0.875rem',
    fontWeight: 500,
    color: '#047857', // Green 700
    marginBottom: '0.25rem',
  },
  outputPrice: {
    fontSize: '2rem',
    fontWeight: 800,
    color: '#065f46', // Green 800
  },
  error: {
    color: '#ef4444',
    textAlign: 'center',
    fontSize: '0.875rem',
    marginTop: '0.5rem',
    fontWeight: 500,
  },
  distanceText: {
    fontSize: '0.8rem',
    color: '#6b7280',
    marginTop: '0.5rem',
  },
  loading: {
    textAlign: 'center',
    color: '#4f46e5',
    fontWeight: 500,
  }
};

const CENTER_LAT = -6.2088;
const CENTER_LON = 106.8456;
const API_URL = 'https://logistics-api-9ybs.onrender.com/predict';

// Helper function to calculate Euclidean distance (in degrees)
const calculateDistance = (lat1, lon1) => {
  if (isNaN(lat1) || isNaN(lon1)) return NaN;
  return Math.sqrt(
    Math.pow(lat1 - CENTER_LAT, 2) + Math.pow(lon1 - CENTER_LON, 2)
  );
};

// Helper function to parse Lat, Lon string input
const parseCoords = (coordsString) => {
  const parts = coordsString.split(',').map(s => s.trim());
  if (parts.length === 2) {
    const lat = parseFloat(parts[0]);
    const lon = parseFloat(parts[1]);
    if (!isNaN(lat) && !isNaN(lon)) {
      return { lat, lon };
    }
  }
  return { lat: NaN, lon: NaN }; 
};

// Helper function to format currency
const formatPrice = (price) => {
  return new Intl.NumberFormat('id-ID', {
    style: 'currency',
    currency: 'IDR',
    minimumFractionDigits: 0
  }).format(price);
};

// Main App Component
const App = () => {
  const [formData, setFormData] = useState({
    // Keep a valid location example to calculate initial distance on load
    location_coords: "-6.1754, 106.8272", 
    delivery_date: new Date().toISOString().substring(0, 10),
    // Set to empty string for visually empty field on load
    demand_volume: "", 
    // Set to empty string for visually empty field on load
    fuel_price_factor: "" 
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [distance, setDistance] = useState(0);
  const [coordsValid, setCoordsValid] = useState(true);

  // Calculate distance whenever the coordinate string changes
  useEffect(() => {
    const { lat, lon } = parseCoords(formData.location_coords);
    if (!isNaN(lat) && !isNaN(lon)) {
      const dist = calculateDistance(lat, lon);
      setDistance(dist.toFixed(4));
      setCoordsValid(true);
      setError(null);
    } else {
      setDistance('Invalid format');
      setCoordsValid(false);
    }
  }, [formData.location_coords]);

  // Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    // Store all inputs as raw strings. Numbers will be parsed at submission.
    setFormData(prev => ({
      ...prev,
      [name]: value,
    }));
  };

  // Handle form submission and API call
  const handleSubmit = async (e) => {
    e.preventDefault();
    setPrediction(null);
    setError(null);
    setLoading(true);

    const { lat, lon } = parseCoords(formData.location_coords);
    
    // Parse numerical inputs just before submission
    const demand = parseFloat(formData.demand_volume);
    const fuel = parseFloat(formData.fuel_price_factor);
    
    // Check all required inputs for validity
    if (!coordsValid || isNaN(lat) || isNaN(lon) || isNaN(demand) || isNaN(fuel)) {
        setError('Please fix the location format and ensure all numerical fields are filled and valid.');
        setLoading(false);
        return;
    }

    // Prepare payload for the Flask API
    const payload = {
      pickup_latitude: lat,
      pickup_longitude: lon,
      delivery_date: formData.delivery_date,
      demand_volume: demand, // Use parsed number
      fuel_price_factor: fuel, // Use parsed number
    };

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`API returned status ${response.status}. Is the Python server running on port 8080?`);
      }

      const data = await response.json();

      if (data.predicted_price) {
        setPrediction(data.predicted_price);
      } else {
        setError("Prediction failed. Check API response structure.");
      }

    } catch (err) {
      console.error("API Error:", err);
      setError(`Failed to connect to the prediction API. Ensure 'python src/api.py' is running on port 8080.`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <form onSubmit={handleSubmit} style={styles.card}>
        <div style={styles.title}>
          JABODETABEK Logistics Cost Predictor
        </div>

        {/* Pickup Location (Combined Lat/Lon) */}
        <div style={styles.inputGroup}>
          <label 
            style={{...styles.label, color: coordsValid ? '#374151' : '#ef4444'}} 
            htmlFor="location_coords"
          >
            <MapPin size={14} style={{ display: 'inline', marginRight: '5px' }} /> 
            Pickup Location (Latitude, Longitude)
          </label>
          <div style={{...styles.inputWrapper, borderColor: coordsValid ? '#d1d5db' : '#ef4444'}}>
            <input
              id="location_coords"
              name="location_coords"
              type="text"
              placeholder="-6.1754, 106.8272"
              value={formData.location_coords}
              onChange={handleChange}
              required
              style={styles.input}
            />
          </div>
          <div style={{...styles.distanceText, color: coordsValid ? '#6b7280' : '#ef4444'}}>
            Distance from Center (degrees): {distance}
          </div>
          {!coordsValid && (
              <div style={styles.error}>
                  Format must be: -6.1754, 106.8272
              </div>
          )}
        </div>
        
        {/* Delivery Date */}
        <div style={styles.inputGroup}>
          <label style={styles.label} htmlFor="delivery_date">
            <Calendar size={14} style={{ display: 'inline', marginRight: '5px' }} /> Delivery Date (Used to determine Weekend Surcharge)
          </label>
          <div style={styles.inputWrapper}>
            <input
              id="delivery_date"
              name="delivery_date"
              type="date"
              value={formData.delivery_date}
              onChange={handleChange}
              required
              style={styles.input}
            />
          </div>
        </div>

        {/* Demand Volume */}
        <div style={styles.inputGroup}>
          <label style={styles.label} htmlFor="demand_volume">
            <Truck size={14} style={{ display: 'inline', marginRight: '5px' }} /> Demand Volume (1-10 scale)
          </label>
          <div style={styles.inputWrapper}>
            <input
              id="demand_volume"
              name="demand_volume"
              type="number"
              step="0.1"
              min="1"
              max="10"
              placeholder="e.g., 5.0"
              value={formData.demand_volume}
              onChange={handleChange}
              required
              style={styles.input}
            />
          </div>
        </div>

        {/* Fuel Price Factor */}
        <div style={styles.inputGroup}>
          <label style={styles.label} htmlFor="fuel_price_factor">
            <Droplets size={14} style={{ display: 'inline', marginRight: '5px' }} /> Fuel Price Factor (Base 1.0)
          </label>
          <div style={styles.inputWrapper}>
            <input
              id="fuel_price_factor"
              name="fuel_price_factor"
              type="number"
              step="0.01"
              min="0.5"
              max="2.0"
              placeholder="e.g., 1.0"
              value={formData.fuel_price_factor}
              onChange={handleChange}
              required
              style={styles.input}
            />
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          style={{
            ...styles.button,
            ...(loading || !coordsValid ? { opacity: 0.7, cursor: 'not-allowed' } : {}),
          }}
          disabled={loading || !coordsValid}
        >
          {loading ? 'Predicting...' : 'Predict Final Price'}
        </button>

        {/* Prediction Output / Error */}
        {(prediction !== null || error) && (
          <div style={{...styles.outputBox, backgroundColor: error ? '#fee2e2' : '#ecfdf5', border: error ? '1px solid #f87171' : '1px solid #a7f3d0'}}>
            {error ? (
              <p style={{...styles.error, color: '#dc2626'}}>
                Error: {error}
              </p>
            ) : (
              <>
                <p style={styles.outputLabel}>Predicted Final Price (IDR)</p>
                <p style={styles.outputPrice}>
                  {formatPrice(prediction)}
                </p>
              </>
            )}
          </div>
        )}
      </form>
    </div>
  );
};

export default App;
