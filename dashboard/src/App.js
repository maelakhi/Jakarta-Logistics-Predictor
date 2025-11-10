import React, { useState } from 'react';
import { Truck, MapPin, Calendar, Droplets, Send, Target } from 'lucide-react';

// --- Styling Constants (Pure JavaScript Objects) ---
const styles = {
  container: {
    padding: '1rem',
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
    maxWidth: '550px',
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
  locationGroup: {
    display: 'flex',
    gap: '1rem',
    flexWrap: 'wrap',
  },
  locationInput: {
    flex: '1 1 calc(50% - 0.5rem)', // Make them stack nicely on mobile
    minWidth: '200px',
  },
  label: {
    fontSize: '0.875rem',
    fontWeight: 500,
    color: '#374151',
    display: 'flex',
    alignItems: 'center',
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
    width: '100%', // Ensure input uses full width of its wrapper
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
  loading: {
    textAlign: 'center',
    color: '#4f46e5',
    fontWeight: 500,
  }
};

// **PENTING: Ganti dengan URL Render API Anda!**
const API_URL = 'https://logistics-api-9ybs.onrender.com/predict';
//const API_URL = 'http://127.0.0.1:8080/predict';
// Helper function to parse Lat, Lon string input
const parseCoords = (coordsString) => {
  const parts = coordsString.split(',').map(s => s.trim());
  if (parts.length === 2) {
    const lat = parseFloat(parts[0]);
    const lon = parseFloat(parts[1]);
    if (!isNaN(lat) && !isNaN(lon) && lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180) {
      return { lat, lon, valid: true };
    }
  }
  return { lat: NaN, lon: NaN, valid: false }; 
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
    // Initial values for 4 coordinates
    pickup_coords: "-6.1754, 106.8272", 
    dropoff_coords: "-6.2088, 106.8456", 
    delivery_date: new Date().toISOString().substring(0, 10),
    demand_volume: "5.0", 
    fuel_price_factor: "1.0",
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [validation, setValidation] = useState({
    pickup_valid: true,
    dropoff_valid: true,
    numerical_valid: true
  });

  // Calculate overall validity (Checks coordinates validity immediately)
  const isFormValid = validation.pickup_valid && validation.dropoff_valid && validation.numerical_valid;

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value,
    }));
    
    // Immediate validation for coordinates
    if (name === 'pickup_coords' || name === 'dropoff_coords') {
        const { valid } = parseCoords(value);
        setValidation(prev => ({
            ...prev,
            [name === 'pickup_coords' ? 'pickup_valid' : 'dropoff_valid']: valid
        }));
    }
  };

  // Handle form submission and API call
  const handleSubmit = async (e) => {
    e.preventDefault();
    setPrediction(null);
    setError(null);
    setLoading(true);

    const pickup = parseCoords(formData.pickup_coords);
    const dropoff = parseCoords(formData.dropoff_coords);
    
    // Numerical validation check
    const demand = parseFloat(formData.demand_volume);
    const fuel = parseFloat(formData.fuel_price_factor);
    const numValid = !isNaN(demand) && !isNaN(fuel);

    setValidation({
        pickup_valid: pickup.valid,
        dropoff_valid: dropoff.valid,
        numerical_valid: numValid
    });

    if (!pickup.valid || !dropoff.valid || !numValid) {
        setError('Mohon perbaiki format lokasi (Lat, Lon) dan pastikan semua angka terisi.');
        setLoading(false);
        return;
    }

    // Prepare payload for the API (must match your Flask/Render API structure)
    const payload = {
      pickup_latitude: pickup.lat,
      pickup_longitude: pickup.lon,
      dropoff_latitude: dropoff.lat, // NEW
      dropoff_longitude: dropoff.lon, // NEW
      delivery_date: formData.delivery_date,
      demand_volume: demand, 
      fuel_price_factor: fuel, 
    };
    
    console.log("Sending payload:", payload);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            // Render/Vercel often requires CORS headers for cross-origin requests
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API returned status ${response.status}. Response: ${errorText.substring(0, 100)}...`);
      }

      const data = await response.json();

      if (data.predicted_price !== undefined) {
        setPrediction(data.predicted_price);
      } else {
        setError("Prediksi gagal. Respon dari API tidak mengandung 'predicted_price'.");
      }

    } catch (err) {
      console.error("API Error:", err);
      setError(`Gagal terhubung ke API prediksi. Pastikan API di Render sudah di-deploy dan berjalan. Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <form onSubmit={handleSubmit} style={styles.card}>
        <div style={styles.title}>
          <Truck size={24} style={{ display: 'inline', marginRight: '10px', verticalAlign: 'middle' }} />
          JABODETABEK Logistics Cost Predictor
        </div>

        <p style={{ fontSize: '0.9rem', color: '#6b7280', marginTop: '-0.5rem' }}>
            Masukkan lokasi, tanggal, dan faktor biaya untuk memprediksi harga pengiriman.
        </p>

        {/* Location Group */}
        <div style={styles.locationGroup}>
            {/* Pickup Location */}
            <div style={{...styles.inputGroup, ...styles.locationInput}}>
              <label 
                style={{...styles.label, color: validation.pickup_valid ? '#374151' : '#ef4444'}} 
                htmlFor="pickup_coords"
              >
                <MapPin size={14} style={{ marginRight: '5px' }} /> 
                Lokasi Jemput (Lat, Lon)
              </label>
              <div style={{...styles.inputWrapper, borderColor: validation.pickup_valid ? '#d1d5db' : '#ef4444'}}>
                <input
                  id="pickup_coords"
                  name="pickup_coords"
                  type="text"
                  placeholder="Contoh: -6.1754, 106.8272"
                  value={formData.pickup_coords}
                  onChange={handleChange}
                  required
                  style={styles.input}
                />
              </div>
            </div>

            {/* Dropoff Location */}
            <div style={{...styles.inputGroup, ...styles.locationInput}}>
              <label 
                style={{...styles.label, color: validation.dropoff_valid ? '#374151' : '#ef4444'}} 
                htmlFor="dropoff_coords"
              >
                <Target size={14} style={{ marginRight: '5px' }} /> 
                Lokasi Antar (Lat, Lon)
              </label>
              <div style={{...styles.inputWrapper, borderColor: validation.dropoff_valid ? '#d1d5db' : '#ef4444'}}>
                <input
                  id="dropoff_coords"
                  name="dropoff_coords"
                  type="text"
                  placeholder="Contoh: -6.2088, 106.8456"
                  value={formData.dropoff_coords}
                  onChange={handleChange}
                  required
                  style={styles.input}
                />
              </div>
            </div>
        </div>

        {/* Non-Location Inputs */}
        
        {/* Delivery Date (Used for Traffic/Weekend) */}
        <div style={styles.inputGroup}>
          <label style={styles.label} htmlFor="delivery_date">
            <Calendar size={14} style={{ marginRight: '5px' }} /> Waktu Pengiriman (Menentukan Traffic & Weekend)
          </label>
          <div style={styles.inputWrapper}>
            {/* Using type datetime-local to capture both date and time */}
            <input
              id="delivery_date"
              name="delivery_date"
              type="datetime-local"
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
            <Truck size={14} style={{ marginRight: '5px' }} /> Demand Volume (Skala 1 - 10, mencerminkan seberapa sibuk/penting pengiriman ini)
          </label>
          <div style={styles.inputWrapper}>
            <input
              id="demand_volume"
              name="demand_volume"
              type="number"
              step="0.1"
              min="1"
              max="10"
              placeholder="Contoh: 5.0"
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
            <Droplets size={14} style={{ marginRight: '5px' }} /> Fuel Price Factor (Basis 1.0, mencerminkan kenaikan harga bahan bakar)
          </label>
          <div style={styles.inputWrapper}>
            <input
              id="fuel_price_factor"
              name="fuel_price_factor"
              type="number"
              step="0.01"
              min="0.5"
              max="2.0"
              placeholder="Contoh: 1.0"
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
            ...(loading || !isFormValid ? { opacity: 0.7, cursor: 'not-allowed' } : {}),
          }}
          disabled={loading || !isFormValid}
        >
          <Send size={16} style={{ display: 'inline', marginRight: '8px', verticalAlign: 'text-bottom' }} />
          {loading ? 'Memprediksi Biaya...' : 'Prediksi Harga Akhir'}
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
                <p style={styles.outputLabel}>Harga Akhir Terprediksi (IDR)</p>
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