// 🎬 App.jsx - Movie Recommender avec Questionnaire Complet
// À placer dans: C:\...\movie-app\src\App.jsx

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:5000/api';

export default function App() {
  const [step, setStep] = useState('welcome'); // welcome, questions, loading, results
  const [formData, setFormData] = useState({
    // Questions principales
    favorite_genre: '',
    min_rating: 7.0,
    era: 'mixed',
    
    // Questions additionnelles
    favorite_actor: '',
    favorite_director: '',
    movie_type: 'any', // blockbuster, hidden_gem, any
    mood: 'any' // dark, lighthearted, intense, romantic
  });

  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [genres, setGenres] = useState([]);

  // Charger les genres au démarrage
  useEffect(() => {
    axios.get(`${API_URL}/genres`)
      .then(res => setGenres(res.data.genres))
      .catch(err => console.error('Erreur chargement genres:', err));
  }, []);

  // Gérer changement formulaire
  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }));
  };

  // Soumettre le questionnaire
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setStep('loading');

    try {
      // Préparer les données pour l'API
      const apiData = {
        min_rating: formData.min_rating,
        era: formData.era,
        niche: formData.movie_type === 'hidden_gem',
        favorite_director: formData.favorite_director,
        genre_preference: formData.favorite_genre,
        favorite_actor: formData.favorite_actor
      };

      const response = await axios.post(
        `${API_URL}/recommendations`,
        apiData
      );

      setRecommendations(response.data.recommendations);
      setStep('results');
    } catch (err) {
      setError('Erreur lors de la récupération des recommandations');
      console.error(err);
      setStep('questions');
    } finally {
      setLoading(false);
    }
  };

  // Reset
  const handleReset = () => {
    setStep('welcome');
    setFormData({
      favorite_genre: '',
      min_rating: 7.0,
      era: 'mixed',
      favorite_actor: '',
      favorite_director: '',
      movie_type: 'any',
      mood: 'any'
    });
    setRecommendations([]);
  };

  return (
    <div className="app">
      {/* HEADER */}
      <header className="header">
        <h1>🎬 MovieMatch</h1>
        <p>Trouvez votre film parfait en 2 minutes</p>
      </header>

      <main className="container">
        {/* ÉTAPE 1: WELCOME */}
        {step === 'welcome' && (
          <section className="welcome-section">
            <div className="welcome-card">
              <h2>Bienvenue à MovieMatch! 🎉</h2>
              <p>Répondez à quelques questions sur vos préférences et nous vous recommanderons des films que vous allez adorer.</p>
              
              <div className="feature-list">
                <div className="feature">
                  <span>🎯</span>
                  <p>Recommandations personnalisées</p>
                </div>
                <div className="feature">
                  <span>⭐</span>
                  <p>Films de qualité garantie</p>
                </div>
                <div className="feature">
                  <span>🎭</span>
                  <p>Tous les genres couverts</p>
                </div>
              </div>

              <button 
                className="btn btn-primary btn-large"
                onClick={() => setStep('questions')}
              >
                Commencer le Questionnaire ➜
              </button>
            </div>
          </section>
        )}

        {/* ÉTAPE 2: QUESTIONS */}
        {step === 'questions' && (
          <section className="questions-section">
            <div className="questions-card">
              <h2>📋 Parlez-moi de vous</h2>
              <p className="subtitle">Vos réponses nous aideront à trouver le film parfait</p>

              <form onSubmit={handleSubmit}>
                {/* Q1: Genre favori */}
                <div className="form-group">
                  <label htmlFor="genre">🎭 Genre préféré?</label>
                  <select
                    id="genre"
                    name="favorite_genre"
                    value={formData.favorite_genre}
                    onChange={handleChange}
                    required
                    className="select"
                  >
                    <option value="">-- Sélectionnez un genre --</option>
                    {genres.map(g => (
                      <option key={g} value={g}>{g}</option>
                    ))}
                  </select>
                </div>

                {/* Q2: Rating minimum */}
                <div className="form-group">
                  <label htmlFor="rating">⭐ Note IMDb minimum: <strong>{formData.min_rating.toFixed(1)}</strong></label>
                  <input
                    id="rating"
                    type="range"
                    name="min_rating"
                    min="5"
                    max="10"
                    step="0.5"
                    value={formData.min_rating}
                    onChange={handleChange}
                    className="slider"
                  />
                  <p className="help-text">Plus haut = meilleurs films selon IMDb</p>
                </div>

                {/* Q3: Époque */}
                <div className="form-group">
                  <label>📅 Époque préférée?</label>
                  <div className="radio-group">
                    {[
                      { value: 'vintage', label: '🎞️ Classiques (avant 2000)' },
                      { value: 'modern', label: '🎬 Modernes (2000+)' },
                      { value: 'mixed', label: '🌀 Sans préférence' }
                    ].map(option => (
                      <label key={option.value}>
                        <input
                          type="radio"
                          name="era"
                          value={option.value}
                          checked={formData.era === option.value}
                          onChange={handleChange}
                        />
                        {option.label}
                      </label>
                    ))}
                  </div>
                </div>

                {/* Q4: Type de film */}
                <div className="form-group">
                  <label>🎯 Type de film?</label>
                  <div className="radio-group">
                    {[
                      { value: 'blockbuster', label: '🍿 Blockbuster populaire' },
                      { value: 'hidden_gem', label: '💎 Film caché mais excellent' },
                      { value: 'any', label: '🌀 Pas d\'importance' }
                    ].map(option => (
                      <label key={option.value}>
                        <input
                          type="radio"
                          name="movie_type"
                          value={option.value}
                          checked={formData.movie_type === option.value}
                          onChange={handleChange}
                        />
                        {option.label}
                      </label>
                    ))}
                  </div>
                </div>

                {/* Q5: Acteur favori */}
                <div className="form-group">
                  <label htmlFor="actor">👤 Acteur favori? (optionnel)</label>
                  <input
                    id="actor"
                    type="text"
                    name="favorite_actor"
                    placeholder="Ex: Leonardo DiCaprio"
                    value={formData.favorite_actor}
                    onChange={handleChange}
                    className="input"
                  />
                  <p className="help-text">Laissez vide si vous n'avez pas de préférence</p>
                </div>

                {/* Q6: Réalisateur favori */}
                <div className="form-group">
                  <label htmlFor="director">🎥 Réalisateur favori? (optionnel)</label>
                  <input
                    id="director"
                    type="text"
                    name="favorite_director"
                    placeholder="Ex: Christopher Nolan"
                    value={formData.favorite_director}
                    onChange={handleChange}
                    className="input"
                  />
                </div>

                {/* BOUTONS */}
                <div className="form-buttons">
                  <button type="submit" className="btn btn-primary">
                    🎯 Obtenir mes recommandations
                  </button>
                </div>
              </form>
            </div>
          </section>
        )}

        {/* ÉTAPE 3: LOADING */}
        {step === 'loading' && (
          <section className="loading-section">
            <div className="spinner"></div>
            <p>⏳ Recherche du film parfait pour vous...</p>
          </section>
        )}

        {/* ÉTAPE 4: RÉSULTATS */}
        {step === 'results' && (
          <section className="results-section">
            <div className="results-header">
              <h2>🎬 Voici vos recommandations!</h2>
              <button onClick={handleReset} className="btn btn-secondary">
                ← Recommencer
              </button>
            </div>

            {recommendations.length === 0 ? (
              <div className="no-results">
                <p>❌ Aucun film trouvé avec ces critères.</p>
                <p>Essayez une note minimum plus basse ou un autre genre.</p>
              </div>
            ) : (
              <div className="movies-grid">
                {recommendations.map((movie, index) => (
                  <div key={index} className="movie-card">
                    <div className="movie-rank">#{index + 1}</div>
                    
                    <h3>{movie.title}</h3>
                    
                    <div className="movie-info">
                      <div className="info-item">
                        <span>⭐</span>
                        <strong>{movie.rating}/10</strong>
                      </div>
                      <div className="info-item">
                        <span>👥</span>
                        <strong>{(movie.votes/1000000).toFixed(1)}M</strong>
                      </div>
                      <div className="info-item">
                        <span>📅</span>
                        <strong>{movie.year}</strong>
                      </div>
                    </div>

                    <div className="genre-badge">
                      {movie.genre}
                    </div>

                    <p className="director">🎥 {movie.director}</p>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        {/* MESSAGE D'ERREUR */}
        {error && (
          <div className="error-message">
            ❌ {error}
            <button onClick={() => setError(null)} className="close">×</button>
          </div>
        )}
      </main>

      {/* FOOTER */}
      <footer className="footer">
        <p>🎬 MovieMatch v1.0 | Recommandations basées sur Naive Bayes</p>
      </footer>
    </div>
  );
}