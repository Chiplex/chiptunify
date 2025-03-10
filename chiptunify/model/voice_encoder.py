import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

class LightweightVoiceEncoder(nn.Module):
    """
    Codificador de voz ligero que extrae características esenciales
    de la voz para su posterior conversión.
    """
    def __init__(self, input_dim=80, hidden_dim=128, latent_dim=64):  # Cambiado de 40 a 80
        super().__init__()
        
        # Arquitectura simplificada para equipos con recursos limitados
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)
        
    def extract_features(self, audio, sr=16000, n_mfcc=40):
        """Extrae características MFCC del audio"""
        # Verificar que el audio tenga suficiente duración
        min_samples = sr * 0.1  # Mínimo 100ms de audio
        if len(audio) < min_samples:
            raise ValueError(f"Audio demasiado corto: {len(audio)/sr:.2f}s. Se necesitan al menos 0.1s.")
            
        # Normalizar audio si es necesario
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
        # Extraer características básicas (MFCC)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # Normalizar y concatenar
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
        mfcc_delta = (mfcc_delta - np.mean(mfcc_delta, axis=1, keepdims=True)) / (np.std(mfcc_delta, axis=1, keepdims=True) + 1e-8)
        
        features = np.concatenate([mfcc, mfcc_delta], axis=0)
        
        # Verificar dimensiones
        if features.shape[0] != 80:  # Asegurar que tengamos 80 características (40 MFCC + 40 deltas)
            raise ValueError(f"Error en extracción de características: se esperaban 80 pero se obtuvieron {features.shape[0]}")
            
        return features.T  # (frames, features)

    def encode_audio(self, audio, sr=16000):
        """Convierte audio en una representación latente"""
        features = self.extract_features(audio, sr)
        features_tensor = torch.FloatTensor(features)
        
        # Dividir en frames si es necesario
        if features_tensor.size(0) > 100:
            # Tomar ventanas para evitar problemas de memoria
            embeddings = []
            for i in range(0, features_tensor.size(0), 50):
                batch = features_tensor[i:i+50]
                with torch.no_grad():
                    embedding = self.forward(batch)
                embeddings.append(embedding)
            return torch.cat(embeddings, dim=0)
        else:
            with torch.no_grad():
                return self.forward(features_tensor)
