import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ChiptunifyConverter(nn.Module):
    """
    Modelo para convertir características de voz en notas musicales
    con estilo chiptune.
    """
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=128):
        super().__init__()
        
        # Codificador para procesar la representación latente de la voz
        self.processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Mapeador de notas MIDI a representación sonora
        self.note_mapper = nn.Embedding(128, output_dim)  # 128 notas MIDI
        
    def forward(self, voice_encoding, midi_notes):
        """
        Args:
            voice_encoding: Tensor con codificación de la voz [batch, frames, input_dim]
            midi_notes: Tensor con las notas MIDI [batch, frames]
        """
        # Procesar la codificación de voz
        voice_features = self.processor(voice_encoding)
        
        # Obtener características de las notas MIDI
        note_features = self.note_mapper(midi_notes)
        
        # Combinar características de voz con notas MIDI
        output = voice_features * note_features
        
        return output
        
    def train_step(self, voice_encoding, midi_notes, target_audio, optimizer):
        """Paso de entrenamiento simple"""
        optimizer.zero_grad()
        
        output = self.forward(voice_encoding, midi_notes)
        
        # Pérdida simplificada (en una implementación real sería más compleja)
        loss = F.mse_loss(output, target_audio)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
        
    def convert(self, voice_encoding, midi_notes):
        """Convierte una codificación de voz a audio cantado según notas MIDI"""
        with torch.no_grad():
            return self.forward(voice_encoding, midi_notes)
