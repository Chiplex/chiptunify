"""
Script de demostración para Chiptunify.
Muestra el flujo completo: entrenamiento con una muestra pequeña y conversión.
"""

import os
import sys
import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt
import tempfile
import shutil

# Añadir el directorio principal al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chiptunify.model.voice_encoder import LightweightVoiceEncoder
from chiptunify.model.voice_converter import ChiptunifyConverter
from chiptunify.audio.effects import ChiptuneEffects
from chiptunify.midi.converter import MIDIConverter

def generate_demo_midi(output_path, duration=10, bpm=120):
    """Genera un archivo MIDI simple para demostración"""
    import mido
    
    print(f"Generando archivo MIDI de demostración en {output_path}...")
    
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Configurar tempo (120 BPM)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
    
    # Notas de una escala mayor (Do mayor)
    scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C4, D4, E4, F4, G4, A4, B4, C5
    
    # Duración de cada nota en beats
    note_duration = 0.5
    ticks_per_beat = mid.ticks_per_beat
    
    # Tiempo actual en ticks
    current_time = 0
    
    # Generar secuencia de notas
    for _ in range(int(duration / (note_duration * len(scale)))):
        for note in scale:
            # Note on
            track.append(mido.Message('note_on', note=note, velocity=64, time=int(current_time)))
            
            # Note off (después de note_duration)
            track.append(mido.Message('note_off', note=note, velocity=0, 
                                     time=int(ticks_per_beat * note_duration)))
            
            current_time = ticks_per_beat * note_duration
    
    # Guardar archivo MIDI
    mid.save(output_path)
    print(f"Archivo MIDI guardado en {output_path}")
    return output_path

def run_demo(input_audio=None):
    """Ejecuta una demostración completa de Chiptunify"""
    print("\n=== CHIPTUNIFY DEMO ===\n")
    
    # Crear directorios temporales
    temp_dir = tempfile.mkdtemp()
    models_dir = os.path.join(temp_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        # Si no se proporciona audio de entrada, usar ruido blanco
        if input_audio is None:
            print("No se proporcionó audio de entrada, generando audio de prueba...")
            sr = 16000
            duration = 5  # segundos
            audio = np.random.randn(sr * duration) * 0.1
            audio_file = os.path.join(temp_dir, "demo_input.wav")
            sf.write(audio_file, audio, sr)
        else:
            print(f"Usando archivo de audio proporcionado: {input_audio}")
            audio_file = input_audio
            audio, sr = librosa.load(audio_file, sr=16000, mono=True)
            
        # Generar archivo MIDI para demostración
        midi_file = os.path.join(temp_dir, "demo_melody.mid")
        generate_demo_midi(midi_file, duration=len(audio)/sr)
        
        # Entrenar modelo con una época (solo para demostración)
        print("\n1. Entrenando modelo de demostración (versión reducida)...")
        
        # Configurar modelo
        encoder = LightweightVoiceEncoder()
        converter = ChiptunifyConverter()
        
        # Extraer características
        features = encoder.extract_features(audio, sr)
        features_tensor = torch.FloatTensor(features)
        
        # Simular entrenamiento breve
        print("Ejecutando mini-entrenamiento...")
        optimizer = torch.optim.Adam(list(encoder.parameters()) + 
                                    list(converter.parameters()), lr=0.001)
        
        for epoch in tqdm(range(5)):  # Solo 5 iteraciones para demo
            batch_size = min(32, features_tensor.size(0))
            indices = np.random.choice(features_tensor.size(0), batch_size, replace=False)
            batch_features = features_tensor[indices]
            batch_notes = torch.randint(48, 84, (batch_size,)).long()
            
            target_audio = torch.randn(batch_size, 128)
            
            optimizer.zero_grad()
            
            # Forward
            encoded = encoder(batch_features)
            output = converter(encoded, batch_notes)
            
            # Loss simplificada
            loss = torch.nn.functional.mse_loss(output, target_audio)
            
            # Backward
            loss.backward()
            optimizer.step()
        
        # Guardar modelo de demostración
        model_path = os.path.join(models_dir, "demo_model.pth")
        checkpoint = {
            'encoder_state_dict': encoder.state_dict(),
            'converter_state_dict': converter.state_dict(),
            'epoch': 5
        }
        torch.save(checkpoint, model_path)
        print(f"Modelo de demostración guardado en {model_path}")
        
        # Convertir audio
        print("\n2. Convirtiendo audio con el modelo de demostración...")
        output_file = os.path.join(temp_dir, "demo_output.wav")
        
        # Preparar para conversión
        encoder.eval()
        converter.eval()
        
        # Procesar archivo MIDI
        midi_converter = MIDIConverter(sr=sr)
        duration = len(audio) / sr
        midi_notes = midi_converter.get_torch_tensor(midi_file, duration)
        
        # Convertir
        features = encoder.extract_features(audio, sr)
        features_tensor = torch.FloatTensor(features)
        
        batch_size = 50
        outputs = []
        
        for i in range(0, len(features_tensor), batch_size):
            end_idx = min(i + batch_size, len(features_tensor))
            batch_features = features_tensor[i:end_idx]
            batch_notes = midi_notes[i:end_idx]
            
            # Codificar
            with torch.no_grad():
                encoded = encoder(batch_features)
                output = converter(encoded, batch_notes)
            
            outputs.append(output.numpy())
        
        # Combinar salidas
        output_array = np.concatenate(outputs, axis=0)
        
        # Convertir la salida a audio
        output_audio = np.mean(output_array, axis=1)
        
        # Aplicar efectos chiptune
        output_audio = ChiptuneEffects.apply_all_effects(
            output_audio, 
            sr=sr, 
            bit_depth=8, 
            distortion_amount=0.5
        )
        
        # Guardar resultado
        sf.write(output_file, output_audio, sr)
        print(f"Audio convertido guardado en {output_file}")
        
        # Mostrar resumen
        print("\n=== DEMOSTRACIÓN COMPLETADA ===")
        print(f"Audio de entrada: {audio_file}")
        print(f"Archivo MIDI: {midi_file}")
        print(f"Modelo: {model_path}")
        print(f"Resultado: {output_file}")
        print("\nPara una conversión completa con tu propio audio, usa:")
        print(f"python scripts/convert.py --model \"{model_path}\" --input \"{audio_file}\" --midi \"{midi_file}\" --output \"mi_cancion_chiptune.wav\"")
        
        return output_file
    
    except Exception as e:
        print(f"Error durante la demostración: {e}")
        raise
    finally:
        print("\nLimpiando archivos temporales...")
        # Comentar la siguiente línea para mantener los archivos temporales
        # shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ejecutar demostración de Chiptunify')
    parser.add_argument('--input', type=str, default=None, 
                        help='Archivo de audio de entrada opcional')
    args = parser.parse_args()
    
    run_demo(args.input)
