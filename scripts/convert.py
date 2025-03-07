import os
import sys
import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# Añadir el directorio principal al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chiptunify.model.voice_encoder import LightweightVoiceEncoder
from chiptunify.model.voice_converter import ChiptunifyConverter
from chiptunify.midi.converter import MIDIConverter
from chiptunify.audio.effects import ChiptuneEffects

def parse_args():
    parser = argparse.ArgumentParser(description='Convertir audio a canto chiptune')
    parser.add_argument('--model', type=str, required=True, 
                        help='Ruta al modelo entrenado')
    parser.add_argument('--input', type=str, required=True, 
                        help='Archivo de audio de entrada')
    parser.add_argument('--midi', type=str, required=True, 
                        help='Archivo MIDI con notas')
    parser.add_argument('--output', type=str, default='output_chiptune.wav', 
                        help='Archivo de audio de salida')
    parser.add_argument('--bit_depth', type=int, default=8, 
                        choices=[8, 16], help='Profundidad de bit (8 o 16)')
    parser.add_argument('--distortion', type=float, default=0.4, 
                        help='Cantidad de distorsión (0.0-1.0)')
    return parser.parse_args()

def load_model(model_path):
    """Carga el modelo guardado"""
    print(f"Cargando modelo desde {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = LightweightVoiceEncoder().to(device)
    converter = ChiptunifyConverter().to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        converter.load_state_dict(checkpoint['converter_state_dict'])
        print(f"Modelo cargado correctamente (época {checkpoint['epoch']})")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        sys.exit(1)
        
    return encoder, converter

def convert_audio(args):
    """Convierte audio a canto estilo chiptune usando un modelo entrenado y un archivo MIDI"""
    print("Iniciando conversión de audio...")
    
    # Cargar modelo
    encoder, converter = load_model(args.model)
    encoder.eval()
    converter.eval()
    
    # Cargar audio de entrada
    print(f"Cargando audio de entrada: {args.input}")
    y, sr = librosa.load(args.input, sr=16000, mono=True)
    
    # Si el audio es muy largo, lo recortamos para la demo
    max_seconds = 30
    if len(y) > max_seconds * sr:
        print(f"Audio recortado a {max_seconds} segundos para demostración")
        y = y[:max_seconds * sr]
    
    # Duración del audio en segundos
    duration = len(y) / sr
    print(f"Duración del audio: {duration:.2f} segundos")
    
    # Convertir MIDI a secuencia de notas
    print(f"Procesando archivo MIDI: {args.midi}")
    midi_converter = MIDIConverter(sr=sr)
    try:
        midi_notes = midi_converter.get_torch_tensor(args.midi, duration)
    except Exception as e:
        print(f"Error al procesar archivo MIDI: {e}")
        sys.exit(1)
    
    print(f"Archivo MIDI procesado, encontradas {torch.sum(midi_notes > 0)} notas")
    
    # Codificar audio
    print("Codificando audio de entrada...")
    device = next(encoder.parameters()).device
    features = encoder.extract_features(y, sr)
    features_tensor = torch.FloatTensor(features).to(device)
    
    # NUEVO: Verificar y normalizar longitudes de tensores
    features_len = features_tensor.size(0)
    midi_len = midi_notes.size(0)
    
    print(f"Longitud de características extraídas: {features_len} frames")
    print(f"Longitud de notas MIDI: {midi_len} frames")
    
    # Normalizar longitudes para que coincidan
    if features_len > midi_len:
        print(f"Recortando características de audio para que coincidan con MIDI ({features_len} → {midi_len})")
        features_tensor = features_tensor[:midi_len]
    elif midi_len > features_len:
        print(f"Recortando notas MIDI para que coincidan con audio ({midi_len} → {features_len})")
        midi_notes = midi_notes[:features_len]
    
    print(f"Tensores normalizados a {features_tensor.size(0)} frames")
    
    # Convertir audio a canto según las notas MIDI
    print("Convirtiendo audio a canto con estilo chiptune...")
    
    # Procesar por lotes para evitar problemas de memoria
    batch_size = 50
    outputs = []
    
    # MODIFICADO: Procesamiento por lotes con verificación adicional
    for i in range(0, len(features_tensor), batch_size):
        end_idx = min(i + batch_size, len(features_tensor))
        batch_features = features_tensor[i:end_idx]
        batch_notes = midi_notes[i:end_idx].to(device)
        
        # Verificación de seguridad adicional
        if batch_features.size(0) != batch_notes.size(0):
            print(f"⚠️ Advertencia: Tamaños de batch no coinciden en el índice {i}.")
            min_size = min(batch_features.size(0), batch_notes.size(0))
            batch_features = batch_features[:min_size]
            batch_notes = batch_notes[:min_size]
            print(f"   Ajustados a {min_size} elementos")
            
        # Codificar
        with torch.no_grad():
            encoded = encoder(batch_features)
            output = converter(encoded, batch_notes)
        
        outputs.append(output.cpu().numpy())
    
    # Combinar salidas
    output_array = np.concatenate(outputs, axis=0)
    
    # Convertir la salida a audio (simplificado)
    # En una implementación real se usaría un vocoder o método más avanzado
    print("Generando audio de salida...")
    output_audio = np.mean(output_array, axis=1)
    
    # Normalizar
    output_audio = output_audio / (np.max(np.abs(output_audio)) + 1e-8)
    
    # Aplicar efectos chiptune
    print(f"Aplicando efectos chiptune (bit depth: {args.bit_depth}, distorsión: {args.distortion:.2f})...")
    output_audio = ChiptuneEffects.apply_all_effects(
        output_audio, 
        sr=sr, 
        bit_depth=args.bit_depth, 
        distortion_amount=args.distortion
    )
    
    # Guardar resultado
    print(f"Guardando audio convertido en {args.output}...")
    sf.write(args.output, output_audio, sr)
    
    print("¡Conversión completada!")
    return args.output

if __name__ == "__main__":
    args = parse_args()
    convert_audio(args)