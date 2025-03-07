import os
import argparse
import torch
import torch.optim as optim
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# Añadir el directorio principal al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chiptunify.model.voice_encoder import LightweightVoiceEncoder
from chiptunify.model.voice_converter import ChiptunifyConverter

def parse_args():
    parser = argparse.ArgumentParser(description='Entrenar modelo Chiptunify')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directorio con archivos de audio para entrenamiento')
    parser.add_argument('--output_dir', type=str, default='./models', 
                        help='Directorio para guardar el modelo')
    parser.add_argument('--model_name', type=str, default='chiptunify_model', 
                        help='Nombre del modelo')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Tamaño de batch')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Tasa de aprendizaje')
    return parser.parse_args()

def prepare_dataset(input_dir):
    """Prepara el dataset a partir de archivos de audio"""
    print(f"Preparando dataset desde {input_dir}")
    
    audio_files = [f for f in os.listdir(input_dir) 
                  if f.endswith(('.wav', '.mp3', '.flac'))]
    
    if not audio_files:
        raise ValueError(f"No se encontraron archivos de audio en {input_dir}")
    
    print(f"Encontrados {len(audio_files)} archivos de audio")
    
    # En un caso real, procesaríamos todos los archivos
    # Para este ejemplo simplificado, solo tomamos hasta 5 archivos
    sample_limit = min(5, len(audio_files))
    audio_data = []
    problemas_encontrados = []
    
    for file in tqdm(audio_files[:sample_limit], desc="Cargando audio"):
        file_path = os.path.join(input_dir, file)
        try:
            y, sr = librosa.load(file_path, sr=16000, mono=True)
            
            # Verificaciones de calidad
            if len(y) < sr * 1.0:  # Audio menor a 1 segundo
                problemas_encontrados.append(f"{file}: duración muy corta ({len(y)/sr:.2f}s)")
                continue
                
            if np.mean(np.abs(y)) < 0.01:  # Audio muy silencioso
                problemas_encontrados.append(f"{file}: audio muy silencioso, posible problema de grabación")
                continue
            
            # Tomar solo hasta 10 segundos para simplificar
            if len(y) > 10 * sr:
                y = y[:10 * sr]
                
            # Verificar que podamos extraer características correctamente
            try:
                encoder = LightweightVoiceEncoder()
                features = encoder.extract_features(y)
                print(f"{file}: {features.shape[0]} frames, {features.shape[1]} características")
            except Exception as e:
                problemas_encontrados.append(f"{file}: error al extraer características: {e}")
                continue
                
            audio_data.append(y)
        except Exception as e:
            print(f"Error al cargar {file}: {e}")
    
    # Mostrar problemas encontrados
    if problemas_encontrados:
        print("\n--- ADVERTENCIAS DE ARCHIVOS DE AUDIO ---")
        for problema in problemas_encontrados:
            print(f"⚠️ {problema}")
        print("------------------------------------\n")
    
    return audio_data

def train(args):
    """Función principal de entrenamiento"""
    print("Inicializando entrenamiento de Chiptunify...")
    
    # Preparar directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Cargar datos
    audio_data = prepare_dataset(args.input_dir)
    if not audio_data:
        print("No se pudieron cargar datos de audio. Abortando.")
        return
    
    # Inicializar modelos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    encoder = LightweightVoiceEncoder().to(device)
    converter = ChiptunifyConverter().to(device)
    
    # Optimizadores
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    converter_optimizer = optim.Adam(converter.parameters(), lr=args.lr)
    
    # Entrenamiento simplificado
    print("Iniciando entrenamiento...")
    losses = []
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        
        # En un caso real usaríamos DataLoader
        for audio in tqdm(audio_data, desc=f"Época {epoch+1}/{args.epochs}"):
            # Extraer características
            features = encoder.extract_features(audio)
            features_tensor = torch.FloatTensor(features).to(device)
            
            # Simular notas MIDI (en un caso real usaríamos notas reales)
            batch_size = min(args.batch_size, features_tensor.size(0))
            indices = np.random.choice(features_tensor.size(0), batch_size, replace=False)
            batch_features = features_tensor[indices]
            
            # Simular notas MIDI aleatorias entre 48 y 84 (rango vocal típico)
            batch_notes = torch.randint(48, 84, (batch_size,)).long().to(device)
            
            # Simular audio objetivo (en un caso real tendríamos pares reales)
            target_audio = torch.randn(batch_size, 128).to(device)
            
            # Entrenar
            # En un modelo real, separaríamos estos pasos
            encoder_optimizer.zero_grad()
            converter_optimizer.zero_grad()
            
            # Forward
            encoded = encoder(batch_features)
            output = converter(encoded, batch_notes)
            
            # Loss simplificada
            loss = torch.nn.functional.mse_loss(output, target_audio)
            
            # Backward
            loss.backward()
            encoder_optimizer.step()
            converter_optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(audio_data)
        losses.append(avg_loss)
        print(f"Época {epoch+1}/{args.epochs} - Pérdida: {avg_loss:.4f}")
        
        # Guardar cada 10 épocas
        if (epoch + 1) % 10 == 0:
            save_checkpoint(encoder, converter, args.output_dir, args.model_name, epoch)
    
    # Guardar modelo final
    save_checkpoint(encoder, converter, args.output_dir, args.model_name, args.epochs)
    
    # Graficar pérdidas
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Pérdida durante entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.savefig(os.path.join(args.output_dir, f"{args.model_name}_loss.png"))
    
    print(f"Entrenamiento completado. Modelo guardado en {args.output_dir}")

def save_checkpoint(encoder, converter, output_dir, model_name, epoch):
    """Guarda punto de control del modelo"""
    checkpoint = {
        'encoder_state_dict': encoder.state_dict(),
        'converter_state_dict': converter.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, os.path.join(output_dir, f"{model_name}_epoch{epoch}.pth"))
    print(f"Modelo guardado: {model_name}_epoch{epoch}.pth")

if __name__ == "__main__":
    args = parse_args()
    train(args)
