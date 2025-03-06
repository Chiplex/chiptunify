"""
Interfaz gráfica para Chiptunify usando Gradio
"""

import os
import sys
import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
import tempfile
import gradio as gr

# Añadir el directorio principal al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chiptunify.model.voice_encoder import LightweightVoiceEncoder
from chiptunify.model.voice_converter import ChiptunifyConverter
from chiptunify.midi.converter import MIDIConverter
from chiptunify.audio.effects import ChiptuneEffects

# Cache para modelos cargados
model_cache = {}

def load_model(model_path):
    """Carga el modelo desde el path especificado"""
    if model_path in model_cache:
        return model_cache[model_path]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = LightweightVoiceEncoder().to(device)
    converter = ChiptunifyConverter().to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        converter.load_state_dict(checkpoint['converter_state_dict'])
        print(f"Modelo cargado correctamente (época {checkpoint['epoch']})")
        
        # Guardar en caché
        model_cache[model_path] = (encoder, converter)
        return encoder, converter
    except Exception as e:
        raise gr.Error(f"Error al cargar el modelo: {e}")

def process_audio(
    audio_input, 
    midi_file, 
    model_path, 
    bit_depth=8, 
    distortion=0.5,
    downsampling=4
):
    """Procesa el audio usando el modelo y los parámetros especificados"""
    
    try:
        # Cargar modelos
        encoder, converter = load_model(model_path)
        encoder.eval()
        converter.eval()
        
        # Si es una tupla (como la que devuelve Gradio), extraer el path
        if isinstance(audio_input, tuple):
            audio_path, sr = audio_input
        else:
            audio_path = audio_input
            sr = 16000
            
        # Cargar audio
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Duración del audio en segundos
        duration = len(y) / sr
        
        # Convertir MIDI a secuencia de notas
        midi_converter = MIDIConverter(sr=sr)
        midi_notes = midi_converter.get_torch_tensor(midi_file, duration)
        
        # Procesar audio
        device = next(encoder.parameters()).device
        
        # Extraer características
        features = encoder.extract_features(y, sr)
        features_tensor = torch.FloatTensor(features).to(device)
        
        # Convertir audio a canto según las notas MIDI (por lotes)
        batch_size = 50
        outputs = []
        
        for i in range(0, len(features_tensor), batch_size):
            end_idx = min(i + batch_size, len(features_tensor))
            batch_features = features_tensor[i:end_idx]
            
            if i < len(midi_notes):
                batch_notes = midi_notes[i:end_idx].to(device)
            else:
                # Si la secuencia MIDI es más corta, repetir las últimas notas
                batch_notes = torch.zeros(len(batch_features), dtype=torch.long).to(device)
            
            # Codificar
            with torch.no_grad():
                encoded = encoder(batch_features)
                output = converter(encoded, batch_notes)
            
            outputs.append(output.cpu().numpy())
        
        # Combinar salidas
        output_array = np.concatenate(outputs, axis=0)
        
        # Convertir la salida a audio
        output_audio = np.mean(output_array, axis=1)
        
        # Normalizar
        output_audio = output_audio / (np.max(np.abs(output_audio)) + 1e-8)
        
        # Aplicar efectos chiptune
        output_audio = ChiptuneEffects.apply_all_effects(
            output_audio, 
            sr=sr, 
            bit_depth=bit_depth, 
            distortion_amount=distortion,
            downsampling_factor=downsampling,
            block_size=16
        )
        
        # Guardar resultado en un archivo temporal
        output_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(output_file, output_audio, sr)
        
        return output_file, f"Conversión exitosa! Audio convertido a {bit_depth}-bit con efectos chiptune."
    
    except Exception as e:
        raise gr.Error(f"Error al procesar audio: {e}")

def create_ui():
    """Crea la interfaz de usuario con Gradio"""
    
    with gr.Blocks(title="Chiptunify - Conversión de Voz a Canto Chiptune") as demo:
        gr.Markdown("""
        # 🎮🎤 Chiptunify
        
        Convierte tu voz a canto estilo chiptune (8-bit/16-bit)
        
        ## Instrucciones
        1. Sube un archivo de audio con tu voz
        2. Sube un archivo MIDI con la melodía
        3. Selecciona un modelo pre-entrenado (o usa el demo)
        4. Ajusta los parámetros de efectos chiptune
        5. ¡Haz clic en "Convertir" y escucha el resultado!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Audio de entrada (voz)",
                    type="filepath"
                )
                
                midi_file = gr.File(
                    label="Archivo MIDI (melodía)",
                    file_types=[".mid", ".midi"],
                )
                
                model_path = gr.Dropdown(
                    label="Modelo",
                    choices=["demo_model.pth"],  # Se puede actualizar con modelos disponibles
                    value="demo_model.pth"
                )
                
                with gr.Accordion("Parámetros de efectos", open=False):
                    bit_depth = gr.Radio(
                        label="Profundidad de bits",
                        choices=[8, 16],
                        value=8
                    )
                    
                    distortion = gr.Slider(
                        label="Distorsión",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05
                    )
                    
                    downsampling = gr.Slider(
                        label="Downsampling factor",
                        minimum=1,
                        maximum=8,
                        value=4,
                        step=1
                    )
                
                convert_button = gr.Button("Convertir", variant="primary")
            
            with gr.Column(scale=1):
                output_audio = gr.Audio(label="Audio convertido")
                output_message = gr.Textbox(label="Estado")
        
        # Conectar funciones
        convert_button.click(
            fn=process_audio,
            inputs=[audio_input, midi_file, model_path, bit_depth, distortion, downsampling],
            outputs=[output_audio, output_message]
        )
        
        # Ejemplos
        gr.Markdown("### Ejemplos")
        examples = gr.Examples(
            examples=[
                # Aquí puedes añadir ejemplos predefinidos
            ],
            inputs=[audio_input, midi_file],
        )
    
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inicia la interfaz web de Chiptunify")
    parser.add_argument("--share", action="store_true", help="Compartir públicamente")
    parser.add_argument("--port", type=int, default=7860, help="Puerto para el servidor")
    
    args = parser.parse_args()
    
    # Verificar que gradio esté instalado
    try:
        import gradio
    except ImportError:
        print("Gradio no está instalado. Instálalo con: pip install gradio")
        sys.exit(1)
    
    # Crear y lanzar UI
    demo = create_ui()
    demo.launch(share=args.share, server_port=args.port)
