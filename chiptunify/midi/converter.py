import mido
import numpy as np
import torch

class MIDIConverter:
    """
    Clase para procesar archivos MIDI y extraer información musical relevante
    para la conversión de voz a canto estilo chiptune.
    """
    
    def __init__(self, sr=44100, hop_length=256):
        """
        Args:
            sr: Frecuencia de muestreo del audio
            hop_length: Tamaño del salto para la conversión de tiempo a frames
        """
        self.sr = sr
        self.hop_length = hop_length
        
    def midi_to_note_sequence(self, midi_file, max_time_seconds=None):
        """
        Convierte un archivo MIDI en una secuencia de notas con tiempos
        
        Args:
            midi_file: Ruta al archivo MIDI
            max_time_seconds: Tiempo máximo en segundos para procesar
        
        Returns:
            Una lista de tuplas (tiempo_inicio, tiempo_final, nota, velocidad)
        """
        mid = mido.MidiFile(midi_file)
        notes = []
        current_time = 0
        
        # Tiempo total en segundos
        if max_time_seconds is None:
            max_time_seconds = mid.length
            
        # Buscar pistas con notas
        for track in mid.tracks:
            track_notes = []
            current_time = 0
            active_notes = {}  # nota -> (tiempo_inicio, velocidad)
            
            for msg in track:
                current_time += msg.time
                
                # Convertir de ticks a segundos
                seconds = mido.tick2second(current_time, mid.ticks_per_beat, get_tempo(mid))
                
                if seconds > max_time_seconds:
                    break
                    
                # Procesar eventos de nota
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Inicio de nota
                    active_notes[msg.note] = (seconds, msg.velocity)
                elif (msg.type == 'note_off' or 
                     (msg.type == 'note_on' and msg.velocity == 0)):
                    # Final de nota
                    if msg.note in active_notes:
                        start_time, velocity = active_notes[msg.note]
                        track_notes.append((start_time, seconds, msg.note, velocity))
                        del active_notes[msg.note]
                        
            notes.extend(track_notes)
            
        # Ordenar por tiempo de inicio
        notes.sort(key=lambda x: x[0])
        return notes
    
    def create_note_frame_sequence(self, notes, duration_seconds):
        """
        Convierte la secuencia de notas en una secuencia de frames
        
        Args:
            notes: Lista de notas (tiempo_inicio, tiempo_final, nota, velocidad)
            duration_seconds: Duración total en segundos
        
        Returns:
            Array numpy con la nota activa para cada frame
        """
        # Calcular número total de frames
        num_frames = int(duration_seconds * self.sr / self.hop_length)
        
        # Inicializar array con silencios (nota 0)
        note_frames = np.zeros(num_frames, dtype=np.int64)
        
        # Si no hay notas, devolver array de silencios del tamaño correcto
        if not notes:
            print("⚠️ Advertencia: No se encontraron notas en el archivo MIDI.")
            return note_frames
            
        # Llenar array con notas
        for start_time, end_time, note, velocity in notes:
            start_frame = int(start_time * self.sr / self.hop_length)
            end_frame = int(end_time * self.sr / self.hop_length)
            
            # Asegurar que estamos dentro de los límites
            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = max(0, min(end_frame, num_frames))
            
            # Asignar nota a frames
            note_frames[start_frame:end_frame] = note
            
        return note_frames
    
    def get_torch_tensor(self, midi_file, duration_seconds):
        """Convierte un archivo MIDI a un tensor de PyTorch"""
        notes = self.midi_to_note_sequence(midi_file, duration_seconds)
        note_frames = self.create_note_frame_sequence(notes, duration_seconds)
        return torch.LongTensor(note_frames)


def get_tempo(mid):
    """Extrae el tempo de un archivo MIDI"""
    default_tempo = 500000  # 120 BPM
    
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return msg.tempo
                
    return default_tempo
