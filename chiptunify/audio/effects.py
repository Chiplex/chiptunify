import numpy as np
from scipy import signal
import librosa

class ChiptuneEffects:
    """
    Clase para aplicar efectos de audio estilo chiptune (8-bit/16-bit)
    """
    
    @staticmethod
    def bitcrush(audio, bit_depth=8, sr=44100, downsampling_factor=1):
        """
        Reduce la calidad de bits del audio para lograr efecto lo-fi
        
        Args:
            audio: numpy array con forma (samples,)
            bit_depth: profundidad de bits (8 o 16)
            sr: frecuencia de muestreo
            downsampling_factor: factor para reducir la frecuencia de muestreo
        """
        # Simular reducción de bits
        max_val = 2**(bit_depth - 1) - 1
        audio_quantized = np.round(audio * max_val) / max_val
        
        # Simular reducción de frecuencia de muestreo
        if downsampling_factor > 1:
            # Submuestreo
            audio_quantized = audio_quantized[::downsampling_factor]
            # Volver a la tasa original (simulando hold)
            audio_quantized = np.repeat(audio_quantized, downsampling_factor)
            # Asegurar que la longitud sea correcta
            if len(audio_quantized) > len(audio):
                audio_quantized = audio_quantized[:len(audio)]
            elif len(audio_quantized) < len(audio):
                audio_quantized = np.pad(audio_quantized, (0, len(audio) - len(audio_quantized)))
                
        return audio_quantized
        
    @staticmethod
    def distortion(audio, amount=0.5):
        """
        Aplica distorsión al audio
        
        Args:
            audio: numpy array con forma (samples,)
            amount: cantidad de distorsión (0.0 - 1.0)
        """
        # Normalizar
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Aplicar distorsión mediante limitación suave
        threshold = 1.0 - amount * 0.9
        audio_distorted = np.tanh(audio / threshold)
        
        # Normalizar nuevamente
        return audio_distorted / (np.max(np.abs(audio_distorted)) + 1e-8)
        
    @staticmethod
    def pixelate(audio, block_size=32):
        """
        Efecto de 'pixelado de audio' - mantiene el valor constante 
        por bloques de muestras
        """
        audio_blocks = np.array([
            np.mean(audio[i:i+block_size]) 
            for i in range(0, len(audio), block_size)
        ])
        
        # Repetir cada valor block_size veces
        audio_pixelated = np.repeat(audio_blocks, block_size)
        
        # Asegurar que la longitud sea correcta
        if len(audio_pixelated) > len(audio):
            audio_pixelated = audio_pixelated[:len(audio)]
        elif len(audio_pixelated) < len(audio):
            audio_pixelated = np.pad(audio_pixelated, (0, len(audio) - len(audio_pixelated)))
            
        return audio_pixelated
    
    @staticmethod
    def apply_all_effects(audio, sr=44100, bit_depth=8, distortion_amount=0.4, 
                         downsampling_factor=4, block_size=16):
        """Aplica todos los efectos chiptune en secuencia"""
        
        audio = ChiptuneEffects.distortion(audio, amount=distortion_amount)
        audio = ChiptuneEffects.bitcrush(audio, bit_depth=bit_depth, 
                                       sr=sr, downsampling_factor=downsampling_factor)
        audio = ChiptuneEffects.pixelate(audio, block_size=block_size)
        
        return audio
