# Chiptunify 🎮🎤

Chiptunify es un modelo ligero de IA basado en Retrieval-Based Voice Conversion que te permite "cantar sin saber cantar" en estilo chiptune (8-bit/16-bit).

## Características principales

- **Modelo ligero**: Diseñado para funcionar en equipos con recursos limitados
- **Conversión de voz a canto**: Transforma tu voz hablada en canto afinado
- **Integración MIDI**: Usa archivos MIDI para controlar notas y melodías
- **Efectos chiptune**: Aplica efectos como distorsión y bitcrush para lograr sonidos retro

## Requisitos

- Python 3.8+
- PyTorch 1.9+
- (Ver requirements.txt para la lista completa)

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Chiplex/chiptunify.git
cd chiptunify

# Instalar dependencias

```

## Uso rápido

### Entrenamiento

```bash
python scripts/train.py --input_dir "./datos_voz" --epochs 50 --model_name "mi_voz_chip"
```

### Conversión

```bash
python scripts/convert.py --model "mi_voz_chip" --input "mi_voz.wav" --midi "melodia.mid" --output "mi_cancion_chiptune.wav"
```

## Cómo funciona

1. El sistema analiza tu voz para extraer características únicas
2. Mapea estas características a un espacio de representación simplificado
3. Utiliza archivos MIDI para determinar notas y ritmo
4. Aplica efectos chiptune para lograr el sonido retro de 8-bit/16-bit

## Ajustes de efectos chiptune

- **Bit Depth**: 8-bit o 16-bit
- **Distorsión**: 0-100%
- **Bitcrush**: Nivel de reducción de frecuencia de muestreo