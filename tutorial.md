# Tutorial de Chiptunify

Este tutorial te guiará a través de los pasos para utilizar Chiptunify, desde la preparación de los datos hasta la conversión final.

## 1. Instalación

Primero, asegúrate de tener todas las dependencias instaladas:

```bash
pip install -r requirements.txt
```

## 2. Preparación de datos

Chiptunify necesita dos tipos de datos:

1. **Audio de tu voz**: Grabaciones claras de tu voz hablando o cantando
2. **Archivos MIDI**: Contienen las notas que quieres que "cante" tu voz

### Preparación de audio

Para obtener mejores resultados:

- Graba en un ambiente silencioso
- Habla claramente y con un ritmo constante
- Usa un micrófono de calidad razonable
- Guarda los archivos como WAV mono a 16kHz o 44.1kHz

### Preparación de archivos MIDI

Puedes crear archivos MIDI con cualquier software de producción musical. Algunas opciones gratuitas:

- [MuseScore](https://musescore.org/)
- [LMMS](https://lmms.io/)
- [Aria Maestosa](https://ariamaestosa.github.io/ariamaestosa/docs/index.html)

## 3. Entrenamiento del modelo

Para entrenar un modelo con tu voz:

```bash
python scripts/train.py --input_dir "./mis_grabaciones" --epochs 100 --model_name "mi_voz_chiptune"
```

Parámetros importantes:
- `--input_dir`: Directorio con tus archivos de audio
- `--epochs`: Número de épocas de entrenamiento (100-200 recomendado)
- `--model_name`: Nombre para guardar tu modelo
- `--output_dir`: Directorio donde se guardará el modelo (por defecto: "./models")

## 4. Conversión a canto chiptune

Una vez entrenado el modelo, puedes convertir tu voz a canto chiptune:

```bash
python scripts/convert.py --model "./models/mi_voz_chiptune_epoch100.pth" --input "mi_voz.wav" --midi "melodia.mid" --output "resultado_chiptune.wav"
```

Parámetros:
- `--model`: Ruta al modelo entrenado
- `--input`: Archivo de audio con tu voz
- `--midi`: Archivo MIDI con la melodía
- `--bit_depth`: Profundidad de bits (8 o 16)
- `--distortion`: Cantidad de distorsión (0.0-1.0)

## 5. Usando la interfaz gráfica

Para una experiencia más amigable, puedes usar la interfaz web:

```bash
python scripts/ui.py
```

Esto abrirá una interfaz en tu navegador donde podrás:
- Subir archivos de audio y MIDI
- Seleccionar modelos preentrenados
- Ajustar parámetros de efectos
- Escuchar el resultado inmediatamente

## 6. Consejos para mejores resultados

- **Entrena con variedad**: Incluye diferentes tonos y estilos de habla
- **Experimenta con efectos**: Prueba diferentes configuraciones de distorsión y bit depth
- **Ajusta el MIDI**: Asegúrate de que las notas estén en un rango vocal cómodo
- **Post-procesamiento**: Puedes mejorar el resultado con efectos adicionales en un DAW

## 7. Ejemplos y demos

Ejecuta el script de demostración para ver el flujo completo:

```bash
python scripts/demo.py
```

## 8. Solución de problemas

### El modelo no aprende bien
- Aumenta el número de épocas
- Verifica la calidad de tus grabaciones
- Prueba con una tasa de aprendizaje diferente

### El audio suena distorsionado
- Reduce el parámetro de distorsión
- Prueba con 16-bit en lugar de 8-bit
- Asegúrate de que las notas MIDI estén en un rango adecuado

### Errores de memoria
- Reduce el tamaño de batch
- Limita la duración de los archivos de audio
- Ejecuta en un equipo con más RAM