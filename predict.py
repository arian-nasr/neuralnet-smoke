import tensorflow as tf
import pyaudio
import wave
from io import BytesIO

audio = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 10

saved_model_path = '/Users/arian/tensorflow-test/smoke_detector_yamnet'
reloaded_model = tf.saved_model.load(saved_model_path)
my_classes = ['nosmoke','smoke']

def load_wav_16k_mono(memory_file):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.constant(memory_file.getvalue())
    wav, _ = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)

    return wav

def record_audio(audio):
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    #input_device_index= 0,
                    frames_per_buffer=CHUNK)
    print('Recording')
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print('Done Recording')
    stream.stop_stream()
    stream.close()

    in_memory_file = BytesIO()
    waveFile = wave.open(in_memory_file, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    return in_memory_file

def make_prediction(wav_data):
    serving_results = reloaded_model.signatures['serving_default'](wav_data)
    smoke_or_no = my_classes[tf.math.argmax(serving_results['classifier'])]
    print(f'The main sound is: {smoke_or_no}')

while True:
    memory_file = record_audio(audio)
    wav_data = load_wav_16k_mono(memory_file)
    make_prediction(wav_data)