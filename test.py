import tensorflow as tf
import pyaudio
import wave
from io import BytesIO
from os import system

audio = pyaudio.PyAudio()

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "temp.wav"

saved_model_path = "/Users/arian/tensorflow-test/smoke_detector_yamnet"
reloaded_model = tf.saved_model.load(saved_model_path)
my_classes = ['nosmoke','smoke']

# Utility functions for loading audio files and making sure the sample rate is correct.
#@tf.function
def load_wav_16k_mono(memory_file):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    #file_contents = tf.io.read_file(filename)
    file_contents = tf.constant(memory_file.getvalue())
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    return wav

while True:
    # start Recording
    #audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    #input_device_index= 0,
                    frames_per_buffer=CHUNK)
    print("recording...")
    frames = []
        
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    system('clear')
    print("finished recording")
        
        
    # stop Recording
    stream.stop_stream()
    stream.close()
    #audio.terminate()

    in_memory_file = BytesIO()

    waveFile = wave.open(in_memory_file, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    #testing_wav_file_name = "./temp.wav"
    #audio_tensor = tf.constant(in_memory_file.getvalue())
    #print(audio_tensor)
    testing_wav_data = load_wav_16k_mono(in_memory_file)

    serving_results = reloaded_model.signatures['serving_default'](testing_wav_data)
    print(serving_results)
    cat_or_dog = my_classes[tf.math.argmax(serving_results['classifier'])]
    print(f'The main sound is: {cat_or_dog}')