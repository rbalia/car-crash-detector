import pyaudio
import tensorflow as tf
from carcrashdetector import CarCrashDetector


if __name__ == "__main__":
    # Load Tensorflow Model
    tf_model = tf.keras.models.load_model("model_BidLSTM.h5")

    # define stream parameters
    p = pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 32000 # actually it is a fixed parameter

    # Select the stream source (Microphone)
    MIC = p.get_default_output_device_info()["hostApi"]
    SPEAKER = 3 # default pyAudio do not allow to take internal inputs.
                # In Windows i suggest to use VBCable drivers to create a virtual device that map speakers into Mic

    print(f"Device selected as MIC: {p.get_default_output_device_info()}")
    print("List of available devices")
    for i in range(p.get_device_count()):
        print(f"{i} - {p.get_device_info_by_index(i)}")

    # Initialize the stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    #input_device_index=SPEAKER)   # Get by index
                    input_host_api_specific_stream_info = MIC) # Get by info

    # Start stream reading and live prediction
    ccd = CarCrashDetector(tf_model, stream)
    ccd.start()