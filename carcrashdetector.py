import os
import tensorflow as tf
import numpy as np
from threading import Thread
from queue import Queue
from pydub import AudioSegment, playback, effects
from matplotlib import pyplot as plt
import datetime

# Force tensorflow to use CPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

TIME_FORMAT = '%H:%M:%S %d/%m/%Y'

def pred2label(pred):
    if pred[0,0] < pred[0,1]:
        return "Crash"
    else:
        return "Noise"

def reshapeWithOverlap(set):
    """
    Create an input with shape supported by the model
    Reshape a sequence of frames with shape (1, 128.000) in to a matrix (16, 160000) with overlap of 50%
    """

    set2D = np.zeros((len(set), 16, 16000), dtype=np.int16)

    step = 16000
    half_step = int(step / 2)
    for i, row in enumerate(set):
        for j in range(int(128000 / half_step) - 1):
            set2D[i, j, :] = row[j * half_step:((j * half_step) + step)]
    return set2D

def normalization(set):
    """ Normalize the set of frames arrays : bring the highest peak at the maximum value (Sample Wise mode)"""

    norm_dataset = set.copy()
    for i, el in enumerate(set):
        audioSeg = AudioSegment(el.tobytes(), frame_rate=32000, sample_width=2, channels=1)
        audioSeg = effects.normalize(audioSeg, 0.01)
        norm_dataset[i] = audioSeg.get_array_of_samples()

    return norm_dataset

def standardizationSampleWise(set):
    """ Apply Standardization """
    # Get parameters
    new_set = []
    for el in set:
        std = np.std(el)
        mean = np.mean(el)
        maximum = np.max(el)
        min = np.min(el)
        scale = max(maximum, abs(min))

        # Normalize
        el = el / scale
        el -= (mean / std)
        new_set.append(el)

    return np.array(new_set)

class CarCrashDetector:
    def __init__(self, model, stream):
        self.model = model
        self.stream = stream
        self.buffer_chunks = []
        self.buffer_time = []
        self.chunk_len = 32000
        self.chunks_to_drop = 1

    def start(self):
        listener = Thread(target=self.audio_listener)
        predictor = Thread(target=self.predict_stream)
        listener.start()
        predictor.start()
        listener.join()
        predictor.join()

    def audio_listener(self):
        """ Append chunks of frames into the buffer """
        while True:
            # Read chunks of frame from stream
            stream_out, timestamp = self.stream.read(self.chunk_len, exception_on_overflow=False), datetime.datetime.now()
            self.buffer_chunks = self.buffer_chunks + [stream_out]
            self.buffer_time = self.buffer_time + [timestamp]


    def predict_stream(self):
        """ Read from buffer and predict"""
        tfmodel_inputs_queue = Queue()
        chunks_to_load = 128000 / self.chunk_len
        history_len = 25
        history = "_"*history_len
        while True:
            # If the buffer contains enough chunks to generate a input for the model
            if len(self.buffer_chunks)*self.chunk_len >=128000:

                # Interpret buffer
                frames = np.frombuffer(b''.join(self.buffer_chunks[0:chunks_to_load]), dtype=np.int16)

                # Frames to Audio
                # audioSeg = AudioSegment(frames.tobytes(), frame_rate=32000,sample_width=2, channels=1)
                # playback.play(audioSeg) # Play

                # Current Chunk informations
                # print(f"Current Chunk infos | dB: {audioSeg.dBFS} "
                #      f"| frame rate: {audioSeg.frame_rate} "
                #      f"| duration: {audioSeg.duration_seconds}")

                # Live wave plot
                # plt.plot(frames)
                # plt.draw()
                # plt.pause(0.0001)
                # plt.clf()

                # Frames preprocessing
                frames = np.expand_dims(frames, axis=0)
                frames = normalization(frames)
                frames = reshapeWithOverlap(frames)
                frames = standardizationSampleWise(frames)
                timestamp = self.buffer_time[3]

                # Remove chunks: The number of chunks to remove, it defines the overlap and prediction frequency
                if self.chunks_to_drop > chunks_to_load:
                    print(f"Error: Cannot drop more than {chunks_to_load} chunks: Dropping {self.chunks_to_drop}")
                    exit(0)
                elif self.chunks_to_drop == 0:
                    print(f"Error: Cannot drop less than 1 chunks: Dropping {self.chunks_to_drop} chunks")
                    exit(0)
                del self.buffer_chunks[0:self.chunks_to_drop]
                del self.buffer_time[0:self.chunks_to_drop]

                # Add preprocessed chunk to model inputs queue
                tfmodel_inputs_queue.put([frames, timestamp])

            # If a new input is ready to be predicted
            if tfmodel_inputs_queue.qsize() > 0:
                # Get input from queue
                batch_frames, timestamp = tfmodel_inputs_queue.get_nowait()

                # Generate predictions
                with tf.device("/cpu:0"):
                    pred = self.model.predict(batch_frames)

                # Interpred prediction
                if pred[0, 0] < pred[0, 1]:
                    textLabel = "Crash"
                    history += "#"
                else:
                    textLabel = "Noise"
                    history += "_"
                if len(history)>history_len:
                    history = history[1:] # drop older

                # Plot prediction and status
                now = datetime.datetime.now()
                print("-" * 10)
                print(f"Current Time:       {now}")#.strftime(TIME_FORMAT)}")
                print(f"Chunk arrival time: {timestamp} | delay: {now-timestamp}")#.strftime(TIME_FORMAT)}")
                print(f"Prediction :        {pred}  | label: {textLabel}")
                print(f"History: old<-<-new {''.join(history)}")
                print(f"Chunks in queue:    {len(self.buffer_chunks)}")

                del batch_frames
                del timestamp

