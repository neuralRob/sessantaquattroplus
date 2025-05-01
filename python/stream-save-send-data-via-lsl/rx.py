#!/usr/bin/env python3
import sys
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from pylsl import StreamInlet, resolve_stream
import threading
import queue

# -----------------------------------------------------------------------------
# Global Plot Config: white background and black foreground.
# -----------------------------------------------------------------------------
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# -----------------------------------------------------------------------------
# Configuration Parameters
# -----------------------------------------------------------------------------
NUM_CHANNELS = 8  # We expect 8-channel envelope data from the sender.
plot_arraysize = None  # Will be set based on the stream's sample rate.

# Global queue for LSL samples.
lsl_queue = queue.Queue()

# -----------------------------------------------------------------------------
# LSL Receiver Thread Function
# -----------------------------------------------------------------------------
def lsl_receiver_thread(inlet):
    """Continuously pulls samples from the LSL inlet and pushes them into a queue."""
    while True:
        sample, timestamp = inlet.pull_sample(timeout=1.0)
        if sample:
            lsl_queue.put(sample)

# -----------------------------------------------------------------------------
# Plotter Class for Receiving Envelopes (using pyqtgraph)
# -----------------------------------------------------------------------------
class Plotter:
    def __init__(self, num_channels, plot_arraysize):
        self.num_channels = num_channels
        self.arraysize = plot_arraysize
        self.win = pg.GraphicsLayoutWidget(title="LSL EMG Envelopes")
        self.win.resize(1000, 600)
        self.buffers = [np.zeros(plot_arraysize) for _ in range(num_channels)]
        self.curves = []
        for i in range(num_channels):
            p = self.win.addPlot(row=i, col=0)
            p.setLabel('left', f"Channel {i+1}")
            p.setLabel('bottom', "Time (s)")
            p.setYRange(0, 1000)  # Adjust as needed.
            p.setXRange(0, 10)
            curve = p.plot(pen=pg.mkPen(color='b', width=3))
            self.curves.append(curve)
        self.win.show()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(10)

    def update(self):
        while not lsl_queue.empty():
            sample = lsl_queue.get_nowait()
            for i in range(self.num_channels):
                self.buffers[i] = np.roll(self.buffers[i], -1)
                self.buffers[i][-1] = sample[i]
        for i in range(self.num_channels):
            self.curves[i].setData(self.buffers[i])

# -----------------------------------------------------------------------------
# Main Function for Receiver
# -----------------------------------------------------------------------------
def main():
    print("Looking for an EMGEnvelope stream...")
    streams = resolve_stream('name', 'EMGEnvelope')
    inlet = StreamInlet(streams[0])
    fs = float(streams[0].nominal_srate())
    global plot_arraysize
    plot_arraysize = int(fs * 10)
    print(f"Found stream with sample rate: {fs} Hz, buffer set to {plot_arraysize} samples.")
    
    receiver_thread = threading.Thread(target=lsl_receiver_thread, args=(inlet,))
    receiver_thread.daemon = True
    receiver_thread.start()

    app = QtWidgets.QApplication([])
    plotter = Plotter(NUM_CHANNELS, plot_arraysize)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
