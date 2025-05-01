#!/usr/bin/env python3
import socket
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import communication_sessantaquattro_plus as communication
import queue
import threading
import csv
import sys
import datetime  # For generating filename based on current date/time
import os

# -----------------------------------------------------------------------------
# Global Plot Config: white background and black foreground.
# -----------------------------------------------------------------------------
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# -----------------------------------------------------------------------------
# Configuration Flags and Parameters
# -----------------------------------------------------------------------------
# Set to True to display the real-time plot (for the main channels and aux); False to disable plotting.
PLOT_DATA = True

# Number of main channels to display (in addition, the 10th channel will be plotted as auxiliary).
NUM_PLOT_CHANNELS = 4

# Index of the auxiliary channel
AUX_INDEX = 8 # 1st aux channel -> index: 8; 2nd aux channel -> index: 9

# Exponential filter coefficient for computing the envelope.
ALPHA = 0.01

# Colors used for plotting curves (for the envelope)
track_color = ['k', 'k', 'k', 'k', 'k', 'k', 'k']

# -----------------------------------------------------------------------------
# Global Queues for inter-thread communication
# -----------------------------------------------------------------------------
csv_queue = queue.Queue()
if PLOT_DATA:
    plot_queue = queue.Queue()
    aux_queue = queue.Queue()  # For the auxiliary channel (10th channel)

# -----------------------------------------------------------------------------
# Reader Thread Function
# -----------------------------------------------------------------------------
def reader_thread_func(socket_conn, num_channels, bytes_in_sample):
    """Continuously reads data from the socket and pushes it to the CSV and (optionally) plot queues."""
    while True:
        sample_bytes = communication.read_raw_bytes(socket_conn, num_channels, bytes_in_sample)
        sample_values = communication.bytes_to_integers(
            sample_bytes, num_channels, bytes_in_sample, output_milli_volts=False)
        # Place the complete sample into the CSV queue.
        csv_queue.put(sample_values)
        if PLOT_DATA:
            # Push the main channels (first NUM_PLOT_CHANNELS)
            plot_queue.put(sample_values[0:NUM_PLOT_CHANNELS])
            # Also push the auxiliary channel (channel 10; index 9) if available.
            if len(sample_values) >= 10:
                aux_queue.put(sample_values[AUX_INDEX])

# -----------------------------------------------------------------------------
# CSV Writer Thread Function (modified to support stopping via stop_event)
# -----------------------------------------------------------------------------
def csv_writer_thread_func(csv_filename, stop_event):
    """Continuously takes samples from the csv_queue and writes them to a CSV file until stop_event is set."""
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        first_item = True
        while not stop_event.is_set():
            try:
                sample = csv_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if first_item:
                header = [f"Channel{i+1}" for i in range(len(sample))]
                writer.writerow(header)
                first_item = False
            writer.writerow(sample)
            csvfile.flush()

# -----------------------------------------------------------------------------
# Plotter Class (using pyqtgraph) with extra auxiliary plot added
# -----------------------------------------------------------------------------
if PLOT_DATA:
    class Plotter:
        def __init__(self, num_channels, plot_arraysize):
            # num_channels here refers to main channels only.
            self.num_channels = num_channels
            self.arraysize = plot_arraysize  # Buffer length covering 10 seconds.
            self.win = pg.GraphicsLayoutWidget(title="Real-time EMG Data")
            self.win.resize(1000, 600)
            # Each main channel has two buffers: one for raw data and one for the envelope.
            self.raw_buffers = [np.zeros(plot_arraysize) for _ in range(num_channels)]
            self.env_buffers = [np.zeros(plot_arraysize) for _ in range(num_channels)]
            # Current envelope value per main channel.
            self.current_env = [0.0] * num_channels

            # Create curves for main channels.
            self.raw_curves = []
            self.env_curves = []
            for i in range(num_channels):
                p = self.win.addPlot(row=i, col=0)
                p.setLabel('left', f"Channel {i+1}")
                p.setLabel('bottom', "Time (s)")
                p.setYRange(-10000, 10000)
                p.setXRange(0, 10)
                raw_curve = p.plot(pen=pg.mkPen('grey', width=1))
                env_curve = p.plot(pen=pg.mkPen(track_color[i % len(track_color)], width=3))
                self.raw_curves.append(raw_curve)
                self.env_curves.append(env_curve)
            # Now add an extra plot for the auxiliary channel (10th channel).
            self.aux_buffer = np.zeros(plot_arraysize)
            self.aux_curve = None
            p_aux = self.win.addPlot(row=self.num_channels, col=0)
            p_aux.setLabel('left', "Auxiliary")
            p_aux.setLabel('bottom', "Time (s)")
            p_aux.setYRange(-10000, 10000)
            p_aux.setXRange(0, 10)
            # For the aux channel, use a distinctive pen (here, blue, thicker).
            self.aux_curve = p_aux.plot(pen=pg.mkPen('b', width=3))
            self.win.show()
            # Use a QTimer to update the plot periodically.
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update)
            self.timer.start(10)  # Update every 10 ms.

        def update(self):
            # Process main channels.
            while not plot_queue.empty():
                sample = plot_queue.get_nowait()
                for i in range(self.num_channels):
                    self.raw_buffers[i] = np.roll(self.raw_buffers[i], -1)
                    self.raw_buffers[i][-1] = sample[i]
                    # Compute envelope using exponential filtering.
                    self.current_env[i] = ALPHA * abs(sample[i]) + (1 - ALPHA) * self.current_env[i]
                    self.env_buffers[i] = np.roll(self.env_buffers[i], -1)
                    self.env_buffers[i][-1] = self.current_env[i]
            for i in range(self.num_channels):
                self.raw_curves[i].setData(self.raw_buffers[i])
                self.env_curves[i].setData(self.env_buffers[i])
            # Process auxiliary channel.
            while not aux_queue.empty():
                aux_sample = aux_queue.get_nowait()
                self.aux_buffer = np.roll(self.aux_buffer, -1)
                self.aux_buffer[-1] = aux_sample
            self.aux_curve.setData(self.aux_buffer)

# -----------------------------------------------------------------------------
# Main Function: Setup Socket, Threads, and UI Controls
# -----------------------------------------------------------------------------
def main():
    ip_address = '0.0.0.0'
    port = 45454

    # Create a TCP socket and set options.
    sq_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sq_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sq_socket.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)

    # Create start command and retrieve device settings.
    (start_command,
     number_of_channels,
     sample_frequency,
     bytes_in_sample) = communication.create_bin_command(start=1)

    print('Starting to log data: {} channels with {} sampling rate'.format(number_of_channels, sample_frequency))
    print("number of channels", number_of_channels)

    # Open connection to the Sessantaquattro device.
    connection = communication.connect_to_sq(sq_socket, ip_address, port, start_command)

    # Start the reader thread.
    reader_thread = threading.Thread(target=reader_thread_func, args=(connection, number_of_channels, bytes_in_sample))
    reader_thread.daemon = True
    reader_thread.start()

    # Global variables for recording control.
    global RECORDING, csv_writer_thread, csv_writer_stop_event
    RECORDING = False
    csv_writer_thread = None
    csv_writer_stop_event = None

    if PLOT_DATA:
        # Create the main application window with a control panel.
        app = QtWidgets.QApplication([])
        mainWindow = QtWidgets.QWidget()
        mainLayout = QtWidgets.QVBoxLayout(mainWindow)
        # --- Control Panel ---
        controlLayout = QtWidgets.QHBoxLayout()
        filenameLineEdit = QtWidgets.QLineEdit()
        filenameLineEdit.setReadOnly(True)
        saveButton = QtWidgets.QPushButton("Save")
        statusLabel = QtWidgets.QLabel("Not recording")
        controlLayout.addWidget(filenameLineEdit)
        controlLayout.addWidget(saveButton)
        controlLayout.addWidget(statusLabel)
        mainLayout.addLayout(controlLayout)
        # --- Plot Area ---
        plot_arraysize = int(sample_frequency * 10)  # 10-second buffer
        plotter = Plotter(NUM_PLOT_CHANNELS, plot_arraysize)
        mainLayout.addWidget(plotter.win)
        mainWindow.setLayout(mainLayout)
        mainWindow.show()

        # Define the toggle function for recording.
        def toggle_recording():
            global csv_writer_thread, csv_writer_stop_event, RECORDING
            if not RECORDING:
                # Start recording and save it inside the "data" folder (which is created if it does not exist)
                os.makedirs("data", exist_ok=True)
                filename = os.path.join("data", datetime.datetime.now().strftime("%Y%m%d_%H%M%S.csv"))
                filenameLineEdit.setText(filename)
                statusLabel.setText("Recording...")
                saveButton.setText("Stop")
                csv_writer_stop_event = threading.Event()
                csv_writer_thread = threading.Thread(target=csv_writer_thread_func, args=(filename, csv_writer_stop_event))
                csv_writer_thread.daemon = True
                csv_writer_thread.start()
                RECORDING = True
            else:
                # Stop recording.
                csv_writer_stop_event.set()
                saveButton.setText("Save")
                statusLabel.setText("Not recording")
                RECORDING = False

        saveButton.clicked.connect(toggle_recording)
        sys.exit(app.exec_())
    else:
        # If not plotting, just run indefinitely.
        while True:
            time.sleep(1)

if __name__ == '__main__':
    main()
