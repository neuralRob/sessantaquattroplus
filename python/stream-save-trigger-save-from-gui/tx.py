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
from pylsl import StreamInfo, StreamOutlet, resolve_byprop, StreamInlet  # LSL imports

# -----------------------------------------------------------------------------
# Global Plot Config: white background and black foreground.
# -----------------------------------------------------------------------------
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# -----------------------------------------------------------------------------
# Configuration Flags and Parameters
# -----------------------------------------------------------------------------
PLOT_DATA = True  # Display the real-time plot (for main channels and auxiliary)
NUM_PLOT_CHANNELS = 6  # Number of main channels to display (the auxiliary channel is plotted separately)
AUX_INDEX = 8          # Index for auxiliary channel (e.g. first auxiliary channel)
ALPHA = 0.01           # Exponential filter coefficient for computing the envelope
track_color = ['k', 'k', 'k', 'k', 'k', 'k', 'k']

# -----------------------------------------------------------------------------
# Global Queues for inter-thread communication
# -----------------------------------------------------------------------------
csv_queue = queue.Queue()
if PLOT_DATA:
    plot_queue = queue.Queue()
    aux_queue = queue.Queue()  # For the auxiliary channel
trigger_queue = queue.Queue() # For LSL triggers from GUI

# -----------------------------------------------------------------------------
# Global variables for LSL and control
# -----------------------------------------------------------------------------
global_envelope = None    # Will be initialized in main (list of length NUM_LSL_CHANNELS)
env_lock = threading.Lock()
lsl_outlet = None         # LSL outlet will be created in main
NUM_LSL_CHANNELS = 8      # Send envelope for first 8 channels
LSL_TRANSMISSION = False  # Global flag to toggle LSL transmission

# -----------------------------------------------------------------------------
# Reader Thread Function
# -----------------------------------------------------------------------------
def reader_thread_func(socket_conn, num_channels, bytes_in_sample):
    """Continuously reads data from the socket and pushes it to the CSV and (optionally) plot queues.
       Also computes envelope values and, if enabled, sends them via LSL."""
    while True:
        sample_bytes = communication.read_raw_bytes(socket_conn, num_channels, bytes_in_sample)
        sample_values = communication.bytes_to_integers(
            sample_bytes, num_channels, bytes_in_sample, output_milli_volts=False)
        # Only put data into the csv_queue if recording is active
        if RECORDING:
            csv_queue.put(sample_values)
        if PLOT_DATA:
            # Push main channels (first NUM_PLOT_CHANNELS) to plot queue.
            plot_queue.put(sample_values[0:NUM_PLOT_CHANNELS])
            # Push the auxiliary channel (from AUX_INDEX) if available.
            if len(sample_values) > AUX_INDEX:
                aux_queue.put(sample_values[AUX_INDEX])
        # If LSL transmission is enabled, compute envelope for first NUM_LSL_CHANNELS and send.
        if LSL_TRANSMISSION and len(sample_values) >= NUM_LSL_CHANNELS:
            env_lock.acquire()
            for i in range(NUM_LSL_CHANNELS):
                global_envelope[i] = ALPHA * abs(sample_values[i]) + (1 - ALPHA) * global_envelope[i]
            current_env = global_envelope.copy()
            env_lock.release()
            lsl_outlet.push_sample(current_env)

# -----------------------------------------------------------------------------
# CSV Writer Thread Function (supports stopping via stop_event)
# -----------------------------------------------------------------------------
def csv_writer_thread_func(csv_filename, stop_event):
    """Continuously takes samples from the csv_queue and writes them to a CSV file in batches."""
    BATCH_SIZE = 100 # Write 100 samples at a time (adjust as needed)
    samples_batch = []
    last_flush_time = time.time()
    FLUSH_INTERVAL = 1.0 # Flush buffer every 1 second (adjust as needed)

    print(f"CSV Writer started. Batch size: {BATCH_SIZE}, Flush interval: {FLUSH_INTERVAL}s")

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        first_item = True

        while not stop_event.is_set() or samples_batch: # Process remaining batch even after stop event
            try:
                # Get a sample, wait longer if needed, but check stop_event often
                sample = csv_queue.get(timeout=0.1)
                samples_batch.append(sample)
                csv_queue.task_done() # Notify queue item is processed

            except queue.Empty:
                # Queue is empty, check if we should stop or flush
                if stop_event.is_set() and not samples_batch:
                    break # Stop event set and batch is empty, exit loop
                # Check if it's time to flush remaining items
                if samples_batch and time.time() - last_flush_time > FLUSH_INTERVAL:
                    # print(f"CSV flushing {len(samples_batch)} items due to timeout...") # Debug
                    if first_item and samples_batch:
                        header = [f"Channel{i+1}" for i in range(len(samples_batch[0]))]
                        writer.writerow(header)
                        first_item = False
                    writer.writerows(samples_batch)
                    csvfile.flush() # Flush the OS buffer
                    samples_batch = []
                    last_flush_time = time.time()
                continue # Go back to waiting for samples

            # Write batch if full
            if len(samples_batch) >= BATCH_SIZE:
                # print(f"CSV writing batch of {len(samples_batch)}...") # Debug
                if first_item:
                    header = [f"Channel{i+1}" for i in range(len(samples_batch[0]))]
                    writer.writerow(header)
                    first_item = False
                writer.writerows(samples_batch)
                samples_batch = [] # Clear the batch

                # Optional: Flush periodically based on time, not just on batch size or stop
                current_time = time.time()
                if current_time - last_flush_time > FLUSH_INTERVAL:
                    # print("CSV flushing buffer...") # Debug
                    csvfile.flush()
                    last_flush_time = current_time

        # --- After loop ---
        # Write any remaining items in the batch when stopping
        if samples_batch:
            print(f"CSV writing final batch of {len(samples_batch)}...")
            if first_item: # Handle case where file might be empty
                 header = [f"Channel{i+1}" for i in range(len(samples_batch[0]))]
                 writer.writerow(header)
            writer.writerows(samples_batch)
            csvfile.flush() # Final flush

    print("CSV Writer finished.")

def lsl_trigger_listener_thread(q_):
    """Listens for LSL triggers ('ActionTriggers') and puts them onto the queue."""
    print("Looking for LSL trigger stream 'ActionTriggers'...")
    try:
        # Resolve the stream (add a timeout to avoid blocking indefinitely if not found) -- controlla qui ••
        streams = resolve_byprop('name', 'ActionTriggers', timeout=5)
        if not streams:
            print("LSL trigger stream 'ActionTriggers' not found. Triggering via LSL disabled.")
            return # Exit the thread if stream not found

        # Create an inlet
        inlet = StreamInlet(streams[0])
        print("✓ LSL Trigger stream resolved and inlet opened. Listening for triggers...")

        while True:
            # Pull a sample (blocking call, use a timeout if you want the thread to be interruptible)
            marker, timestamp = inlet.pull_sample(timeout=1.0) # Using timeout allows checking loop conditions if needed later
            if marker:
                # Put the received marker content (e.g., the string 'start' or 'stop') onto the queue
                print(f"Received LSL trigger: {marker[0]} at LSL time {timestamp:.3f}") # Optional: Log received triggers
                q_.put(marker[0]) # We only need the marker content for control

    except Exception as e:
        print(f"Error in LSL trigger listener thread: {e}")
        # You might want more specific error handling (e.g., for pylsl.LSLError)

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
            # Buffers for main channels: raw data and envelope.
            self.raw_buffers = [np.zeros(plot_arraysize) for _ in range(num_channels)]
            self.env_buffers = [np.zeros(plot_arraysize) for _ in range(num_channels)]
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
            # Add an extra plot for the auxiliary channel.
            self.aux_buffer = np.zeros(plot_arraysize)
            p_aux = self.win.addPlot(row=self.num_channels, col=0)
            p_aux.setLabel('left', "Auxiliary")
            p_aux.setLabel('bottom', "Time (s)")
            p_aux.setYRange(-10000, 10000)
            p_aux.setXRange(0, 10)
            self.aux_curve = p_aux.plot(pen=pg.mkPen('b', width=3))
            self.win.show()
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update)
            self.timer.start(50)  # Update every 10 ms.

        def update(self):
            # Process main channels.
            while not plot_queue.empty():
                sample = plot_queue.get_nowait()
                for i in range(self.num_channels):
                    self.raw_buffers[i] = np.roll(self.raw_buffers[i], -1)
                    self.raw_buffers[i][-1] = sample[i]
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
# Main Function: Setup Socket, Threads, UI Controls, and LSL Outlet
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

    # Initialize global envelope for LSL streaming.
    global global_envelope, lsl_outlet, LSL_TRANSMISSION
    global_envelope = [0.0] * NUM_LSL_CHANNELS
    info = StreamInfo('EMGEnvelope', 'EMG', NUM_LSL_CHANNELS, sample_frequency, 'float32', 'myuid34234')
    lsl_outlet = StreamOutlet(info)
    LSL_TRANSMISSION = False  # Initially off.

    # Start the reader thread.
    reader_thread = threading.Thread(target=reader_thread_func, args=(connection, number_of_channels, bytes_in_sample))
    reader_thread.daemon = True
    reader_thread.start()

    # Start the LSL trigger listener thread <--- ADD THIS
    lsl_trigger_thread = threading.Thread(target=lsl_trigger_listener_thread, args=(trigger_queue,))
    lsl_trigger_thread.daemon = True
    lsl_trigger_thread.start()

    # Global variables for CSV recording control.
    global RECORDING, csv_writer_thread, csv_writer_stop_event
    RECORDING = False
    csv_writer_thread = None
    csv_writer_stop_event = None

    if PLOT_DATA:
        app = QtWidgets.QApplication([])
        mainWindow = QtWidgets.QWidget()
        mainLayout = QtWidgets.QVBoxLayout(mainWindow)
        # --- Control Panel ---
        controlLayout = QtWidgets.QHBoxLayout()
        filenameLineEdit = QtWidgets.QLineEdit()
        filenameLineEdit.setReadOnly(True)
        saveButton = QtWidgets.QPushButton("Save")
        statusLabel = QtWidgets.QLabel("Not recording")
        lslButton = QtWidgets.QPushButton("Open LSL")
        controlLayout.addWidget(filenameLineEdit)
        controlLayout.addWidget(saveButton)
        controlLayout.addWidget(statusLabel)
        controlLayout.addWidget(lslButton)
        mainLayout.addLayout(controlLayout)
        # --- Plot Area ---
        plot_arraysize = int(sample_frequency * 10)  # 10-second buffer
        plotter = Plotter(NUM_PLOT_CHANNELS, plot_arraysize)
        mainLayout.addWidget(plotter.win)
        mainWindow.setLayout(mainLayout)
        mainWindow.show()

        # Toggle function for CSV recording.
        # Inside main(), within the 'if PLOT_DATA:' block where toggle_recording is defined:
        def toggle_recording(force_state=None): # Add optional argument
            global csv_writer_thread, csv_writer_stop_event, RECORDING

            # Determine the target state
            should_be_recording = not RECORDING # Default: toggle
            if force_state == 'start':
                should_be_recording = True
            elif force_state == 'stop':
                should_be_recording = False

            # Act only if the state needs to change
            if should_be_recording and not RECORDING:
                # --- Start Recording --- (Keep existing logic)
                print("Starting recording...") # Add feedback if desired
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
            elif not should_be_recording and RECORDING:
                # --- Stop Recording --- (Keep existing logic)
                print("Stopping recording...") # Add feedback if desired
                if csv_writer_stop_event: # Check if event exists before setting
                    csv_writer_stop_event.set()
                # Optional: you might want to wait for the thread to finish writing
                # if csv_writer_thread:
                #    csv_writer_thread.join(timeout=0.5)
                saveButton.setText("Save")
                statusLabel.setText("Not recording")
                RECORDING = False

        # Update the button connection to call the modified function without force_state (maintaining toggle behavior for button)
        saveButton.clicked.connect(lambda: toggle_recording()) # Use lambda to call without arguments

        # --- Function to Check LSL Triggers ---
        def check_lsl_triggers():
            try:
                # Check the queue without blocking
                trigger_marker = trigger_queue.get_nowait()
                print(f"Processing LSL trigger: {trigger_marker}") # Optional debug

                # Act based on the trigger content (case-insensitive comparison)
                if isinstance(trigger_marker, str): # Basic check
                    if trigger_marker.lower() == 'start':
                        toggle_recording(force_state='start')
                    elif trigger_marker.lower() == 'stop':
                        toggle_recording(force_state='stop')
                    else:
                        # Optional: Define behavior for other triggers (e.g., ignore, toggle)
                        print(f"Ignoring unknown LSL trigger: {trigger_marker}")

            except queue.Empty:
                # No trigger waiting in the queue, do nothing
                pass
            except Exception as e:
                # Log errors during trigger processing
                print(f"Error processing trigger queue: {e}")

        # --- Timer for Checking LSL Triggers ---
        trigger_check_timer = QtCore.QTimer()
        trigger_check_timer.timeout.connect(check_lsl_triggers)
        trigger_check_timer.start(50)  # Check for triggers every 100 ms

        # Toggle function for LSL transmission.
        def toggle_lsl():
            global LSL_TRANSMISSION
            if not LSL_TRANSMISSION:
                LSL_TRANSMISSION = True
                lslButton.setText("Close LSL")
            else:
                LSL_TRANSMISSION = False
                lslButton.setText("Open LSL")

        lslButton.clicked.connect(toggle_lsl)

        # --- Start the Application ---
        mainWindow.show() # Ensure window is shown before exec_
        sys.exit(app.exec_())

    else:
        while True:
            time.sleep(1)

if __name__ == '__main__':
    main()
