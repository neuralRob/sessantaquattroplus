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
PLOT_DATA = True  # Display the real-time plot (for main channels and auxiliary) - se False, il plotter non viene nemmeno inizializzato
# --- NUOVA FLAG GLOBALE: Controlla se il plotting è attualmente attivo ---
PLOT_ACTIVE = True # Se PLOT_DATA è True, il plotting sarà inizialmente attivo

NUM_PLOT_CHANNELS = 6  # Number of main channels to display (the auxiliary channel is plotted separately)
AUX_INDEX = 8          # Index for auxiliary channel (e.g. first auxiliary channel)
ALPHA = 0.01           # Exponential filter coefficient for computing the envelope
track_color = ['k', 'k', 'k', 'k', 'k', 'k', 'k']

# --- Frequenza di trasmissione LSL per l'inviluppo ---
LSL_ENVELOPE_TARGET_RATE_HZ = 100 # Esempio: invia l'inviluppo a 100 Hz

# -----------------------------------------------------------------------------
# Global Queues for inter-thread communication - NOW WITH BOUNDED SIZES
# -----------------------------------------------------------------------------
csv_queue = queue.Queue()
if PLOT_DATA:
    plot_queue = queue.Queue(maxsize=5 * 1000)
    aux_queue = queue.Queue(maxsize=5 * 1000)
trigger_queue = queue.Queue(maxsize=10)

# -----------------------------------------------------------------------------
# Global variables for LSL and control
# -----------------------------------------------------------------------------
global_envelope = None
env_lock = threading.Lock()
lsl_outlet = None
NUM_LSL_CHANNELS = 6
LSL_TRANSMISSION = False
lsl_downsample_factor = 1

# -----------------------------------------------------------------------------
# Reader Thread Function
# -----------------------------------------------------------------------------
def reader_thread_func(socket_conn, num_channels, bytes_in_sample):
    """Continuously reads data from the socket and pushes it to the CSV and (optionally) plot queues.
       Also computes envelope values and, if enabled, sends them via LSL."""
    
    global_lsl_push_counter = 0
    
    while True:
        try:
            sample_bytes = communication.read_raw_bytes(socket_conn, num_channels, bytes_in_sample)
            sample_values = communication.bytes_to_integers(
                sample_bytes, num_channels, bytes_in_sample, output_milli_volts=False)

            # Only put data into the csv_queue if recording is active
            if RECORDING:
                csv_queue.put(sample_values)
            
            # Condiziona l'inserimento nelle code di plotting al flag PLOT_ACTIVE
            if PLOT_DATA and PLOT_ACTIVE:
                try:
                    plot_queue.put_nowait(sample_values[0:NUM_PLOT_CHANNELS])
                except queue.Full:
                    pass

                if len(sample_values) > AUX_INDEX:
                    try:
                        aux_queue.put_nowait(sample_values[AUX_INDEX])
                    except queue.Full:
                        pass

            if LSL_TRANSMISSION and len(sample_values) >= NUM_LSL_CHANNELS:
                env_lock.acquire()
                for i in range(NUM_LSL_CHANNELS):
                    global_envelope[i] = ALPHA * abs(sample_values[i]) + (1 - ALPHA) * global_envelope[i]
                env_lock.release()

                global_lsl_push_counter += 1
                if lsl_downsample_factor == 1 or (global_lsl_push_counter % lsl_downsample_factor == 0):
                    current_env_to_send = global_envelope.copy()
                    lsl_outlet.push_sample(current_env_to_send)
                    if lsl_downsample_factor > 1:
                        global_lsl_push_counter = 0
                        
        except Exception as e:
            print(f"Error in reader_thread_func: {e}")
            time.sleep(0.1)

# -----------------------------------------------------------------------------
# CSV Writer Thread Function (NO CHANGES)
# -----------------------------------------------------------------------------
def csv_writer_thread_func(csv_filename, stop_event):
    """Continuously takes samples from the csv_queue and writes them to a CSV file in batches."""
    BATCH_SIZE = 100
    samples_batch = []
    last_flush_time = time.time()
    FLUSH_INTERVAL = 1.0

    print(f"CSV Writer started. Batch size: {BATCH_SIZE}, Flush interval: {FLUSH_INTERVAL}s")

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        first_item = True

        while not stop_event.is_set() or samples_batch:
            try:
                sample = csv_queue.get(timeout=0.1)
                samples_batch.append(sample)
                csv_queue.task_done()

            except queue.Empty:
                if stop_event.is_set() and not samples_batch:
                    break
                if samples_batch and time.time() - last_flush_time > FLUSH_INTERVAL:
                    if first_item and samples_batch:
                        header = [f"Channel{i+1}" for i in range(len(samples_batch[0]))]
                        writer.writerow(header)
                        first_item = False
                    writer.writerows(samples_batch)
                    csvfile.flush()
                    samples_batch = []
                    last_flush_time = time.time()
                continue

            if len(samples_batch) >= BATCH_SIZE:
                if first_item:
                    header = [f"Channel{i+1}" for i in range(len(samples_batch[0]))]
                    writer.writerow(header)
                    first_item = False
                writer.writerows(samples_batch)
                samples_batch = []

                current_time = time.time()
                if current_time - last_flush_time > FLUSH_INTERVAL:
                    csvfile.flush()
                    last_flush_time = current_time

        if samples_batch:
            print(f"CSV writing final batch of {len(samples_batch)}...")
            if first_item:
                 header = [f"Channel{i+1}" for i in range(len(samples_batch[0]))]
                 writer.writerow(header)
            writer.writerows(samples_batch)
            csvfile.flush()

    print("CSV Writer finished.")

# -----------------------------------------------------------------------------
# LSL Trigger Listener Thread (NO CHANGES)
# -----------------------------------------------------------------------------
def lsl_trigger_listener_thread(q_):
    """Listens for LSL triggers ('ActionTriggers') and puts them onto the queue."""
    print("Looking for LSL trigger stream 'ActionTriggers'...")
    try:
        streams = resolve_byprop('name', 'ActionTriggers', timeout=5)
        if not streams:
            print("LSL trigger stream 'ActionTriggers' not found. Triggering via LSL disabled.")
            return

        inlet = StreamInlet(streams[0])
        print("✓ LSL Trigger stream resolved and inlet opened. Listening for triggers...")

        while True:
            marker, timestamp = inlet.pull_sample(timeout=1.0)
            if marker:
                print(f"Received LSL trigger: {marker[0]} at LSL time {timestamp:.3f}")
                try:
                    q_.put_nowait(marker[0])
                except queue.Full:
                    print("Warning: LSL trigger queue is full, dropping trigger.")

    except Exception as e:
        print(f"Error in LSL trigger listener thread: {e}")

# -----------------------------------------------------------------------------
# Plotter Class (NO CHANGES - il suo timer viene controllato esternamente)
# -----------------------------------------------------------------------------
if PLOT_DATA:
    class Plotter:
        def __init__(self, num_channels, plot_arraysize, sample_frequency):
            self.num_channels = num_channels
            self.arraysize = plot_arraysize
            self.sample_frequency = sample_frequency

            self.win = pg.GraphicsLayoutWidget(title="Real-time EMG Data")
            self.win.resize(1000, 600)

            self.raw_buffers = [np.zeros(plot_arraysize) for _ in range(num_channels)]
            self.env_buffers = [np.zeros(plot_arraysize) for _ in range(num_channels)]
            self.current_env = [0.0] * num_channels

            self.raw_write_indices = [0] * num_channels
            self.env_write_indices = [0] * num_channels

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

            self.aux_buffer = np.zeros(plot_arraysize)
            self.aux_write_index = 0

            p_aux = self.win.addPlot(row=self.num_channels, col=0)
            p_aux.setLabel('left', "Auxiliary")
            p_aux.setLabel('bottom', "Time (s)")
            p_aux.setYRange(-10000, 10000)
            p_aux.setXRange(0, 10)
            self.aux_curve = p_aux.plot(pen=pg.mkPen('b', width=3))
            
            self.timer = QtCore.QTimer() 
            self.timer.timeout.connect(self.update)

        def update(self):
            while not plot_queue.empty():
                try:
                    sample = plot_queue.get_nowait()
                    for i in range(self.num_channels):
                        self.raw_buffers[i][self.raw_write_indices[i]] = sample[i]
                        self.raw_write_indices[i] = (self.raw_write_indices[i] + 1) % self.arraysize

                        self.current_env[i] = ALPHA * abs(sample[i]) + (1 - ALPHA) * self.current_env[i]
                        self.env_buffers[i][self.env_write_indices[i]] = self.current_env[i]
                        self.env_write_indices[i] = (self.env_write_indices[i] + 1) % self.arraysize
                    plot_queue.task_done()
                except queue.Empty:
                    break

            while not aux_queue.empty():
                try:
                    aux_sample = aux_queue.get_nowait()
                    self.aux_buffer[self.aux_write_index] = aux_sample
                    self.aux_write_index = (self.aux_write_index + 1) % self.arraysize
                    aux_queue.task_done()
                except queue.Empty:
                    break

            for i in range(self.num_channels):
                raw_data_to_plot = np.concatenate((self.raw_buffers[i][self.raw_write_indices[i]:],
                                                   self.raw_buffers[i][:self.raw_write_indices[i]]))
                self.raw_curves[i].setData(raw_data_to_plot)

                env_data_to_plot = np.concatenate((self.env_buffers[i][self.env_write_indices[i]:],
                                                   self.env_buffers[i][:self.env_write_indices[i]]))
                self.env_curves[i].setData(env_data_to_plot)

            aux_data_to_plot = np.concatenate((self.aux_buffer[self.aux_write_index:],
                                               self.aux_buffer[:self.aux_write_index]))
            self.aux_curve.setData(aux_data_to_plot)

# -----------------------------------------------------------------------------
# Main Function: Setup Socket, Threads, UI Controls, and LSL Outlet
# -----------------------------------------------------------------------------
def main():
    ip_address = '0.0.0.0'
    port = 45454

    sq_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sq_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sq_socket.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)

    (start_command,
     number_of_channels,
     device_sample_frequency,
     bytes_in_sample) = communication.create_bin_command(start=1)

    print('Starting to log data: {} channels with {} sampling rate'.format(number_of_channels, device_sample_frequency))
    print("number of channels", number_of_channels)

    connection = communication.connect_to_sq(sq_socket, ip_address, port, start_command)

    global lsl_downsample_factor
    if LSL_ENVELOPE_TARGET_RATE_HZ > 0 and device_sample_frequency > LSL_ENVELOPE_TARGET_RATE_HZ:
        lsl_downsample_factor = int(round(device_sample_frequency / LSL_ENVELOPE_TARGET_RATE_HZ))
        actual_lsl_output_rate = device_sample_frequency / lsl_downsample_factor
        print(f"LSL envelope will be downsampled. Device rate: {device_sample_frequency} Hz, Target LSL rate: {LSL_ENVELOPE_TARGET_RATE_HZ} Hz.")
        print(f"Calculated LSL downsample factor: {lsl_downsample_factor}. Actual LSL output rate: {actual_lsl_output_rate:.2f} Hz.")
        lsl_stream_srate = actual_lsl_output_rate
    else:
        lsl_downsample_factor = 1
        lsl_stream_srate = device_sample_frequency
        print(f"LSL envelope will be sent at device's full rate: {lsl_stream_srate} Hz.")


    global global_envelope, lsl_outlet, LSL_TRANSMISSION
    global_envelope = [0.0] * NUM_LSL_CHANNELS
    info = StreamInfo('EMGEnvelope', 'EMG', NUM_LSL_CHANNELS, lsl_stream_srate, 'float32', 'myuid34234')
    lsl_outlet = StreamOutlet(info)
    LSL_TRANSMISSION = False

    reader_thread = threading.Thread(target=reader_thread_func, args=(connection, number_of_channels, bytes_in_sample))
    reader_thread.daemon = True
    reader_thread.start()

    lsl_trigger_thread = threading.Thread(target=lsl_trigger_listener_thread, args=(trigger_queue,))
    lsl_trigger_thread.daemon = True
    lsl_trigger_thread.start()

    global RECORDING, csv_writer_thread, csv_writer_stop_event
    RECORDING = False
    csv_writer_thread = None
    csv_writer_stop_event = None

    if PLOT_DATA:
        app = QtWidgets.QApplication([])
        mainWindow = QtWidgets.QWidget()
        mainLayout = QtWidgets.QVBoxLayout(mainWindow)
        
        controlLayout = QtWidgets.QHBoxLayout()
        filenameLineEdit = QtWidgets.QLineEdit()
        filenameLineEdit.setReadOnly(True)
        saveButton = QtWidgets.QPushButton("Save")
        statusLabel = QtWidgets.QLabel("Not recording")
        lslButton = QtWidgets.QPushButton("Open LSL")
        
        global PLOT_ACTIVE # Dichiara che userai la variabile globale PLOT_ACTIVE
        
        plotToggleButton = QtWidgets.QPushButton("Disable Plot" if PLOT_ACTIVE else "Enable Plot")
        
        controlLayout.addWidget(filenameLineEdit)
        controlLayout.addWidget(saveButton)
        controlLayout.addWidget(statusLabel)
        controlLayout.addWidget(lslButton)
        controlLayout.addWidget(plotToggleButton)
        mainLayout.addLayout(controlLayout)
        
        plot_arraysize = int(device_sample_frequency * 10)
        plotter = Plotter(NUM_PLOT_CHANNELS, plot_arraysize, device_sample_frequency)
        
        # Aggiungi il widget del plotter al layout (la sua visibilità sarà controllata)
        mainLayout.addWidget(plotter.win)
        mainWindow.setLayout(mainLayout)

        # --- Gestisci lo stato iniziale del plotter e della finestra ---
        if PLOT_ACTIVE:
            plotter.win.show()
            plotter.timer.start(50) 
        else:
            plotter.win.hide()
            plotter.timer.stop()
        
        # Dopo aver impostato la visibilità iniziale, regola la dimensione della finestra
        mainWindow.adjustSize() # Permette al layout di calcolare la dimensione ideale
        mainWindow.show() # Mostra la finestra

        # --- Funzione per abilitare/disabilitare il plot e ridimensionare la finestra ---
        def toggle_plot():
            global PLOT_ACTIVE
            if PLOT_ACTIVE: # Il plot è attualmente attivo, disabilitalo
                PLOT_ACTIVE = False
                plotter.win.hide()       # Nasconde il widget del plot
                plotter.timer.stop()     # Ferma il timer di aggiornamento
                plotToggleButton.setText("Enable Plot")
                print("Plotting disabled. Adjusting window size.")

                # Permetti al layout di ricalcolare le dimensioni, poi ridimensiona la finestra
                mainWindow.adjustSize() 

                # Svuota le code di plotting per non accumulare dati inutilmente mentre il plot è nascosto
                while not plot_queue.empty():
                    try: plot_queue.get_nowait()
                    except queue.Empty: break
                while not aux_queue.empty():
                    try: aux_queue.get_nowait()
                    except queue.Empty: break

            else: # Il plot è attualmente inattivo, abilitalo
                PLOT_ACTIVE = True
                plotter.win.show()       # Mostra il widget del plot
                plotter.timer.start(50)  # Avvia il timer di aggiornamento
                plotToggleButton.setText("Disable Plot")
                print("Plotting enabled. Adjusting window size.")

                # Permetti al layout di ricalcolare le dimensioni, poi ridimensiona la finestra
                mainWindow.adjustSize() 

        plotToggleButton.clicked.connect(toggle_plot)


        def toggle_recording(force_state=None):
            global csv_writer_thread, csv_writer_stop_event, RECORDING

            should_be_recording = not RECORDING
            if force_state == 'start':
                should_be_recording = True
            elif force_state == 'stop':
                should_be_recording = False

            if should_be_recording and not RECORDING:
                print("Starting recording...")
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
                print("Stopping recording...")
                if csv_writer_stop_event:
                    csv_writer_stop_event.set()
                saveButton.setText("Save")
                statusLabel.setText("Not recording")
                RECORDING = False

        saveButton.clicked.connect(lambda: toggle_recording())

        def check_lsl_triggers():
            try:
                trigger_marker = trigger_queue.get_nowait()
                print(f"Processing LSL trigger: {trigger_marker}")

                if isinstance(trigger_marker, str):
                    if trigger_marker.lower() == 'start':
                        toggle_recording(force_state='start')
                    elif trigger_marker.lower() == 'stop':
                        toggle_recording(force_state='stop')
                    else:
                        print(f"Ignoring unknown LSL trigger: {trigger_marker}")

            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error processing trigger queue: {e}")

        trigger_check_timer = QtCore.QTimer()
        trigger_check_timer.timeout.connect(check_lsl_triggers)
        trigger_check_timer.start(50)

        def toggle_lsl():
            global LSL_TRANSMISSION
            if not LSL_TRANSMISSION:
                LSL_TRANSMISSION = True
                lslButton.setText("Close LSL")
                print("LSL EMGEnvelope stream opened.")
            else:
                LSL_TRANSMISSION = False
                lslButton.setText("Open LSL")
                print("LSL EMGEnvelope stream closed.")

        lslButton.clicked.connect(toggle_lsl)

        sys.exit(app.exec_())

    else:
        while True:
            time.sleep(1)

if __name__ == '__main__':
    main()