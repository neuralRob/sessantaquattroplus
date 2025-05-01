# Sessantaquattro+
This repo contains python script to stream real time raw data (2kHz) from the OTB Sessantaquattro+ data logger. It provides real time plotting, trigger data save (either by pressing a button, or from external scripts via an LSL trigger), send out EMG envelopes via LSL.

# Real-Time EMG / Aux-Channel Logger & LSL Bridge  
*(`sq_emg_logger.py` – working title)*  

A single-script utility that

* streams raw Sessantaquattro EMG data over TCP,  
* shows it live (per-channel traces + envelope) in a PyQtGraph window,  
* records to CSV on demand,  
* broadcasts an envelope stream over **LSL**, and  
* reacts to external “start/stop” LSL triggers.

---

## Quick start

```bash
python sq_emg_logger.py           # requires Python ≥3.9
```

### Hot-keys & GUI buttons

| Control              | Action |
|----------------------|--------|
| **Save** / **Stop**  | start / stop CSV recording |
| **Open LSL** / **Close LSL** | toggle envelope transmission on `EMGEnvelope` stream |
| **LSL trigger** `"start"` | starts recording (remote) |
| **LSL trigger** `"stop"`  | stops recording  |

Recorded files land in `./data/` with a timestamped filename  
(e.g. `20250501_141732.csv`).

---

## Dependencies

```
numpy  •  pyqtgraph  •  PySide6 | PyQt5
pylsl  •  communication_sessantaquattro_plus
```

Install them with:

```bash
pip install numpy pyqtgraph pylsl
# plus whatever Qt binding you prefer and your communication_* package
```

---

## Main building blocks

| Module / object | Purpose |
|-----------------|---------|
| **`reader_thread_func`** | pulls raw samples from the SQ device → queues (plot / CSV) → updates exponential envelope → pushes envelope to LSL (if enabled). |
| **`csv_writer_thread_func`** | drains `csv_queue` → writes to a rolling CSV file until signalled to stop. |
| **`lsl_trigger_listener_thread`** | resolves `ActionTriggers` marker stream → enqueues received markers for the GUI timer to inspect. |
| **`Plotter` class** | PyQtGraph live display (N main channels + 1 auxiliary) with 10 s rolling buffers and per-channel envelope overlay. |
| **Qt **GUI** | minimal control panel (filename field, record toggle, LSL toggle) + stacked plots. |
| **Global queues** | decouple threads: `csv_queue`, `plot_queue`, `aux_queue`, `trigger_queue`. |

### LSL streams

| Direction | Stream name | Channels | Format | Sent when |
|-----------|-------------|----------|--------|-----------|
| **out**   | `EMGEnvelope` | 8 | `float32` | every sample in reader thread (if transmission enabled) |
| **in**    | `ActionTriggers` (type `Markers`) | 1 | `string` | listener thread waits & forwards to GUI |

---

## Configuration flags

At the top of the script:

```python
PLOT_DATA           = True   # disable to run headless
NUM_PLOT_CHANNELS   = 6      # traces drawn in GUI
AUX_INDEX           = 8      # index of auxiliary channel
ALPHA               = 0.01   # envelope low-pass factor
NUM_LSL_CHANNELS    = 8      # first N channels → envelope stream
```

Change these constants or expose them as command-line arguments to suit your setup.

---

## Thread diagram

```
┌──────────┐  raw bytes          ┌─────────────────┐
│ Sessanta │───TCP──────────────▶│ reader thread   │
│ quattro  │                     └─┬────┬─────┬────┘
└──────────┘                       │    │     │
                                   │    │     └───► LSL: EMGEnvelope
                                   │    │
                                   │    └────────► plot_queue (GUI)
                                   │
                                   └─────────────► csv_queue (writer)

┌───────────────────────┐
│ lsl_trigger_listener  │─────▶ trigger_queue ───▶ Qt timer → toggle rec.
└───────────────────────┘

csv_queue ──▶ csv_writer_thread ──► disk (CSV)
```

---

## Extending / customizing

* **Additional LSL metadata** – fill in `info.desc()` before creating the outlet.  
* **More trigger words** – edit `check_lsl_triggers()` to map markers to actions.  
* **Headless mode** – set `PLOT_DATA = False`; plotting queues and GUI are skipped.  
* **Different device** – replace calls to `communication_*` with your own I/O shim.

---