# BCI Speller — Dual-Paradigm (P300 + SSVEP) Production System

## Repository Architecture

```
bci-speller/
├── paradigms/
│   ├── p300/              # ERP speller: epoching, xDAWN, LDA classification
│   │   ├── __init__.py
│   │   ├── epoching.py
│   │   ├── xdawn.py
│   │   └── classifier.py
│   └── ssvep/              # CCA/FBCCA + CNN baseline, frequency stimulus generation
│       ├── __init__.py
│       ├── cca.py
│       ├── fbcca.py
│       ├── cnn_baseline.py
│       └── stimulus.py
│
├── spatial_filters/         # Shared spatial filtering utilities (covariance, Riemannian tools)
│   └── __init__.py
│
├── hardware/                 # Hardware Abstraction Layer (HAL)
│   ├── config/
│   │   └── config.yaml       # Declares headset, channel map, sample rate, LSL props
│   ├── drivers/
│   │   ├── lsl_discovery.py  # resolve_byprop auto-discovery
│   │   ├── resampler.py      # sample-rate harmonization
│   │   └── channel_mapper.py # dynamic 10-20 channel auto-mapping
│   └── playback.py           # Dataset mock-playback engine (hardware-free dev)
│
├── ui/                       # Speller UI: P300 flash matrix, SSVEP flicker targets
│   └── __init__.py
│
├── llm/                      # Local LLM next-word prediction
│   └── __init__.py
│
├── os_hooks/                 # PyAutoGUI system keypress injection
│   └── __init__.py
│
├── scripts/                  # One-off verification / diagnostic scripts (this phase's output)
│   └── verify_display_and_permissions.py
│
├── data/
│   └── mock_recordings/      # Local cached datasets (BCI Comp III, Wang2016) — gitignored
│
├── tests/                    # Unit tests, mirrors package structure
│
├── docs/                     # Architecture diagrams, benchmark tables (Phase 8)
│
├── requirements.txt
├── .gitignore
└── README.md
```

**Design rationale:**
- `paradigms/p300` and `paradigms/ssvep` are siblings, not shared — their epoching, filtering, and classification math diverge enough (ERP time-locking vs frequency-domain CCA) that a shared module would become an if/else maze.
- `hardware/` is the only layer that knows about LSL, device names, or channel counts. Everything above it (paradigms, ui) consumes clean numpy arrays + metadata — this is what makes OpenBCI/Muse/Ganglion/dataset-playback interchangeable via YAML alone.
- `llm/` and `os_hooks/` are isolated because they're the two subsystems most likely to need swapping (different local model backends, different OS injection strategies on Windows/Linux later) — isolating them keeps that swap a one-file change.
