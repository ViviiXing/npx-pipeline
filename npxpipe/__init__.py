"""npxpipe: Neuropixels 1.0 (Open Ephys) -> SpikeInterface -> Kilosort4 -> Phy pipeline.

Modules
-------
config    : all user-adjustable settings in one place
utils     : logging, console prompts, small helpers
session   : finds probes/streams in an Open Ephys session and keeps the pipeline state
qc        : data inspection and bad-channel detection (with figures)
sorting   : sorting-input chain, Kilosort4 runs, result placement and checks
drift     : drift figures and the per-unit residual-drift check
curation  : Phy helpers (launch, path repair, read-back check after manual curation)
postprocess: waveforms, metrics, unit locations, putative cell types (after curation)
"""
__version__ = "1.2.0"
