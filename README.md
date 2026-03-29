# wokada-pfs

> Mod of [Deiteris' w-okada voice changer fork](https://github.com/deiteris/voice-changer) with pre-ContentVec spectral envelope shift for improved M2F voice conversion.

## What's new

Standard RVC forks apply formant shift **after** VITS synthesis. This mod applies a spectral envelope nudge **before** ContentVec feature extraction — reducing the acoustic distance the model needs to bridge for male-to-female conversion.

**Pipeline comparison:**

Original:
```
Audio → F0 Extract → ContentVec → FAISS → VITS → Formant Shift → Output
```

wokada-pfs:
```
Audio → F0 Extract → [PRE FORMANT SHIFT] → ContentVec → FAISS → VITS → Formant Shift → Output
```

**PRE F.SHIFT** parameter added to UI (0–5 semitones). Recommended starting point: 2.0. No quality degradation observed across full range.

Also includes PyTorch 2.6 compatibility fix (`weights_only=False`) for all `torch.load` calls.

## Audio samples

| Input | PFS 0 (baseline) | PFS 2 (recommended) | PFS 5 |
|-------|-----------------|---------------------|-------|
| — | [test1_pfs0](samples/test_out_pfs0.wav) | [test1_pfs2](samples/test_out_pfs2.wav) | [test1_pfs5](samples/test_out_pfs5.wav) |
| [test2_input](samples/test2_input.wav) | [test2_pfs0](samples/test2_out_pfs0.wav) | [test2_pfs2](samples/test2_out_pfs2.wav) | [test2_pfs5](samples/test2_out_pfs5.wav) |

## Installation

### Portable (recommended for most users)

No Python required. Extract and run.

**[📥 Download wokada-pfs-portable.zip](https://huggingface.co/MUSTAR/wokada-pfs-prebuilt/resolve/main/wokada-pfs-portable.zip?download=true)**

1. Extract the zip
2. Run `install.bat` once to install dependencies
3. Run `start.bat` to launch

Requires [VAC Lite by Muzychenko](https://software.muzychenko.net/freeware/vac470lite.zip).

### Standalone (for users with Python 3.10 already installed)

**[📥 Download wokada-pfs.zip](https://huggingface.co/MUSTAR/wokada-pfs-prebuilt/resolve/main/wokada-pfs.zip?download=true)**

1. Extract the zip
2. Install dependencies: `pip install -r server/requirements-common.txt -r server/requirements-cuda.txt`
3. Run `start.bat` to launch

Requires [VAC Lite by Muzychenko](https://software.muzychenko.net/freeware/vac470lite.zip).

### From source

Requires Python 3.10.
```bash
git clone https://github.com/mustar22/wokada-pfs
cd wokada-pfs/server
pip install -r requirements-common.txt -r requirements-cuda.txt
cd ../client/demo
npm install
npm run build:mod_dos
npm run build:dev
cd ../../server
python main.py
```

## How it works

RVC's ContentVec embedder is supposed to extract speaker-independent phonetic features, but in practice it leaks — strongly male inputs produce embeddings with residual male timbral characteristics. FAISS retrieval then finds suboptimal matches in the female voice index, and the decoder has to compensate harder, producing less natural output.

The idea came from working with WAN 2.2 video generation — specifically the observation that pre-conditioning input toward the target domain before feature extraction produces smoother, more natural results than letting the model bridge the full distance alone. Applied to voice: instead of feeding raw male audio directly into ContentVec, nudge it toward female acoustic territory first.

We apply an STFT-based spectral envelope stretch to the 16kHz audio **before** ContentVec sees it, shifting formant energy upward toward female characteristics. F0 extraction happens before the nudge and is untouched. Phase is preserved throughout — only the magnitude spectrum is stretched.

The result: ContentVec receives a signal that's acoustically closer to female speech, finds better FAISS matches, and the decoder synthesizes more convincing output with less strain. Tested clean across the full 0–5 semitone range with no observed quality degradation on two independent setups (RTX 4070 Ti and RTX 5070).


## Credits

- [w-okada](https://github.com/w-okada/voice-changer) — original voice changer
- [deiteris](https://github.com/deiteris/voice-changer) — fork this mod is based on