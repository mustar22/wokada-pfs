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
| [test1_input](#) | [test1_pfs0](samples/test1_out_pfs0.wav) | [test1_pfs2](samples/test1_out_pfs2.wav) | [test1_pfs5](samples/test1_out_pfs5.wav) |
| [test2_input](#) | [test2_pfs0](samples/test2_out_pfs0.wav) | [test2_pfs2](samples/test2_out_pfs2.wav) | [test2_pfs5](samples/test2_out_pfs5.wav) |

## Installation

### From release (recommended)

Download the latest release from the [Releases](https://github.com/mustar22/wokada-pfs/releases) page.

Download the latest compiled release from the [Releases](https://github.com/mustar22/wokada-pfs/releases) page. *(coming soon)*

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

## Credits

- [w-okada](https://github.com/w-okada/voice-changer) — original voice changer
- [deiteris](https://github.com/deiteris/voice-changer) — fork this mod is based on