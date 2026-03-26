from faiss import IndexIVFFlat
import faiss.contrib.torch_utils

from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info, make_opsetid
)

import numpy as np
import sys
import torch
import torch.nn.functional as F
import onnxruntime
from torchaudio import transforms as tat
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
import logging

from voice_changer.RVC.consts import HUBERT_SAMPLE_RATE, WINDOW_SIZE
from voice_changer.common.TorchUtils import circular_write
from voice_changer.embedder.Embedder import Embedder
from voice_changer.RVC.inferencer.Inferencer import Inferencer

from voice_changer.pitch_extractor.PitchExtractor import PitchExtractor
from voice_changer.utils.Timer import Timer2
from const import F0_MEL_MIN, F0_MEL_MAX

logger = logging.getLogger(__name__)


class Pipeline:
    embedder: Embedder
    inferencer: Inferencer
    pitchExtractor: PitchExtractor

    index: IndexIVFFlat | None
    index_reconstruct: torch.Tensor | None

    model_sr: int
    device: torch.device
    isHalf: bool

    def __init__(
        self,
        embedder: Embedder,
        inferencer: Inferencer,
        pitchExtractor: PitchExtractor,
        index: IndexIVFFlat | None,
        index_reconstruct: torch.Tensor | None,
        use_f0: bool,
        model_sr: int,
        embChannels: int,
    ):
        self.embedder = embedder
        self.inferencer = inferencer
        self.pitchExtractor = pitchExtractor
        logger.info("GENERATE INFERENCER" + str(self.inferencer))
        logger.info("GENERATE EMBEDDER" + str(self.embedder))
        logger.info("GENERATE PITCH EXTRACTOR" + str(self.pitchExtractor))

        self.device_manager = DeviceManager.get_instance()
        self.device = self.device_manager.device
        self.is_half = self.device_manager.use_fp16()

        self.index = index
        self.index_reconstruct: torch.Tensor | None = index_reconstruct
        self.use_index = index is not None and self.index_reconstruct is not None
        self.use_gpu_index = sys.platform == 'linux' and '+cu' in torch.__version__ and self.device.type == 'cuda'
        self.use_f0 = use_f0

        self.onnx_upscaler = self.make_onnx_upscaler(embChannels) if self.device.type == 'privateuseone' else None

        self.model_sr = model_sr
        self.model_window = model_sr // 100

        self.dtype = torch.float16 if self.is_half else torch.float32

        self.resamplers = {}

        # Cache STFT window for pre-formant nudge to avoid reallocating every chunk
        self._stft_window: torch.Tensor | None = None
        self._stft_n_fft: int = 512
        self._stft_hop: int = 160   # 10ms at 16kHz
        self._stft_win: int = 512

    def make_onnx_upscaler(self, dim_size: int):
        # Inputs
        input = make_tensor_value_info('in', TensorProto.FLOAT16 if self.is_half else TensorProto.FLOAT, [1, dim_size, None])
        scales = make_tensor_value_info('scales', TensorProto.FLOAT, [None])
        # Outputs
        output = make_tensor_value_info('out', TensorProto.FLOAT16 if self.is_half else TensorProto.FLOAT, [1, dim_size, None])

        resize_node = make_node(
            "Resize",
            inputs=["in", "", "scales"],
            outputs=["out"],
            mode="nearest",
            axes=[2]
        )

        graph = make_graph([resize_node], 'upscaler', [input, scales], [output])

        onnx_model = make_model(graph, opset_imports=[make_opsetid("", 21)])

        (
            providers,
            provider_options,
        ) = self.device_manager.get_onnx_execution_provider()
        return onnxruntime.InferenceSession(onnx_model.SerializeToString(), providers=providers, provider_options=provider_options)

    def getPipelineInfo(self):
        inferencerInfo = self.inferencer.getInferencerInfo() if self.inferencer else {}
        embedderInfo = self.embedder.get_embedder_info()
        pitchExtractorInfo = self.pitchExtractor.getPitchExtractorInfo()
        return {"inferencer": inferencerInfo, "embedder": embedderInfo, "pitchExtractor": pitchExtractorInfo}

    def setPitchExtractor(self, pitchExtractor: PitchExtractor):
        self.pitchExtractor = pitchExtractor

    def _get_stft_window(self) -> torch.Tensor:
        """Lazily initialize and cache the STFT window on the correct device."""
        if self._stft_window is None or self._stft_window.device != self.device:
            self._stft_window = torch.hann_window(self._stft_win, device=self.device)
        return self._stft_window

    def _pre_formant_nudge(self, audio: torch.Tensor, semitones: float) -> torch.Tensor:
        """
        Pre-ContentVec spectral envelope nudge for M2F preprocessing.

        Gently shifts the spectral envelope toward female voice characteristics
        BEFORE feature extraction by ContentVec/HuBERT. This reduces the acoustic
        distance the model needs to bridge during conversion.

        F0 (fundamental frequency / pitch) is NOT touched here — it is handled
        entirely by the existing pitch pipeline (extract_pitch + f0_up_key).
        Only the spectral envelope shape is modified.

        Positive semitones = shift toward higher formants = female direction.
        Recommended range: 0.5 to 3.0 semitones for M2F.
        At 0.0 this function is never called (skipped in exec).

        Args:
            audio:    1D tensor, 16kHz, fp16 or fp32, on GPU
            semitones: shift strength in semitones (float, positive = female direction)

        Returns:
            Processed audio tensor — same shape, dtype, and device as input.
        """
        original_dtype = audio.dtype

        # STFT requires float32
        audio_f32 = audio.float()

        window = self._get_stft_window()

        # Forward STFT — complex output shape: [freq_bins, time_frames]
        stft = torch.stft(
            audio_f32,
            n_fft=self._stft_n_fft,
            hop_length=self._stft_hop,
            win_length=self._stft_win,
            window=window,
            return_complex=True,
        )

        magnitude = stft.abs()       # [freq_bins, time_frames]
        phase = stft.angle()         # [freq_bins, time_frames]

        freq_bins = magnitude.shape[0]  # 257 for n_fft=512

        # Stretch factor: positive semitones = compress source bins
        # = shift spectral energy toward higher frequencies
        stretch = 2.0 ** (semitones / 12.0)

        # For each output bin, find where it maps back in the original spectrum
        orig_bins = torch.arange(freq_bins, device=audio.device, dtype=torch.float32)
        source_bins = orig_bins / stretch
        source_bins = source_bins.clamp(0.0, freq_bins - 1.0)

        # Linear interpolation along frequency axis
        floor_bins = source_bins.floor().long()
        ceil_bins = (floor_bins + 1).clamp(max=freq_bins - 1)
        frac = (source_bins - floor_bins.float()).unsqueeze(1)  # [freq_bins, 1]

        # Interpolate magnitude only — preserve original phase
        # This avoids the phase artifacts that aggressive processing would introduce
        mag_nudged = (
            magnitude[floor_bins] * (1.0 - frac) +
            magnitude[ceil_bins] * frac
        )

        # Reconstruct complex STFT: nudged magnitude + original phase
        stft_out = torch.polar(mag_nudged, phase)

        # Inverse STFT — recover waveform at original length
        audio_out = torch.istft(
            stft_out,
            n_fft=self._stft_n_fft,
            hop_length=self._stft_hop,
            win_length=self._stft_win,
            window=window,
            length=audio_f32.shape[0],
        )

        # Return same dtype as input (fp16 or fp32)
        return audio_out.to(original_dtype)

    def extract_pitch(self, audio: torch.Tensor, pitch: torch.Tensor | None, pitchf: torch.Tensor | None, f0_up_key: int, formant_shift: float) -> tuple[torch.Tensor, torch.Tensor]:
        f0 = self.pitchExtractor.extract(
            audio,
            HUBERT_SAMPLE_RATE,
            WINDOW_SIZE,
        )
        f0 *= 2 ** ((f0_up_key - formant_shift) / 12)

        f0_mel = 1127.0 * torch.log(1.0 + f0 / 700.0)
        f0_mel = torch.clip(
            (f0_mel - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1,
            1,
            255,
            out=f0_mel
        )
        f0_coarse = torch.round(f0_mel, out=f0_mel).long()

        if pitch is not None and pitchf is not None:
            circular_write(f0_coarse, pitch)
            circular_write(f0, pitchf)
        else:
            pitch = f0_coarse
            pitchf = f0

        return pitch.unsqueeze(0), pitchf.unsqueeze(0)

    def _search_index(self, audio: torch.Tensor, top_k: int = 1):
        if top_k == 1:
            _, ix = self.index.search(audio if self.use_gpu_index else audio.detach().cpu(), 1)
            ix = ix.to(self.device)
            return self.index_reconstruct[ix.squeeze()]

        score, ix = self.index.search(audio if self.use_gpu_index else audio.detach().cpu(), k=top_k)
        score, ix = (
            score.to(self.device),
            ix.to(self.device),
        )
        weight = torch.square(1 / score)
        weight /= weight.sum(dim=1, keepdim=True)
        return torch.sum(self.index_reconstruct[ix] * weight.unsqueeze(2), dim=1)

    def _upscale(self, feats: torch.Tensor) -> torch.Tensor:
        if self.onnx_upscaler is not None:
            feats = self.onnx_upscaler.run(['out'], { 'in': feats.permute(0, 2, 1).detach().cpu().numpy(), 'scales': np.array([2], dtype=np.float32) })
            return torch.as_tensor(feats[0], dtype=self.dtype, device=self.device).permute(0, 2, 1).contiguous()
        return F.interpolate(feats.permute(0, 2, 1), scale_factor=2, mode='nearest').permute(0, 2, 1).contiguous()

    def exec(
        self,
        sid: int,
        audio: torch.Tensor,  # torch.tensor [n]
        pitch: torch.Tensor | None,  # torch.tensor [m]
        pitchf: torch.Tensor | None,  # torch.tensor [m]
        f0_up_key: int,
        formant_shift: float,
        index_rate: float,
        audio_feats_len: int,
        silence_front: int,
        embOutputLayer: int,
        useFinalProj: bool,
        skip_head: int,
        return_length: int,
        protect: float = 0.5,
        pre_formant_shift: float = 0.0,
    ) -> torch.Tensor:
        with Timer2("Pipeline-Exec", False) as t:  # NOQA
            # 16000のサンプリングレートで入ってきている。以降この世界は16000で処理。
            assert audio.dim() == 1, audio.dim()

            formant_factor = 2 ** (formant_shift / 12)
            formant_length = int(np.ceil(return_length * formant_factor))
            t.record("pre-process")

            # ピッチ検出
            # NOTE: pitch extraction uses original audio — pre_formant_nudge
            # only affects the embedder input below, not F0 extraction.
            pitch, pitchf = self.extract_pitch(audio[silence_front:], pitch, pitchf, f0_up_key, formant_shift) if self.use_f0 else (None, None)
            t.record("extract-pitch")

            # Pre-ContentVec formant nudge (M2F preprocessing)
            # Shifts spectral envelope toward female characteristics BEFORE
            # feature extraction. Reduces acoustic distance the model must bridge.
            # F0 is untouched — handled entirely by extract_pitch above.
            # Set preFormantShift = 0 in settings to disable (default behavior).
            if pre_formant_shift != 0.0:
                audio = self._pre_formant_nudge(audio, pre_formant_shift)

            # embedding
            feats = self.embedder.extract_features(audio.view(1, -1), embOutputLayer, useFinalProj)
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            t.record("extract-feats")

            # Index - feature抽出
            is_active_index = self.use_index and index_rate > 0
            use_protect = protect < 0.5
            if self.use_f0 and is_active_index and use_protect:
                feats_orig = feats.detach().clone()

            if is_active_index:
                skip_offset = skip_head // 2
                index_audio = feats[0][skip_offset :]

                index_audio = self._search_index(index_audio.float(), 8).unsqueeze(0)
                if self.is_half:
                    index_audio = index_audio.half()

                # Recover silent front
                feats[0][skip_offset :] = index_audio * index_rate + feats[0][skip_offset :] * (1 - index_rate)

            feats = self._upscale(feats)[:, :audio_feats_len, :]
            if self.use_f0:
                pitch = pitch[:, -audio_feats_len:]
                pitchf = pitchf[:, -audio_feats_len:] * (formant_length / return_length)
                if is_active_index and use_protect:
                    # FIXME: Another interpolate on feats is a big performance hit.
                    feats_orig = self._upscale(feats_orig)[:, :audio_feats_len, :]
                    pitchff = pitchf.detach().clone()
                    pitchff[pitchf > 0] = 1
                    pitchff[pitchf < 1] = protect
                    pitchff = pitchff.unsqueeze(-1)
                    feats = feats * pitchff + feats_orig * (1 - pitchff)

            p_len = torch.tensor([audio_feats_len], device=self.device, dtype=torch.int64)

            sid = torch.tensor([sid], device=self.device, dtype=torch.int64)
            t.record("mid-precess")

            # 推論実行
            out_audio = self.inferencer.infer(feats, p_len, pitch, pitchf, sid, skip_head, return_length, formant_length).float()
            t.record("infer")

            # Formant shift sample rate adjustment (existing post-inference formant shift)
            scaled_window = int(np.floor(formant_factor * self.model_window))
            if scaled_window != self.model_window:
                if scaled_window not in self.resamplers:
                    self.resamplers[scaled_window] = tat.Resample(
                        orig_freq=scaled_window,
                        new_freq=self.model_window,
                        dtype=torch.float32,
                    ).to(self.device)
                out_audio = self.resamplers[scaled_window](
                    out_audio[: return_length * scaled_window]
                )
        return out_audio