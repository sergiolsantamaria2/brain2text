# src/brain2text/lm/local_lm.py
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import os
import numpy as np

try:
    import editdistance
except Exception:  # pragma: no cover
    editdistance = None

NEG_INF = np.float32(-1e9)


def _remove_punctuation(sentence: str) -> str:
    import re
    sentence = re.sub(r"[^a-zA-Z\- \']", "", sentence)
    sentence = sentence.replace("- ", " ").lower()
    sentence = sentence.replace("--", "").lower()
    sentence = sentence.replace(" '", "'").lower()
    sentence = sentence.strip()
    sentence = " ".join([w for w in sentence.split() if w])
    return sentence


# --- reemplaza tu función por esta ---
def _rearrange_logits_to_lm_order(logits_tc: np.ndarray, sil_index: int, mode: str = "identity") -> np.ndarray:
    """
    Reorder columns for LM if needed.

    mode:
      - "identity": do not reorder (LM expects same order as acoustic model)
      - "blank_sil_phones": LM expects [BLANK, SIL, phones...]
    """
    mode = str(mode).lower().strip()
    if mode in ("identity", "none", "acoustic"):
        return logits_tc

    if mode not in ("blank_sil_phones", "bsph"):
        raise ValueError(f"Unknown reorder mode: {mode}")

    C = int(logits_tc.shape[-1])
    if sil_index < 0:
        sil_index = C - 1
    if sil_index == 0 or sil_index >= C:
        raise ValueError(f"Invalid sil_index={sil_index} for C={C}. BLANK must be 0 and SIL != 0.")

    idx = [0, sil_index] + [i for i in range(1, C) if i != sil_index]
    return logits_tc[:, idx]




def _log_softmax_tc(x_tc: np.ndarray) -> np.ndarray:
    # stable log-softmax over last dim
    m = x_tc.max(axis=-1, keepdims=True)
    z = x_tc - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True) + 1e-8)


def _decode_result_best_text(out: Any) -> str:
    if out is None:
        return ""

    # Many builds return list/tuple (nbest). Pick first non-empty.
    if isinstance(out, (list, tuple)):
        if len(out) == 0:
            return ""
        # en tu build suele venir list[DecodeResult]
        return _decode_result_best_text(out[0])

    # bytes / str
    if isinstance(out, bytes):
        return out.decode("utf-8", errors="ignore")
    if isinstance(out, str):
        return out

    # Different builds expose different attribute names on DecodeResult-like objects
    for k in ("best_sentence", "best_text", "sentence", "text", "decoded"):
        if hasattr(out, k):
            v = getattr(out, k)
            if callable(v):
                try:
                    v = v()
                except Exception:
                    continue
            if v is None:
                continue
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="ignore")
            return str(v)

    # Fallback
    s = str(out)
    # avoid returning noise-like "None" or "[]"
    if s in ("None", "[]"):
        return ""
    return s

def _best_text_from_decoder(dec: Any) -> Optional[str]:
    if hasattr(dec, "result") and callable(getattr(dec, "result")):
        try:
            r = dec.result()
            txt = _decode_result_best_text(r)
            return txt if txt else None
        except Exception:
            return None
    return None

@dataclass
class LocalLMConfig:
    lm_dir: str
    # DecodeOptions (8-arg constructor in your build)
    max_active: int = 7000
    min_active: int = 200
    beam: float = 15.0
    lattice_beam: float = 8.0
    ctc_blank_skip_threshold: float = 0.95
    length_penalty: float = 0.0
    acoustic_scale: float = 0.35
    nbest: int = 50
    # optional
    blank_penalty: float = 90.0
    sil_index: int = -1  # -1 means "last index"
    reorder_mode: str = "identity"




class LocalNgramDecoder:
    """
    Clean, non-Redis decoder wrapper around lm_decoder.*.
    Safe to import on Mac even if lm_decoder is not installed:
    it only imports lm_decoder when you instantiate the class.
    """

    def __init__(self, cfg: LocalLMConfig):
        self.cfg = cfg
        self._lm_decoder = None
        self._decoder = None

        # Lazy import so Mac dev doesn't break
        try:
            import lm_decoder  # type: ignore
            self._lm_decoder = lm_decoder
        except Exception as e:
            raise RuntimeError(
                "lm_decoder is not available in this environment. "
                "This is expected on Mac. Run on the cluster where lm_decoder is installed."
            ) from e

        self._decoder = self._build_decoder()

    def _build_options(self):
        lm_decoder = self._lm_decoder
        cfg = self.cfg
        return lm_decoder.DecodeOptions(
            int(cfg.max_active),
            int(cfg.min_active),
            float(cfg.beam),
            float(cfg.lattice_beam),
            float(cfg.ctc_blank_skip_threshold),
            float(cfg.length_penalty),
            float(cfg.acoustic_scale),
            int(cfg.nbest),
        )

    def _build_decoder(self):
        lm_decoder = self._lm_decoder
        cfg = self.cfg
        opts = self._build_options()

        try:
            return lm_decoder.BrainSpeechDecoder(cfg.lm_dir, opts)
        except TypeError:
            pass

        lm_dir = Path(cfg.lm_dir).resolve()
        tlg = str(lm_dir / "TLG.fst")
        words = str(lm_dir / "words.txt")

        res = lm_decoder.DecodeResource(tlg, tlg, "", words, "")
        return lm_decoder.BrainSpeechDecoder(res, opts)
    
    def _maybe_reset_decoder(self, dec: Any) -> bool:
        # Try common reset/clear methods exposed by bindings
        for name in ("Reset", "reset", "Clear", "clear", "ResetDecoding", "reset_decoding", "ClearResult", "clear_result"):
            if hasattr(dec, name) and callable(getattr(dec, name)):
                try:
                    getattr(dec, name)()
                    return True
                except Exception:
                    pass

        # Some bindings expose reset as a function on lm_decoder module
        for name in ("Reset", "ResetDecoder", "ResetDecoding"):
            if hasattr(self._lm_decoder, name) and callable(getattr(self._lm_decoder, name)):
                try:
                    getattr(self._lm_decoder, name)(dec)
                    return True
                except Exception:
                    pass
        return False


    def decode_from_logits(self, logits_tc: np.ndarray, input_is_log_probs: bool = False) -> str:
        lm_decoder = self._lm_decoder
        dec = self._decoder

        if not self._maybe_reset_decoder(dec):
            # Fallback: rebuild decoder if no reset API exists
            dec = self._build_decoder()
            self._decoder = dec

        logits_tc = np.asarray(logits_tc, dtype=np.float32)      # (T,41)
        logits_rearr_tc = _rearrange_logits_to_lm_order(
            logits_tc,
            int(self.cfg.sil_index),
            mode=str(getattr(self.cfg, "reorder_mode", "identity")),
        )

        lp41 = logits_rearr_tc if input_is_log_probs else _log_softmax_tc(logits_rearr_tc)
        lp41 = lp41.astype(np.float32)

        # (T,42) = [EPS, BLANK, SIL, phones...] donde EPS es imposible
        lp42 = np.concatenate([np.full((lp41.shape[0], 1), NEG_INF, dtype=np.float32), lp41], axis=-1)

        if hasattr(lm_decoder, "DecodeNumpyLogProbs"):
            lm_decoder.DecodeNumpyLogProbs(dec, lp42)
        else:
            lm_decoder.DecodeNumpy(dec, np.exp(lp42).astype(np.float32))

        res = dec.result() if hasattr(dec, "result") and callable(dec.result) else None
        if isinstance(res, list) and len(res) > 0:
            s = getattr(res[0], "sentence", "")
            return "" if s is None else str(s)
        return ""

    def wer_percent(self, true_sentence: str, pred_sentence: str) -> Optional[float]:
        if editdistance is None:
            return None
        t = _remove_punctuation(str(true_sentence)).split()
        p = _remove_punctuation(str(pred_sentence)).split()
        if len(t) == 0:
            return None
        ed = editdistance.eval(t, p)
        return 100.0 * float(ed) / float(len(t))