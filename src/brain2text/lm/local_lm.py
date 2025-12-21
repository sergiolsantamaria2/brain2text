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


def _rearrange_logits_btc(logits_btc: np.ndarray) -> np.ndarray:
    """
    Input order expected from your acoustic model: [BLANK, phonemes..., SIL(last)].
    LM expects: [BLANK, SIL, phonemes...].
    """
    return np.concatenate((logits_btc[:, :, 0:1], logits_btc[:, :, -1:], logits_btc[:, :, 1:-1]), axis=-1)


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

    def decode_from_logits(
        self,
        logits_tc: np.ndarray,
        input_is_log_probs: bool = False,
    ) -> str:
        """
        logits_tc: (T, C=41) float32, en orden acústico [BLANK, phonemes..., SIL(last)]
        El decoder (por tu TLG) necesita 42 ilabels porque incluye <eps>=0.
        Nosotros construimos logprobs (T,42): [EPS, BLANK, SIL, phonemes...]
        """
        lm_decoder = self._lm_decoder
        dec = self._decoder

        logits_tc = np.asarray(logits_tc, dtype=np.float32)
        logits_btc = logits_tc[None, :, :]               # (1,T,41)
        logits_rearr_tc = _rearrange_logits_btc(logits_btc)[0]  # (T,41) => [BLANK,SIL,phones...]

        # A log-probs (T,41)
        if input_is_log_probs:
            lp41 = logits_rearr_tc.astype(np.float32)
        else:
            lp41 = _log_softmax_tc(logits_rearr_tc).astype(np.float32)

        # PAD EPS -> (T,42) con <eps>=0 a -inf
        lp42 = np.concatenate(
            [np.full((lp41.shape[0], 1), NEG_INF, dtype=np.float32), lp41],
            axis=-1,
        )

        # Decodifica (en tu build devuelve None, pero rellena dec.result())
        if hasattr(lm_decoder, "DecodeNumpyLogProbs"):
            lm_decoder.DecodeNumpyLogProbs(dec, lp42)
        else:
            # fallback si alguna vez no existe LogProbs: pasamos probs
            probs42 = np.exp(lp42).astype(np.float32)
            lm_decoder.DecodeNumpy(dec, probs42)

        # Lo correcto en tu build: leer del decoder
        if hasattr(dec, "result") and callable(dec.result):
            res = dec.result()
            return _decode_result_best_text(res)

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