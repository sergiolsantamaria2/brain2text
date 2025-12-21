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
    # Different builds expose different attribute names
    for k in ("best_sentence", "best_text", "sentence", "text", "decoded"):
        if hasattr(out, k):
            v = getattr(out, k)
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="ignore")
            return str(v)
    return str(out)

def _best_text_from_decoder(dec: Any) -> Optional[str]:
    # Try attributes first
    for k in ("best_sentence", "best_text", "sentence", "text"):
        if hasattr(dec, k):
            v = getattr(dec, k)
            if v is None:
                continue
            if callable(v):
                try:
                    v = v()
                except Exception:
                    continue
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="ignore")
            return str(v)

    # Try common methods (no-arg)
    for m in ("GetBestSentence", "BestSentence", "GetBestText", "BestText", "GetSentence", "Sentence", "GetText", "Text"):
        if hasattr(dec, m) and callable(getattr(dec, m)):
            try:
                v = getattr(dec, m)()
                if v is None:
                    continue
                if isinstance(v, bytes):
                    return v.decode("utf-8", errors="ignore")
                return str(v)
            except Exception:
                pass

    # Try nbest-style getters
    for m in ("GetNBestSentences", "NBestSentences", "GetNBest", "NBest"):
        if hasattr(dec, m) and callable(getattr(dec, m)):
            try:
                v = getattr(dec, m)(1)
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    v0 = v[0]
                    if isinstance(v0, bytes):
                        return v0.decode("utf-8", errors="ignore")
                    return str(v0)
            except Exception:
                pass

    if hasattr(dec, "result"):
        try:
            r = getattr(dec, "result")
            if callable(r):
                r = r()
            if r is not None:
                return _decode_result_best_text(r)
        except Exception:
            pass

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
        logits_tc: (T, C) float32, in acoustic model order [BLANK, phonemes..., SIL(last)]
        """
        lm_decoder = self._lm_decoder
        dec = self._decoder

        logits_tc = np.asarray(logits_tc, dtype=np.float32)
        logits_btc = logits_tc[None, :, :]  # (1,T,C)
        logits_rearr_tc = _rearrange_logits_btc(logits_btc)[0]  # (T,C)

            # run decode
        if hasattr(lm_decoder, "DecodeNumpyLogProbs"):
            log_probs_tc = logits_rearr_tc if input_is_log_probs else _log_softmax_tc(logits_rearr_tc)
            out = lm_decoder.DecodeNumpyLogProbs(dec, log_probs_tc.astype(np.float32))
        else:
            if input_is_log_probs:
                probs_tc = np.exp(logits_rearr_tc).astype(np.float32)
            else:
                m = logits_rearr_tc.max(axis=-1, keepdims=True)
                e = np.exp(logits_rearr_tc - m)
                probs_tc = (e / (e.sum(axis=-1, keepdims=True) + 1e-8)).astype(np.float32)
            out = lm_decoder.DecodeNumpy(dec, probs_tc)

        # Some builds return None; pull best sentence from decoder object
        if out is None:
            txt = _best_text_from_decoder(dec)
            return "" if txt is None else txt

        return _decode_result_best_text(out)

    def wer_percent(self, true_sentence: str, pred_sentence: str) -> Optional[float]:
        if editdistance is None:
            return None
        t = _remove_punctuation(str(true_sentence)).split()
        p = _remove_punctuation(str(pred_sentence)).split()
        if len(t) == 0:
            return None
        ed = editdistance.eval(t, p)
        return 100.0 * float(ed) / float(len(t))