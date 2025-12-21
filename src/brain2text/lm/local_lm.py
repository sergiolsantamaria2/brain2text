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
        """
        Build lm_decoder.DecodeOptions in a way that is compatible with different bindings:
        - some builds support DecodeOptions() + attribute assignment
        - others require positional constructor args
        """
        lm_decoder = self._lm_decoder
        cfg = self.cfg

        # 1) Try no-arg constructor
        try:
            opts = lm_decoder.DecodeOptions()
        except TypeError:
            opts = None

        # 2) If no-arg doesn't exist, try a few positional signatures
        if opts is None:
            candidates = [
                # Common 8-arg form (your comment says 8-arg)
                (cfg.max_active, cfg.min_active, cfg.beam, cfg.lattice_beam,
                cfg.ctc_blank_skip_threshold, cfg.length_penalty, cfg.acoustic_scale, cfg.nbest),
                # Some builds include blank_penalty as a 9th arg
                (cfg.max_active, cfg.min_active, cfg.beam, cfg.lattice_beam,
                cfg.ctc_blank_skip_threshold, cfg.length_penalty, cfg.acoustic_scale, cfg.nbest, cfg.blank_penalty),
            ]
            last_err = None
            for args in candidates:
                try:
                    opts = lm_decoder.DecodeOptions(*args)
                    break
                except Exception as e:
                    last_err = e
            if opts is None:
                raise RuntimeError(f"Could not construct DecodeOptions with known signatures. Last error: {last_err}") from last_err

        # 3) Best-effort attribute assignment (covers no-arg builds and ignores missing attrs)
        attr_map = {
            "max_active": cfg.max_active,
            "maxActive": cfg.max_active,
            "min_active": cfg.min_active,
            "minActive": cfg.min_active,
            "beam": cfg.beam,
            "lattice_beam": cfg.lattice_beam,
            "latticeBeam": cfg.lattice_beam,
            "ctc_blank_skip_threshold": cfg.ctc_blank_skip_threshold,
            "ctcBlankSkipThreshold": cfg.ctc_blank_skip_threshold,
            "length_penalty": cfg.length_penalty,
            "lengthPenalty": cfg.length_penalty,
            "acoustic_scale": cfg.acoustic_scale,
            "acousticScale": cfg.acoustic_scale,
            "nbest": cfg.nbest,
            "blank_penalty": cfg.blank_penalty,
            "blankPenalty": cfg.blank_penalty,
        }
        for k, v in attr_map.items():
            if hasattr(opts, k):
                try:
                    setattr(opts, k, v)
                except Exception:
                    pass

        return opts

    def _build_decoder(self):
        lm_decoder = self._lm_decoder
        cfg = self.cfg
        opts = self._build_options()

        # Preferred path: some builds accept (lm_dir: str, opts)
        try:
            return lm_decoder.BrainSpeechDecoder(cfg.lm_dir, opts)
        except TypeError:
            pass  # Fall back to DecodeResource API

        lm_dir = Path(cfg.lm_dir)
        lm_dir = lm_dir.resolve()
        tlg = str(lm_dir / "TLG.fst")
        words = str(lm_dir / "words.txt")

        # Some builds expect BrainSpeechDecoder(DecodeResource, DecodeOptions).
        # DecodeResource is a 5-string constructor, but the argument order can differ across builds.
        # We try a small set of plausible orders and pick the first that initializes successfully.
        candidates = [
            (tlg, words, "", "", ""),
            (words, tlg, "", "", ""),
            (str(lm_dir), tlg, words, "", ""),
            (str(lm_dir), words, tlg, "", ""),
            (tlg, words, str(lm_dir), "", ""),
            (words, tlg, str(lm_dir), "", ""),
        ]

        last_err = None
        for args in candidates:
            try:
                res = lm_decoder.DecodeResource(*args)
                return lm_decoder.BrainSpeechDecoder(res, opts)
            except Exception as e:
                last_err = e

        raise RuntimeError(
            f"Failed to initialize BrainSpeechDecoder via DecodeResource. "
            f"lm_dir={cfg.lm_dir}, tried {len(candidates)} DecodeResource argument orders. "
            f"Last error: {last_err}"
        ) from last_err

    def decode_from_logits(
        self,
        logits_tc: np.ndarray,
        input_is_log_probs: bool = False,
    ) -> str:
        """
        logits_tc: (T, C) float32, in acoustic model order [BLANK, phonemes..., SIL]
        """
        lm_decoder = self._lm_decoder
        dec = self._decoder

        logits_tc = np.asarray(logits_tc, dtype=np.float32)
        logits_btc = logits_tc[None, :, :]  # (1,T,C)
        logits_rearr_tc = _rearrange_logits_btc(logits_btc)[0]  # (T,C)

        if input_is_log_probs:
            log_probs_tc = logits_rearr_tc
            if hasattr(lm_decoder, "DecodeNumpyLogProbs"):
                out = lm_decoder.DecodeNumpyLogProbs(dec, log_probs_tc)
            else:
                # fallback: some builds only have DecodeNumpy
                out = lm_decoder.DecodeNumpy(dec, log_probs_tc)
        else:
            log_probs_tc = _log_softmax_tc(logits_rearr_tc)
            if hasattr(lm_decoder, "DecodeNumpyLogProbs"):
                out = lm_decoder.DecodeNumpyLogProbs(dec, log_probs_tc)
            else:
                out = lm_decoder.DecodeNumpy(dec, logits_rearr_tc)

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