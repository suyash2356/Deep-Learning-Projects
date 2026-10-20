import os, re, json
import torch
import torch.nn.functional as F
import sentencepiece as spm
from model import Transformer, make_src_mask, make_tgt_mask


class Translator:
    def __init__(self, base_dir=None, device=None):
        base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.sp = spm.SentencePieceProcessor(
            model_file=os.path.join(base_dir, "tokenizer", "bpe.model"))
        with open(os.path.join(base_dir, "config.json")) as f:
            cfg = json.load(f)
        self.PAD, self.BOS, self.EOS = cfg["pad_id"], cfg["bos_id"], cfg["eos_id"]
        self.max_len = cfg["max_len"]

        ckpt = torch.load(os.path.join(base_dir, "exports", "model_inference.pt"),
                          map_location=self.device)
        self.net = Transformer(**ckpt["model_cfg"]).to(self.device)
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()
        self.info = ckpt.get("info", {})
        self.n_params = sum(p.numel() for p in self.net.parameters())

    # ---------- helpers ----------
    def split_sentences(self, text):
        parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    @torch.no_grad()
    def _encode_source(self, sentence):
        ids = self.sp.encode(sentence)
        truncated = len(ids) > self.max_len - 2
        ids = [self.BOS] + ids[: self.max_len - 2] + [self.EOS]
        src = torch.tensor([ids], device=self.device)
        src_mask = make_src_mask(src, self.PAD)
        memory = self.net.encode(src, src_mask)
        return memory, src_mask, truncated

    # ---------- decoding ----------
    @torch.no_grad()
    def greedy(self, sentence, max_new=70):
        memory, src_mask, truncated = self._encode_source(sentence)
        out = [self.BOS]
        for _ in range(max_new):
            tgt = torch.tensor([out], device=self.device)
            logits = self.net.decode(tgt, memory, src_mask, make_tgt_mask(tgt, self.PAD))
            nxt = logits[0, -1].argmax().item()
            if nxt == self.EOS:
                break
            out.append(nxt)
        return self.sp.decode(out[1:]), truncated

    @torch.no_grad()
    def beam(self, sentence, beam=4, alpha=0.6, max_new=70):
        memory, src_mask, truncated = self._encode_source(sentence)
        beams = [([self.BOS], 0.0)]
        finished = []

        for _ in range(max_new):
            seqs = torch.tensor([b[0] for b in beams], device=self.device)
            nb = seqs.size(0)
            logits = self.net.decode(seqs, memory.expand(nb, -1, -1),
                                     src_mask.expand(nb, -1, -1, -1),
                                     make_tgt_mask(seqs, self.PAD))
            logp = F.log_softmax(logits[:, -1].float(), dim=-1)

            cands = []
            for bi, (toks, score) in enumerate(beams):
                top_lp, top_id = logp[bi].topk(beam)
                for lp, tid in zip(top_lp.tolist(), top_id.tolist()):
                    cands.append((toks + [tid], score + lp))
            cands.sort(key=lambda x: x[1], reverse=True)

            beams = []
            for toks, score in cands:
                if toks[-1] == self.EOS:
                    finished.append((toks, score))
                else:
                    beams.append((toks, score))
                if len(beams) == beam:
                    break
            if not beams or len(finished) >= beam:
                break

        if not finished:
            finished = beams

        def normalized(item):
            toks, score = item
            return score / (((5 + len(toks) - 1) / 6) ** alpha)

        best = max(finished, key=normalized)
        return self.sp.decode([t for t in best[0][1:] if t != self.EOS]), truncated

    # ---------- main entry ----------
    def translate(self, text, mode="beam", beam=4, alpha=0.6):
        pairs, any_truncated = [], False
        for sent in self.split_sentences(text):
            if mode == "greedy":
                hi, trunc = self.greedy(sent)
            else:
                hi, trunc = self.beam(sent, beam=beam, alpha=alpha)
            any_truncated = any_truncated or trunc
            pairs.append((sent, hi))
        return {
            "translation": " ".join(hi for _, hi in pairs),
            "pairs": pairs,
            "truncated": any_truncated,
        }
