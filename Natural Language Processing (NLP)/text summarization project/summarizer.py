import os, re
import torch
import torch.nn.functional as F
import sentencepiece as spm
from model import Transformer, make_src_mask, make_tgt_mask

_SENT = re.compile(r"(?<=[.!?])\s+")


class Summarizer:
    def __init__(self, base_dir=None, device=None):
        base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.sp = spm.SentencePieceProcessor(model_file=os.path.join(base_dir, "tokenizer", "bpe.model"))
        self.PAD, self.BOS, self.EOS = self.sp.pad_id(), self.sp.bos_id(), self.sp.eos_id()

        ck = torch.load(os.path.join(base_dir, "exports", "summarizer_inference.pt"), map_location=self.device)
        self.max_src, self.max_tgt = ck["max_src"], ck["max_tgt"]
        self.net = Transformer(**ck["model_cfg"]).to(self.device)
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.info = ck.get("info", {})
        self.n_params = sum(p.numel() for p in self.net.parameters())

    @staticmethod
    def lead3(text):
        return " ".join(_SENT.split(text.strip())[:3])

    def prepare(self, text):
        text = re.sub(r"\s+", " ", text).strip()
        ids = self.sp.encode(text[:4000])
        used = ids[: self.max_src - 2]
        return used, len(ids) > len(used), text

    @torch.no_grad()
    def generate(self, ids, beam=1, alpha=0.8, no_repeat=3, min_len=25):
        net, dev = self.net, self.device
        src = torch.tensor([[self.BOS] + list(ids) + [self.EOS]], device=dev)
        src_mask = make_src_mask(src, self.PAD)
        memory = net.encode(src, src_mask)

        beams, finished = [([self.BOS], 0.0)], []
        for step in range(self.max_tgt - 2):
            seqs = torch.tensor([b[0] for b in beams], device=dev)
            nb = seqs.size(0)
            logits = net.decode(seqs, memory.expand(nb, -1, -1), src_mask.expand(nb, -1, -1, -1),
                                make_tgt_mask(seqs, self.PAD))
            logp = F.log_softmax(logits[:, -1].float(), dim=-1)
            logp[:, self.PAD] = -1e9
            logp[:, self.BOS] = -1e9
            if step < min_len:
                logp[:, self.EOS] = -1e9
            if no_repeat > 1:
                for bi, (toks, _) in enumerate(beams):
                    if len(toks) >= no_repeat:
                        prefix = tuple(toks[-(no_repeat - 1):])
                        banned = [toks[i + no_repeat - 1] for i in range(len(toks) - no_repeat + 1)
                                  if tuple(toks[i:i + no_repeat - 1]) == prefix]
                        if banned:
                            logp[bi, banned] = -1e9

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
        best = max(finished, key=lambda it: it[1] / (((5 + len(it[0]) - 1) / 6) ** alpha))
        return self.sp.decode([t for t in best[0][1:] if t != self.EOS])

    def summarize(self, text, beam=1, alpha=0.8, no_repeat=3, min_len=25):
        ids, truncated, clean = self.prepare(text)
        summary = self.generate(ids, beam, alpha, no_repeat, min_len) if len(ids) >= 5 else ""
        return {
            "summary": summary,
            "lead3": self.lead3(clean),
            "truncated": truncated,
            "words_used": len(self.sp.decode(list(ids)).split()),
            "words_total": len(clean.split()),
        }
