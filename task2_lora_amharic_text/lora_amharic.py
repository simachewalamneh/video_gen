"""Task 2: parameter-efficient (LoRA) fine-tuning of a lightweight multilingual model (mT5-small, ~300M) for English->Amharic
text generation on the Colab free tier.  Freezes the base weights, trains only low-rank adapters.
Course-concept extras:
  * iterative error correction  -> round 2 re-samples training data in proportion to per-example loss (error-focused replay)
  * uncertainty-aware adaptation -> MC-dropout candidates; agreement = uncertainty; MBR selection; selective generation
Resource notes: fp32 base (mT5 overflows in fp16), LoRA only, gradient checkpointing, small batch/length, memory vs full-FT estimate."""
import argparse, json, os, random, time
import numpy as np, torch


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_pairs(a):
    """Returns list of (english, amharic). Default: OPUS-100 am-en. Or --tsv file with 'english<TAB>amharic' lines."""
    if a.tsv:
        pairs = [tuple(l.rstrip("\n").split("\t")[:2]) for l in open(a.tsv, encoding="utf-8") if "\t" in l]
    else:
        from datasets import load_dataset
        ds = None
        for cfg in ("am-en", "en-am"):
            try: ds = load_dataset("Helsinki-NLP/opus-100", cfg, split="train"); break
            except Exception as e: err = e
        if ds is None: raise SystemExit(f"Could not load OPUS-100 ({err}); pass --tsv instead.")
        pairs = [(r["translation"]["en"], r["translation"]["am"]) for r in ds]
    pairs = [p for p in pairs if 3 < len(p[0]) < 200 and 3 < len(p[1]) < 200]
    random.Random(a.seed).shuffle(pairs)
    return pairs


def batches(tok, pairs, bs, dev, max_len):
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        x = tok([f"translate English to Amharic: {e}" for e, _ in chunk], max_length=max_len, truncation=True, padding=True, return_tensors="pt")
        y = tok([m for _, m in chunk], max_length=max_len, truncation=True, padding=True, return_tensors="pt").input_ids
        y[y == tok.pad_token_id] = -100
        yield x.input_ids.to(dev), x.attention_mask.to(dev), y.to(dev)


def per_example_loss(model, ids, mask, y):
    logits = model(input_ids=ids, attention_mask=mask, labels=y).logits
    tl = torch.nn.functional.cross_entropy(logits.transpose(1, 2).float(), y, ignore_index=-100, reduction="none")
    return tl.sum(1) / (y != -100).sum(1).clamp_min(1)


@torch.no_grad()
def generate(model, tok, srcs, dev, max_len, bs=32, **kw):
    out = []
    for i in range(0, len(srcs), bs):
        x = tok([f"translate English to Amharic: {s}" for s in srcs[i:i + bs]], max_length=max_len, truncation=True, padding=True, return_tensors="pt").to(dev)
        g = model.generate(**x, max_new_tokens=max_len, num_beams=1, do_sample=False, **kw)
        out += tok.batch_decode(g, skip_special_tokens=True)
    return out


def ranks(v): return np.argsort(np.argsort(v)).astype(float)


def main(a):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import LoraConfig, get_peft_model
    import sacrebleu
    seed_all(a.seed); dev = "cuda"; os.makedirs(a.out, exist_ok=True)
    pairs = load_pairs(a); test, val, train = pairs[:a.n_test], pairs[a.n_test:a.n_test + 300], pairs[a.n_test + 300:a.n_test + 300 + a.n_train]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev)
    total = sum(p.numel() for p in model.parameters())
    model = get_peft_model(model, LoraConfig(task_type="SEQ_2_SEQ_LM", r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.1, target_modules=["q", "k", "v", "o"] + (["wi_0", "wi_1", "wo"] if a.ffn else [])))
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {train_p/1e6:.2f}M / {total/1e6:.0f}M ({100*train_p/total:.2f}%)")
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    use_bf16 = torch.cuda.get_device_capability()[0] >= 8
    torch.cuda.reset_peak_memory_stats(); t0, log = time.time(), []

    def run_epoch(data, tag):
        model.train(); tot = n = 0
        for ids, mask, y in batches(tok, data, a.bs, dev, a.max_len):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = per_example_loss(model, ids, mask, y).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); n += 1
            if n % 50 == 0: print(f"[{tag}] step {n} loss {tot/n:.3f} | {time.time()-t0:.0f}s", flush=True)
            log.append((tag, loss.item()))
        return tot / max(n, 1)

    run_epoch(train, "round1-uniform")
    peak_r1 = torch.cuda.max_memory_allocated() / 2**30
    # --- iterative error correction: error-focused replay ---
    model.eval(); pool = train[:a.replay_pool]; losses = []
    with torch.no_grad():
        for ids, mask, y in batches(tok, pool, 16, dev, a.max_len):
            losses += per_example_loss(model, ids, mask, y).tolist()
    losses = np.array(losses); losses = np.minimum(losses, np.percentile(losses, 95))  # clip outliers / noisy pairs
    hard = np.random.choice(len(pool), a.replay_pool // 2, p=losses / losses.sum())
    round2 = [pool[i] for i in hard] + train[:a.replay_pool // 2]; random.shuffle(round2)
    run_epoch(round2, "round2-replay")
    train_s, peak = time.time() - t0, torch.cuda.max_memory_allocated() / 2**30
    model.save_pretrained(a.out)  # adapter only (few MB)
    # --- evaluation ---
    model.gradient_checkpointing_disable(); model.eval(); srcs, refs = [e for e, _ in test], [m for _, m in test]
    with model.disable_adapter(): base = generate(model, tok, srcs, dev, a.max_len)
    greedy = generate(model, tok, srcs, dev, a.max_len)
    # uncertainty-aware decoding: K MC-dropout passes (dropout active), agreement = confidence, MBR pick
    model.train(); cands = [generate(model, tok, srcs, dev, a.max_len) for _ in range(a.k)]; model.eval()
    chrf = lambda h, r: sacrebleu.sentence_chrf(h, [r]).score
    mbr, unc = [], []
    for i in range(len(srcs)):
        c = [greedy[i]] + [cc[i] for cc in cands]
        sc = [np.mean([chrf(c[j], c[k]) for k in range(len(c)) if k != j]) for j in range(len(c))]
        mbr.append(c[int(np.argmax(sc))]); unc.append(100 - max(sc))
    unc = np.array(unc); q = np.array([chrf(h, r) for h, r in zip(greedy, refs)])
    corr = float(np.corrcoef(ranks(unc), ranks(q))[0, 1]); keep = unc <= np.median(unc)
    C = lambda h: sacrebleu.corpus_chrf(h, [refs]).score
    res = dict(trainable_params_m=train_p / 1e6, trainable_pct=100 * train_p / total, train_s=train_s, peak_gpu_gb=peak, peak_gpu_gb_round1=peak_r1,
               lora_weights_plus_states_gb=(total * 4 + train_p * 12) / 2**30,
               full_ft_adam_fp32_est_gb=total * 16 / 2**30,
               chrf_base_zero_shot=C(base), chrf_lora_greedy=C(greedy), chrf_lora_mc_mbr=C(mbr),
               bleu_lora_greedy=sacrebleu.corpus_bleu(greedy, [refs]).score,
               chrf_confident_half=sacrebleu.corpus_chrf([g for g, k in zip(greedy, keep) if k], [[r for r, k in zip(refs, keep) if k]]).score,
               uncertainty_vs_quality_spearman=corr)
    json.dump(res, open(f"{a.out}/summary.json", "w"), indent=1)
    with open(f"{a.out}/generations.jsonl", "w", encoding="utf-8") as f:
        for i in range(len(srcs)):
            f.write(json.dumps(dict(src=srcs[i], ref=refs[i], base=base[i], lora=greedy[i], mbr=mbr[i], uncertainty=float(unc[i])), ensure_ascii=False) + "\n")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for tag in ("round1-uniform", "round2-replay"): plt.plot([l for t, l in log if t == tag], label=tag, alpha=.7)
    plt.legend(); plt.xlabel("step"); plt.ylabel("loss"); plt.savefig(f"{a.out}/loss.png", dpi=120)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/mt5-small"); p.add_argument("--out", default="outputs/mt5_lora_r8")
    p.add_argument("--tsv", default=None); p.add_argument("--n_train", type=int, default=8000); p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--rank", type=int, default=8); p.add_argument("--bs", type=int, default=16); p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--k", type=int, default=4, help="MC-dropout passes")
    p.add_argument("--ffn", action="store_true", help="also adapt FFN layers (wi_0, wi_1, wo)")
    p.add_argument("--replay_pool", type=int, default=4000); p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
