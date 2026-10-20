"""Dataset + preprocessing for Task 1 (synthetic moving shapes, or your own video folder)."""
import glob, numpy as np, torch, torch.nn.functional as F


def make_moving_shapes(n, T=16, S=48, seed=0):
    """Synthetic grayscale clips (uint8, n x T x S x S): circle/square bouncing on a dark background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:S, 0:S]
    out = np.zeros((n, T, S, S), np.uint8)
    for i in range(n):
        r = rng.integers(S // 8, S // 5)
        pos = rng.uniform(r, S - r, 2)
        vel = rng.uniform(-S / 12, S / 12, 2)
        inten, circle = rng.uniform(0.6, 1.0), rng.random() < 0.5
        for t in range(T):
            pos += vel
            for k in range(2):  # bounce on walls
                if pos[k] < r or pos[k] > S - r:
                    vel[k] *= -1
                    pos[k] = np.clip(pos[k], r, S - r)
            if circle:
                m = (yy - pos[0]) ** 2 + (xx - pos[1]) ** 2 <= r * r
            else:
                m = (abs(yy - pos[0]) <= r) & (abs(xx - pos[1]) <= r)
            out[i, t] = (m * inten * 255).astype(np.uint8)
    return out


def read_video_folder(folder, T, max_clips=None):
    """Optional: real videos (mp4/avi/mov/gif). Returns list of uint8 arrays (frames,H,W,3)."""
    import imageio.v2 as imageio
    clips = []
    for p in sorted(sum([glob.glob(f"{folder}/*.{e}") for e in ("mp4", "avi", "mov", "gif")], []))[:max_clips]:
        fr = [f for f in imageio.get_reader(p)]
        if len(fr) >= T:
            clips.append(np.stack(fr))
    return clips


def preprocess(clip, T, S):
    """Preprocessing pipeline: (1) grayscale -> (2) uniform temporal sampling to T frames ->
    (3) centre-crop to square -> (4) resize to SxS -> (5) normalise to [-1, 1]. Returns (1,T,S,S) float."""
    x = torch.as_tensor(clip).float() / 255.0
    if x.ndim == 4:  # (frames,H,W,3) -> gray
        x = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])
    idx = torch.linspace(0, x.shape[0] - 1, T).round().long()
    x = x[idx]
    h, w = x.shape[1:]
    m = min(h, w)
    x = x[:, (h - m) // 2:(h - m) // 2 + m, (w - m) // 2:(w - m) // 2 + m]
    if m != S:
        x = F.interpolate(x[:, None], size=(S, S), mode="area" if m > S else "bilinear")[:, 0]
    return (x * 2 - 1).unsqueeze(0)


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, clips, T, S, train=False):
        self.clips, self.T, self.S, self.train = clips, T, S, train

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        x = preprocess(self.clips[i], self.T, self.S)
        if self.train:  # augmentation: horizontal flip + time reversal (both physically valid here)
            if torch.rand(1) < 0.5:
                x = x.flip(-1)
            if torch.rand(1) < 0.5:
                x = x.flip(1)
        return x


def get_splits(n_clips=1500, T=8, S=32, seed=0, video_dir=None, split=(0.8, 0.1, 0.1)):
    clips = read_video_folder(video_dir, T, n_clips) if video_dir else list(make_moving_shapes(n_clips, T=max(T, 16), S=48, seed=seed))
    perm = np.random.default_rng(seed).permutation(len(clips))
    a, b = int(split[0] * len(clips)), int((split[0] + split[1]) * len(clips))
    pick = lambda ids: [clips[j] for j in ids]
    return (VideoDataset(pick(perm[:a]), T, S, True), VideoDataset(pick(perm[a:b]), T, S), VideoDataset(pick(perm[b:]), T, S))
