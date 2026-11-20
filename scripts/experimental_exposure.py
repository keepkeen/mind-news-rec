"""Offline future exposure windows from the public impression timeline."""
import numpy as np
from numba import njit
from exposures import BASE, ITEM_BITS, ORIGIN, TIME_BITS, search
from ranking import arr

HOUR = 3600
DAY = 24 * HOUR
WEEK = 7 * DAY
WINDOWS = (HOUR, 6 * HOUR, DAY)
FORWARD = (
    "log_future_count_1h", "log_future_count_6h", "log_future_count_24h",
    "log_future_per_million_1h", "log_future_per_million_6h", "log_future_per_million_24h",
    "user_shown_again_7d", "log_user_future_count_7d", "log_seconds_to_next_user_show_7d",
    "log_seconds_to_last_article_show_7d",
    "observed_fraction_1h", "observed_fraction_6h", "observed_fraction_24h", "observed_fraction_7d",
)

def stable_uniform(ids, salt):
    # SplitMix64, independent of slate contents, labels and A/B user assignments.
    x = np.asarray(ids, dtype=np.uint64) + np.uint64(salt)
    x = (x ^ (x >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> 27)) * np.uint64(0x94d049bb133111eb)
    return ((x ^ (x >> 31)) >> 11).astype(np.float64) * 2.0**-53

class Timeline:
    def __init__(self):
        times = arr(BASE, "test", "time")
        self.end = int(times.max())
        hour = (times % DAY) // HOUR
        days_left = self.end // DAY - times // DAY
        if days_left.min() != 0 or days_left.max() != 6:
            raise ValueError("Expected seven chronological test dates")
        self.by_hour = np.bincount(hour * 7 + days_left, minlength=24 * 7).reshape(24, 7)
        self.cdf = self.by_hour.cumsum(axis=1) / self.by_hour.sum(axis=1, keepdims=True)
        # Candidate slots per second. Only public timestamps and slate lengths enter this index.
        volumes = np.zeros(self.end - ORIGIN + 1, np.int64)
        for split in ("train", "dev", "test"):
            tt = arr(BASE, split, "time").astype(np.int64) - ORIGIN
            sizes = np.diff(arr(BASE, split, "offset"))
            volumes += np.bincount(tt, weights=sizes, minlength=len(volumes)).astype(np.int64)
        self.traffic = np.r_[np.int64(0), volumes.cumsum(dtype=np.int64)]

    def horizons(self, split):
        times = arr(BASE, split, "time").astype(np.int64)
        real = np.minimum(WEEK, self.end - times)
        if split == "test":
            return real.astype(np.int32)
        # Give an entire labeled impression one independent, test-like recording endpoint.
        # Public test timestamps supply only the 24 x 7 aggregate table.
        ids = arr(BASE, split, "imp_id")
        u = stable_uniform(ids, {"train": 202611191, "dev": 202611192}[split])
        tod = times % DAY
        hour = tod // HOUR
        days = np.zeros(len(times), np.int32)
        for k in range(6):
            days += u >= self.cdf[hour, k]
        virtual = days.astype(np.int64) * DAY + self.end % DAY - tod
        return np.minimum(real, np.maximum(virtual, 0)).astype(np.int32)

    def coverage(self, times, horizon):
        hour = (times % DAY) // HOUR
        windows = (*WINDOWS, WEEK)
        return {
            "impressions": len(horizon),
            "mean_observed_fraction": {str(w): float(np.minimum(horizon, w).mean() / w) for w in windows},
            "truncated_fraction": {str(w): float((horizon < w).mean()) for w in windows},
            "truncated_24h_by_hour": [float((horizon[hour == h] < DAY).mean()) if np.any(hour == h) else None for h in range(24)],
        }

@njit(cache=True)
def forward_values(rows, off, cand, times, users, uids, article, ap, pairs, up, horizon, traffic):
    output = np.empty((np.sum(off[rows + 1] - off[rows]), 14), np.float32)
    at = 0
    for i in rows:
        t = np.int64(times[i]) - ORIGIN
        future = np.int64(horizon[i])
        uid = np.int64(uids[users[i]])
        for j in range(off[i], off[i + 1]):
            news = np.int64(cand[j])
            key = news << TIME_BITS
            # Times have integer-second resolution. Every upper endpoint is inclusive.
            first = search(article, ap[news], ap[news + 1], key + t + 1)
            last = first
            for k, width in enumerate(WINDOWS):
                seconds = min(future, width)
                last = search(article, last, ap[news + 1], key + t + seconds + 1)
                count = last - first
                slots = traffic[t + seconds + 1] - traffic[t + 1]
                output[at, k] = np.log1p(count)
                output[at, 3 + k] = np.log1p(1_000_000.0 * count / slots) if slots else 0.0
                output[at, 10 + k] = seconds / width
            last = search(article, last, ap[news + 1], key + t + future + 1)
            pairkey = ((uid << ITEM_BITS) + news) << TIME_BITS
            nxt = search(pairs, up[uid], up[uid + 1], pairkey + t + 1)
            end = search(pairs, nxt, up[uid + 1], pairkey + t + future + 1)
            output[at, 6] = end > nxt
            output[at, 7] = np.log1p(end - nxt)
            output[at, 8] = np.log1p(pairs[nxt] - pairkey - t) if end > nxt else 0.0
            output[at, 9] = np.log1p(article[last - 1] - key - t) if last > first else 0.0
            output[at, 13] = future / WEEK
            at += 1
    return output

def values(reader, rows, horizon, timeline):
    return forward_values(rows, reader.off, reader.cand, reader.t, reader.u, reader.uid,
                          reader.article, reader.ap, reader.ua, reader.up, horizon, timeline.traffic)
