import os.path

from common import *

# ============================================================
# Parameters
# ============================================================

np.random.seed(0)

NUM_REQUESTS = 5000

ARRIVAL_RATE = 0.08  # requests/s
PREFILL_TOKENS = 32
DECODE_TOKENS = 64

PREFILL_TIME = 0.05  # s/token
DECODE_TIME = 0.1  # s/token

HIGH_PRIORITY_FRACTION = 0.1

# ============================================================
# Generate workload
# ============================================================

arrival_times = np.cumsum(
    np.random.exponential(
        scale=1 / ARRIVAL_RATE,
        size=NUM_REQUESTS
    )
)

priorities = np.random.rand(NUM_REQUESTS) < HIGH_PRIORITY_FRACTION


class Request:
    def __init__(self, idx, arrival, high):
        self.id = idx
        self.arrival = arrival
        self.high = high

        self.prefill_remaining = PREFILL_TOKENS
        self.decode_remaining = DECODE_TOKENS

        self.started = False
        self.start_time = None


requests = [
    Request(i, arrival_times[i], priorities[i])
    for i in range(NUM_REQUESTS)
]


# ============================================================
# Simulator
# ============================================================

def simulate(requests, preemptive):
    # Copy state
    reqs = [
        Request(r.id, r.arrival, r.high)
        for r in requests
    ]

    for r, old in zip(reqs, requests):
        r.prefill_remaining = old.prefill_remaining
        r.decode_remaining = old.decode_remaining

    time = 0
    next_arrival = 0

    waiting = []

    completed = []

    current = None

    while len(completed) < len(reqs):

        # Add arrivals
        while (
                next_arrival < len(reqs)
                and reqs[next_arrival].arrival <= time
        ):
            waiting.append(reqs[next_arrival])
            next_arrival += 1

        # Select next request
        if current is None:

            if waiting:

                if preemptive:

                    # high priority first
                    waiting.sort(
                        key=lambda x: (
                            not x.high,
                            x.arrival
                        )
                    )

                else:
                    waiting.sort(
                        key=lambda x: x.arrival
                    )

                current = waiting.pop(0)

                if not current.started:
                    current.started = True
                    current.start_time = time

            else:
                # idle until next request
                time = reqs[next_arrival].arrival
                continue

        # Execute one scheduling quantum
        if current.prefill_remaining > 0:

            time += current.prefill_remaining * PREFILL_TIME
            current.prefill_remaining = 0

        elif current.decode_remaining > 0:

            current.decode_remaining -= 1
            time += DECODE_TIME

        # Completed?
        if (
                current.prefill_remaining == 0
                and current.decode_remaining == 0
        ):
            completed.append(current)
            current = None

        # Preemption point
        elif preemptive and current.decode_remaining > 0:

            # Check if high-priority work is waiting
            if any(r.high for r in waiting):
                waiting.append(current)
                current = None

    queue_times = np.array([
        r.start_time - r.arrival
        for r in reqs
    ])

    return queue_times, reqs


# ============================================================
# Run experiments
# ============================================================

fcfs_queue, _ = simulate(
    requests,
    preemptive=False
)

prio_queue, prio_reqs = simulate(
    requests,
    preemptive=True
)

prio_high = prio_queue[priorities]
prio_low = prio_queue[~priorities]

df = pd.DataFrame({
    "Queue Time (s)": np.concatenate(
        [
            fcfs_queue,
            prio_high,
            prio_low,
        ]
    ),
    "Policy": (
            ["FCFS"] * len(fcfs_queue)
            +
            ["High Priority"] * len(prio_high)
            +
            ["Low Priority"] * len(prio_low)
    )
})

# ============================================================
# Plot CDF
# ============================================================

fig, ax = plt.subplots(figsize=(figwidth_third, 1.6))

sns.ecdfplot(
    data=df,
    x="Queue Time (s)",
    hue="Policy",
    ax=ax,
    palette=palette[:3]
)

# Compact legend so it fits the third-width figure
sns.move_legend(ax, "lower right", title=None)

ax.set_xlabel("Queue Time (s)")
ax.set_ylabel("Cumulative Distribution")

ax.set_title(left_better_str, fontsize=FONTSIZE, color="navy")
# sns.despine()

fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(plots_dir, "mock", "queue_time_cdf.pdf"))

# ============================================================
# Print summary
# ============================================================

print(
    df.groupby("Policy")["Queue Time (s)"]
    .quantile([0.5, 0.95, 0.99])
    .unstack()
    .round(2)
)
