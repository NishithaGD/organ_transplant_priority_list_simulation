"""
Simulation of Organ Transplant Waiting List Management (with graphs)

How to run (in terminal, inside this folder):
    python organ_transplant_sim.py

Before first run, install packages:
    pip install simpy numpy matplotlib
"""

import simpy
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# GLOBAL PARAMETERS (you can change these)
# ------------------------------------------------
RANDOM_SEED = 42
SIM_TIME = 365 * 5          # total time in days

PATIENT_ARRIVAL_RATE = 2.0  # patients per day
ORGAN_ARRIVAL_RATE = 1.0    # organs per day

# priority weights
W_URGENCY = 0.7
W_WAITING = 0.3

# death rate on waiting list
BASE_DEATH_RATE = 0.0005
URGENCY_DEATH_MULTIPLIER = 0.002

# blood type distribution
BLOOD_TYPES = ["O", "A", "B", "AB"]
BLOOD_PROBS = [0.44, 0.42, 0.10, 0.04]


# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------
def sample_blood_type():
    return np.random.choice(BLOOD_TYPES, p=BLOOD_PROBS)


def blood_compatible(donor_bt, patient_bt):
    """Very simple ABO compatibility rules."""
    if donor_bt == "O":
        return True
    if donor_bt == "A":
        return patient_bt in ["A", "AB"]
    if donor_bt == "B":
        return patient_bt in ["B", "AB"]
    if donor_bt == "AB":
        return patient_bt == "AB"
    return False


# ------------------------------------------------
# DATA CLASSES
# ------------------------------------------------
class Patient:
    def __init__(self, pid, env_now):
        self.id = pid
        self.blood_type = sample_blood_type()
        self.age = np.random.randint(18, 75)
        self.urgency = np.random.uniform(0, 100)  # 0–100
        self.join_time = env_now                  # time patient joined list
        self.status = "waiting"                   # waiting / transplanted / dead

    def __repr__(self):
        return f"Patient(id={self.id}, BT={self.blood_type}, urg={self.urgency:.1f}, status={self.status})"


class Stats:
    def __init__(self):
        # 👉 This is the attribute that caused the earlier error
        self.patients_arrived = 0

        self.transplants = 0
        self.deaths = 0
        self.organs_discarded = 0
        self.waiting_times = []            # list of waiting times (days)
        self.waiting_times_by_bt = {bt: [] for bt in BLOOD_TYPES}

        self.outcome_counts = {           # for pie chart
            "transplanted": 0,
            "died_waiting": 0,
            "still_waiting": 0,
        }


# ------------------------------------------------
# PROCESSES
# ------------------------------------------------
def patient_arrival_process(env, waiting_list, stats):
    """Generate patients according to a Poisson process."""
    pid = 0
    while True:
        inter_arrival = np.random.exponential(1.0 / PATIENT_ARRIVAL_RATE)
        yield env.timeout(inter_arrival)

        pid += 1
        stats.patients_arrived += 1

        p = Patient(pid, env.now)
        waiting_list.append(p)

        # start that patient’s death process
        env.process(patient_death_process(env, p, waiting_list, stats))


def patient_death_process(env, patient, waiting_list, stats):
    """Simulate when a patient might die while waiting."""
    urg_norm = patient.urgency / 100.0
    death_rate = BASE_DEATH_RATE + URGENCY_DEATH_MULTIPLIER * urg_norm  # per day

    if death_rate <= 0:
        return

    time_to_death = np.random.exponential(1.0 / death_rate)
    yield env.timeout(time_to_death)

    if patient.status == "waiting":
        patient.status = "dead"
        stats.deaths += 1
        if patient in waiting_list:
            waiting_list.remove(patient)


def organ_arrival_process(env, waiting_list, stats):
    """Generate organs and try to allocate them."""
    organ_id = 0
    while True:
        inter_arrival = np.random.exponential(1.0 / ORGAN_ARRIVAL_RATE)
        yield env.timeout(inter_arrival)

        organ_id += 1
        donor_bt = sample_blood_type()
        donor_age = np.random.randint(18, 65)
        organ_quality = np.random.uniform(0, 1)  # not used yet, but kept for realism

        allocate_organ(env, donor_bt, donor_age, organ_quality, waiting_list, stats)


def allocate_organ(env, donor_bt, donor_age, organ_quality, waiting_list, stats):
    """Pick the highest-priority compatible patient and transplant."""
    compatible = [
        p for p in waiting_list
        if p.status == "waiting" and blood_compatible(donor_bt, p.blood_type)
    ]

    if not compatible:
        stats.organs_discarded += 1
        return

    best_patient = None
    best_score = -1.0

    for p in compatible:
        urg_norm = p.urgency / 100.0
        waiting_days = env.now - p.join_time
        waiting_norm = waiting_days / SIM_TIME  # normalize

        priority = W_URGENCY * urg_norm + W_WAITING * waiting_norm

        if priority > best_score:
            best_score = priority
            best_patient = p

    if best_patient is not None:
        best_patient.status = "transplanted"
        stats.transplants += 1

        wait_time = env.now - best_patient.join_time
        stats.waiting_times.append(wait_time)
        stats.waiting_times_by_bt[best_patient.blood_type].append(wait_time)

        if best_patient in waiting_list:
            waiting_list.remove(best_patient)


# ------------------------------------------------
# RUN + PRINT + PLOT
# ------------------------------------------------
def run_simulation():
    np.random.seed(RANDOM_SEED)

    env = simpy.Environment()
    waiting_list = []
    stats = Stats()

    env.process(patient_arrival_process(env, waiting_list, stats))
    env.process(organ_arrival_process(env, waiting_list, stats))

    env.run(until=SIM_TIME)

    # classify remaining patients as "still waiting"
    still_waiting = sum(1 for p in waiting_list if p.status == "waiting")
    stats.outcome_counts["transplanted"] = stats.transplants
    stats.outcome_counts["died_waiting"] = stats.deaths
    stats.outcome_counts["still_waiting"] = still_waiting

    print_results(stats)
    plot_results(stats)


def print_results(stats: Stats):
    print("=" * 60)
    print(" SIMULATION RESULTS: Organ Transplant Waiting List ")
    print("=" * 60)
    print(f"Simulation time (days):            {SIM_TIME}")
    print(f"Total patients arrived:            {stats.patients_arrived}")
    print(f"Total transplants performed:       {stats.transplants}")
    print(f"Total deaths on waiting list:      {stats.deaths}")
    print(f"Total organs discarded:            {stats.organs_discarded}")

    if stats.patients_arrived > 0:
        print(f"Death probability on waiting list: {stats.deaths / stats.patients_arrived:.3f}")

    if stats.waiting_times:
        print(f"Average waiting time (all):        {np.mean(stats.waiting_times):.2f} days")

    print("\nAverage waiting time by blood type:")
    for bt in BLOOD_TYPES:
        wts = stats.waiting_times_by_bt[bt]
        if wts:
            print(f"  {bt}: {np.mean(wts):.2f} days (n={len(wts)})")
        else:
            print(f"  {bt}: No transplants")

    print("\nOutcome distribution (counts):")
    for k, v in stats.outcome_counts.items():
        print(f"  {k}: {v}")

    print("=" * 60)
    print("Graphs saved in this folder.")
    print("=" * 60)


def plot_results(stats: Stats):
    """Create and save graphs as PNG files."""

    # 1) Histogram of waiting times
    if stats.waiting_times:
        plt.figure()
        plt.hist(stats.waiting_times, bins=30)
        plt.title("Distribution of Waiting Time to Transplant")
        plt.xlabel("Waiting time (days)")
        plt.ylabel("Number of patients")
        plt.tight_layout()
        plt.savefig("waiting_time_histogram.png", dpi=300)
        plt.close()

    # 2) Average waiting time by blood type
    labels = []
    avg_waits = []
    for bt in BLOOD_TYPES:
        wts = stats.waiting_times_by_bt[bt]
        labels.append(bt)
        avg_waits.append(np.mean(wts) if wts else 0.0)

    plt.figure()
    plt.bar(labels, avg_waits)
    plt.title("Average Waiting Time by Blood Type")
    plt.xlabel("Blood type")
    plt.ylabel("Average waiting time (days)")
    plt.tight_layout()
    plt.savefig("avg_wait_by_blood_type.png", dpi=300)
    plt.close()

    # 3) Pie chart of outcomes
    labels = list(stats.outcome_counts.keys())
    sizes = list(stats.outcome_counts.values())

    plt.figure()
    plt.pie(sizes, labels=labels, autopct="%1.1f%%")
    plt.title("Patient Outcomes at End of Simulation")
    plt.tight_layout()
    plt.savefig("patient_outcomes_pie.png", dpi=300)
    plt.close()

    print("\nGraphs generated:")
    print("  waiting_time_histogram.png")
    print("  avg_wait_by_blood_type.png")
    print("  patient_outcomes_pie.png")


if __name__ == "__main__":
    run_simulation()