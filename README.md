
# Organ Transplant Waiting List Simulation

A discrete-event simulation of an organ transplant waiting list using **SimPy**.
This project models patient arrivals, organ availability, blood-type compatibility,
urgency-based prioritization, and patient mortality while waiting.

The goal is to study fairness, waiting times, and outcomes under a simplified
organ allocation policy.

---

## 🚀 Features

- Poisson arrival of patients and donor organs
- Blood type (ABO) compatibility rules
- Priority-based organ allocation using:
  - Medical urgency
  - Waiting time
- Patient mortality risk while on the waiting list
- Detailed statistics and visual analytics
- Reproducible simulation with fixed random seed

---

## 🧠 Allocation Policy

Each compatible patient is assigned a priority score:

