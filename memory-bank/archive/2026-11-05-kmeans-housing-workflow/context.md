# Context

## Goal

Implement the 4Geeks K-Means housing tutorial on this template repository.

## Scope

Load the root `housing.csv`, cluster `Latitude`, `Longitude`, and `MedInc` into six groups, imitate those labels with a supervised classifier, save both workflows, and document the measured results.

## Constraints

Keep all 20,640 rows. Do not predict `MedHouseVal`. Do not change the cluster count. Do not call the database helper. Do not commit or push as part of the implementation.

## Relevant files

`housing.csv`, `src/app.py`, `src/kmeans_housing.py`, `src/utils.py`, `src/explore.ipynb`, `tests/test_workflow.py`, `README.md`.

## Dependencies

The existing `requirements.txt` plus `pytest`.

## Background

The specification is `ML-K-Means-Context.md`. The raw file is already at the repository root. `data/raw/` is empty.
