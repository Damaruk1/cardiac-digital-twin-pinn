"""
lead_mapping.py
-----------------
Standard clinical correspondence between 12-lead ECG leads and the
heart wall region each lead is most sensitive to. This is the same
convention cardiologists use to localize a myocardial infarction from
a 12-lead ECG.

IMPORTANT: this mapping is a well-established clinical heuristic, not
a precise physical measurement -- real localization also depends on
patient anatomy variation. We use it here as a reasonable, standard
approximation to drive the anatomical visualization.
"""

from typing import Dict, List

# Wall region names used consistently across this module and heart_mesh.py
ANATOMICAL_REGIONS = ["anterior", "inferior", "lateral", "septal"]

LEAD_TO_REGIONS: Dict[str, List[str]] = {
    "I": ["lateral"],
    "II": ["inferior"],
    "III": ["inferior"],
    "aVR": [],  # no reliable wall correlation -- views the cavity, not a wall
    "aVL": ["lateral"],
    "aVF": ["inferior"],
    "V1": ["septal"],
    "V2": ["septal"],
    "V3": ["anterior"],
    "V4": ["anterior"],
    "V5": ["lateral"],
    "V6": ["lateral"],
    # MIT-BIH's modified limb lead II approximates standard lead II
    "MLII": ["inferior"],
}


def get_regions_for_leads(lead_names: List[str]) -> List[str]:
    """
    Returns the unique set of anatomical regions implicated by a given
    list of lead names, preserving a stable region order.

    Args:
        lead_names: e.g. ["MLII", "V5"]

    Returns:
        List of region names implicated by at least one of the given leads.
    """
    implicated = set()
    for lead in lead_names:
        implicated.update(LEAD_TO_REGIONS.get(lead, []))

    # Return in the canonical order for consistent downstream display
    return [region for region in ANATOMICAL_REGIONS if region in implicated]
