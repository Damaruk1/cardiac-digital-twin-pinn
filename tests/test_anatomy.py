"""
test_anatomy.py
-----------------
Phase 11 tests: verify the lead-to-region mapping is complete and
correct, and that the heart mesh generates valid, correctly labeled
geometry.

Run with:
    pytest tests/test_anatomy.py -v
"""

import numpy as np

from src.anatomy.heart_mesh import generate_heart_mesh
from src.anatomy.lead_mapping import (
    ANATOMICAL_REGIONS,
    LEAD_TO_REGIONS,
    get_regions_for_leads,
)


def test_all_standard_12_leads_are_mapped():
    standard_leads = ["I", "II", "III", "aVR", "aVL", "aVF",
                       "V1", "V2", "V3", "V4", "V5", "V6"]
    for lead in standard_leads:
        assert lead in LEAD_TO_REGIONS, f"Missing mapping for standard lead: {lead}"


def test_avr_has_no_region_correlation():
    """aVR is clinically known to not localize to a specific wall."""
    assert LEAD_TO_REGIONS["aVR"] == []


def test_get_regions_for_leads_combines_and_deduplicates():
    # V5 and V6 both map to lateral -- should appear once, not twice
    regions = get_regions_for_leads(["V5", "V6"])
    assert regions == ["lateral"]


def test_get_regions_for_leads_handles_mitbih_leads():
    regions = get_regions_for_leads(["MLII", "V5"])
    assert "inferior" in regions
    assert "lateral" in regions
    assert len(regions) == 2


def test_get_regions_for_leads_handles_unknown_lead_gracefully():
    """An unrecognized lead name shouldn't crash, just contribute nothing."""
    regions = get_regions_for_leads(["NOT_A_REAL_LEAD"])
    assert regions == []


def test_heart_mesh_generates_correct_point_count():
    mesh = generate_heart_mesh(n_theta=10, n_phi=20)
    assert mesh.vertices.shape == (200, 3)
    assert mesh.region_labels.shape == (200,)


def test_heart_mesh_all_labels_are_valid_regions():
    mesh = generate_heart_mesh(n_theta=15, n_phi=20)
    valid_labels = set(ANATOMICAL_REGIONS) | {"posterior"}
    assert set(mesh.region_labels).issubset(valid_labels)


def test_heart_mesh_apex_points_are_labeled_inferior():
    """Points near theta=0 (the apex) should be labeled 'inferior',
    regardless of their azimuthal angle."""
    mesh = generate_heart_mesh(n_theta=20, n_phi=20, inferior_fraction=0.25)
    # The first n_phi rows correspond to the smallest theta values (near apex)
    apex_labels = mesh.region_labels[: mesh.n_phi]
    assert all(label == "inferior" for label in apex_labels)


def test_heart_mesh_vertices_are_finite():
    mesh = generate_heart_mesh()
    assert np.all(np.isfinite(mesh.vertices))
