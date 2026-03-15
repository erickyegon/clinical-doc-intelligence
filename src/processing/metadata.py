"""
Metadata Extraction and Enrichment
Extracts and normalizes metadata for filtering and scoped retrieval.

Module 7: Metadata Design & Filtering
"""
import re
import logging

logger = logging.getLogger(__name__)

# Therapeutic area normalization map
THERAPEUTIC_AREA_MAP = {
    "Sodium-Glucose Transporter 2 Inhibitor": "Diabetes - SGLT2i",
    "SGLT2 Inhibitor": "Diabetes - SGLT2i",
    "Glucagon-Like Peptide-1 Receptor Agonist": "Diabetes - GLP1-RA",
    "GLP-1 Receptor Agonist": "Diabetes - GLP1-RA",
    "Dipeptidyl Peptidase-4 Inhibitor": "Diabetes - DPP4i",
    "DPP-4 Inhibitor": "Diabetes - DPP4i",
    "HMG-CoA Reductase Inhibitor": "Cardiovascular - Statin",
    "Statin": "Cardiovascular - Statin",
    "Angiotensin Receptor Blocker": "Cardiovascular - ARB",
    "Proton Pump Inhibitor": "Gastrointestinal - PPI",
    "Selective Serotonin Reuptake Inhibitor": "Psychiatry - SSRI",
    "Programmed Death Receptor-1 Blocking Antibody": "Oncology - PD1",
    "Kinase Inhibitor": "Oncology - Kinase Inhibitor",
    "Tumor Necrosis Factor Blocker": "Immunology - TNF",
}

# Safety signal keywords
SAFETY_KEYWORDS = {
    "black_box": ["black box", "boxed warning", "WARNING:"],
    "rems": ["REMS", "Risk Evaluation and Mitigation"],
    "contraindication": ["contraindicated", "must not", "do not use"],
    "pregnancy_risk": ["pregnancy category", "fetal harm", "contraindicated in pregnancy"],
}


class MetadataExtractor:
    """Extracts and enriches metadata for improved retrieval filtering."""

    def enrich(self, metadata: dict, label) -> dict:
        """
        Enrich chunk metadata with derived fields.
        
        Adds:
        - normalized_therapeutic_area: Standardized therapeutic area
        - has_boxed_warning: Boolean
        - safety_flags: List of safety-relevant tags
        - section_priority: Numeric priority for ranking
        """
        metadata = dict(metadata)

        # Normalize therapeutic area for consistent filtering
        raw_area = metadata.get("therapeutic_area", "")
        metadata["normalized_therapeutic_area"] = (
            THERAPEUTIC_AREA_MAP.get(raw_area, raw_area) if raw_area else "Other"
        )

        # Check for boxed warning presence
        metadata["has_boxed_warning"] = "boxed_warning" in label.sections

        # Extract safety flags from content
        section_type = metadata.get("section_type", "")
        section_content = ""
        if section_type in label.sections:
            section_content = label.sections[section_type].get("content", "")
        
        safety_flags = []
        for flag_name, keywords in SAFETY_KEYWORDS.items():
            if any(kw.lower() in section_content.lower() for kw in keywords):
                safety_flags.append(flag_name)
        metadata["safety_flags"] = ",".join(safety_flags) if safety_flags else ""

        # Section priority for retrieval ranking boost
        metadata["section_priority"] = self._get_section_priority(section_type)

        # Extract route of administration if available
        raw = label.raw_data
        openfda = raw.get("openfda", {})
        routes = openfda.get("route", [])
        metadata["route"] = routes[0] if routes else ""

        # Dosage form
        forms = openfda.get("dosage_form", [])
        metadata["dosage_form"] = forms[0] if forms else ""

        return metadata

    def _get_section_priority(self, section_type: str) -> int:
        """
        Assign priority scores to sections for retrieval boosting.
        Higher = more clinically important = should rank higher.
        """
        priorities = {
            "boxed_warning": 10,
            "contraindications": 9,
            "warnings_and_cautions": 8,
            "warnings": 8,
            "adverse_reactions": 7,
            "drug_interactions": 7,
            "dosage_and_administration": 6,
            "indications_and_usage": 6,
            "use_in_specific_populations": 5,
            "clinical_studies": 5,
            "clinical_pharmacology": 4,
            "mechanism_of_action": 4,
            "pharmacokinetics": 3,
            "description": 2,
            "how_supplied": 1,
        }
        return priorities.get(section_type, 3)

    def extract_drug_class_from_text(self, text: str) -> list[str]:
        """Extract potential drug class mentions from free text."""
        classes = []
        text_lower = text.lower()
        for pattern, normalized in THERAPEUTIC_AREA_MAP.items():
            if pattern.lower() in text_lower:
                classes.append(normalized)
        return list(set(classes))
