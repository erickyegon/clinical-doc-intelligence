"""
FDA Drug Label Ingestion Client
Pulls structured drug label data from the openFDA API (api.fda.gov)
and DailyMed SPL files.

Module 3: API Integration | Module 7: Data Ingestion & Parsing
"""
import json
import logging
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import httpx

from config.settings import (
    OPENFDA_BASE_URL,
    OPENFDA_API_KEY,
    SAMPLE_LABELS_DIR,
)

logger = logging.getLogger(__name__)


@dataclass
class DrugLabel:
    """Structured representation of an FDA drug label."""
    label_id: str
    drug_name: str
    generic_name: str
    manufacturer: str
    approval_date: Optional[str] = None
    therapeutic_area: Optional[str] = None
    sections: dict = field(default_factory=dict)
    raw_data: dict = field(default_factory=dict)

    # Standard SPL sections mapped to human-readable names
    SECTION_MAP = {
        "indications_and_usage": "Indications and Usage",
        "contraindications": "Contraindications",
        "warnings_and_cautions": "Warnings and Precautions",
        "boxed_warning": "Black Box Warning",
        "dosage_and_administration": "Dosage and Administration",
        "adverse_reactions": "Adverse Reactions",
        "drug_interactions": "Drug Interactions",
        "use_in_specific_populations": "Use in Specific Populations",
        "clinical_pharmacology": "Clinical Pharmacology",
        "overdosage": "Overdosage",
        "description": "Description",
        "how_supplied": "How Supplied",
        "patient_medication_information": "Patient Medication Information",
        "mechanism_of_action": "Mechanism of Action",
        "pharmacodynamics": "Pharmacodynamics",
        "pharmacokinetics": "Pharmacokinetics",
        "clinical_studies": "Clinical Studies",
        "pregnancy": "Pregnancy",
        "pediatric_use": "Pediatric Use",
        "geriatric_use": "Geriatric Use",
        "warnings": "Warnings",
        "precautions": "Precautions",
    }


class FDALabelClient:
    """Client for fetching drug labels from the openFDA API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENFDA_API_KEY
        self.base_url = OPENFDA_BASE_URL
        self.client = httpx.Client(timeout=30.0)
        self._request_count = 0
        self._last_request_time = 0

    def _rate_limit(self):
        """Respect openFDA rate limits: 240 requests/minute with key, 40 without."""
        limit = 240 if self.api_key else 40
        min_interval = 60.0 / limit
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1

    def search_labels(
        self,
        drug_name: Optional[str] = None,
        therapeutic_area: Optional[str] = None,
        manufacturer: Optional[str] = None,
        limit: int = 10,
        skip: int = 0,
    ) -> list[dict]:
        """
        Search FDA drug labels using openFDA API.
        
        Args:
            drug_name: Brand or generic drug name
            therapeutic_area: Pharmacologic class
            manufacturer: Manufacturer name
            limit: Results per page (max 100)
            skip: Offset for pagination
            
        Returns:
            List of raw label records from openFDA
        """
        from urllib.parse import quote

        search_parts = []
        if drug_name:
            safe_name = quote(drug_name, safe='')
            search_parts.append(
                f'(openfda.brand_name:"{safe_name}"+openfda.generic_name:"{safe_name}")'
            )
        if therapeutic_area:
            safe_area = quote(therapeutic_area, safe='')
            search_parts.append(
                f'openfda.pharm_class_epc:"{safe_area}"'
            )
        if manufacturer:
            safe_mfr = quote(manufacturer, safe='')
            search_parts.append(
                f'openfda.manufacturer_name:"{safe_mfr}"'
            )

        # Build URL manually to avoid httpx double-encoding the openFDA search syntax
        url = f"{self.base_url}?limit={min(limit, 100)}&skip={skip}"
        if search_parts:
            search_str = "+AND+".join(search_parts)
            url += f"&search={search_str}"
        if self.api_key:
            url += f"&api_key={self.api_key}"

        self._rate_limit()

        try:
            response = self.client.get(url)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            total = data.get("meta", {}).get("results", {}).get("total", 0)
            logger.info(f"Found {total} labels, returned {len(results)}")
            return results
        except httpx.HTTPStatusError as e:
            logger.error(f"FDA API error: {e.response.status_code} - {e.response.text[:200]}")
            return []
        except Exception as e:
            logger.error(f"FDA API request failed: {e}")
            return []

    def fetch_labels_by_drug_class(
        self, drug_class: str, max_results: int = 50
    ) -> list[dict]:
        """Fetch all labels for a pharmacologic drug class."""
        all_results = []
        skip = 0
        batch_size = 100

        while len(all_results) < max_results:
            batch = self.search_labels(
                therapeutic_area=drug_class,
                limit=min(batch_size, max_results - len(all_results)),
                skip=skip,
            )
            if not batch:
                break
            all_results.extend(batch)
            skip += batch_size
            logger.info(f"Fetched {len(all_results)}/{max_results} labels for {drug_class}")

        return all_results[:max_results]

    def parse_label(self, raw_label: dict) -> DrugLabel:
        """
        Parse a raw openFDA label record into a structured DrugLabel.
        
        The openFDA API returns labels with sections as top-level arrays.
        Each section value is a list of strings (usually one element).
        """
        openfda = raw_label.get("openfda", {})

        label = DrugLabel(
            label_id=raw_label.get("id", raw_label.get("set_id", "unknown")),
            drug_name=self._first_or_default(openfda.get("brand_name", []), "Unknown"),
            generic_name=self._first_or_default(openfda.get("generic_name", []), "Unknown"),
            manufacturer=self._first_or_default(
                openfda.get("manufacturer_name", []), "Unknown"
            ),
            approval_date=raw_label.get("effective_time"),
            therapeutic_area=self._first_or_default(
                openfda.get("pharm_class_epc", []), None
            ),
            raw_data=raw_label,
        )

        # Extract all standard label sections
        for api_key, display_name in DrugLabel.SECTION_MAP.items():
            content = raw_label.get(api_key)
            if content:
                # openFDA returns sections as lists of strings
                text = content[0] if isinstance(content, list) else str(content)
                # Clean HTML artifacts from SPL
                text = self._clean_label_text(text)
                if text.strip():
                    label.sections[api_key] = {
                        "display_name": display_name,
                        "content": text,
                    }

        return label

    def _clean_label_text(self, text: str) -> str:
        """Remove HTML tags and clean up label text."""
        import re
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove bullet artifacts
        text = re.sub(r"[•●]", "- ", text)
        return text.strip()

    def _first_or_default(self, lst: list, default):
        return lst[0] if lst else default

    def close(self):
        self.client.close()


class LocalLabelLoader:
    """Load pre-downloaded FDA labels from local JSON files."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or SAMPLE_LABELS_DIR

    def load_all(self) -> list[dict]:
        """Load all JSON label files from the data directory."""
        labels = []
        if not self.data_dir.exists():
            logger.warning(f"Sample labels directory not found: {self.data_dir}")
            return labels

        for json_file in sorted(self.data_dir.glob("*.json")):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        labels.extend(data)
                    else:
                        labels.append(data)
                logger.info(f"Loaded {json_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")

        logger.info(f"Loaded {len(labels)} labels from {self.data_dir}")
        return labels

    def save_labels(self, labels: list[dict], filename: str):
        """Save fetched labels to local storage for offline use."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.data_dir / filename
        with open(filepath, "w") as f:
            json.dump(labels, f, indent=2)
        logger.info(f"Saved {len(labels)} labels to {filepath}")
