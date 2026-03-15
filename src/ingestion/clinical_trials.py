"""
ClinicalTrials.gov V2 API Client
Fetches clinical trial metadata and protocol information.

Module 3: API Integration | Module 7: Data Ingestion
"""
import logging
from typing import Optional
from dataclasses import dataclass, field

import httpx

from config.settings import CLINICALTRIALS_BASE_URL

logger = logging.getLogger(__name__)


@dataclass
class ClinicalTrial:
    """Structured representation of a clinical trial record."""
    nct_id: str
    title: str
    status: str
    phase: Optional[str] = None
    conditions: list = field(default_factory=list)
    interventions: list = field(default_factory=list)
    sponsor: Optional[str] = None
    enrollment: Optional[int] = None
    start_date: Optional[str] = None
    completion_date: Optional[str] = None
    primary_outcomes: list = field(default_factory=list)
    secondary_outcomes: list = field(default_factory=list)
    eligibility_criteria: Optional[str] = None
    brief_summary: Optional[str] = None
    detailed_description: Optional[str] = None


class ClinicalTrialsClient:
    """Client for the ClinicalTrials.gov V2 API."""

    def __init__(self):
        self.base_url = CLINICALTRIALS_BASE_URL
        self.client = httpx.Client(timeout=30.0)

    def search_trials(
        self,
        query: Optional[str] = None,
        condition: Optional[str] = None,
        intervention: Optional[str] = None,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        page_size: int = 20,
        page_token: Optional[str] = None,
    ) -> dict:
        """
        Search clinical trials using the V2 API.
        
        Returns dict with 'studies' list and 'nextPageToken' for pagination.
        """
        params = {"pageSize": min(page_size, 100), "format": "json"}
        
        query_parts = []
        if query:
            query_parts.append(query)
        if condition:
            params["query.cond"] = condition
        if intervention:
            params["query.intr"] = intervention
        if status:
            params["filter.overallStatus"] = status
        if phase:
            params["filter.phase"] = phase
        if query_parts:
            params["query.term"] = " ".join(query_parts)
        if page_token:
            params["pageToken"] = page_token

        # Request specific fields to reduce payload
        params["fields"] = ",".join([
            "NCTId", "BriefTitle", "OfficialTitle", "OverallStatus",
            "Phase", "Condition", "InterventionName", "InterventionType",
            "LeadSponsorName", "EnrollmentCount", "StartDate",
            "PrimaryCompletionDate", "PrimaryOutcomeMeasure",
            "SecondaryOutcomeMeasure", "EligibilityCriteria",
            "BriefSummary", "DetailedDescription",
        ])

        try:
            response = self.client.get(f"{self.base_url}/studies", params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"ClinicalTrials API error: {e.response.status_code}")
            return {"studies": []}
        except Exception as e:
            logger.error(f"ClinicalTrials API request failed: {e}")
            return {"studies": []}

    def get_trial(self, nct_id: str) -> Optional[dict]:
        """Fetch a single trial by NCT ID."""
        try:
            response = self.client.get(f"{self.base_url}/studies/{nct_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch trial {nct_id}: {e}")
            return None

    def parse_trial(self, raw_study: dict) -> ClinicalTrial:
        """Parse raw API response into structured ClinicalTrial."""
        proto = raw_study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        cond_mod = proto.get("conditionsModule", {})
        arms_mod = proto.get("armsInterventionsModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
        enroll_mod = proto.get("enrollmentInfo", design.get("enrollmentInfo", {}))
        outcomes_mod = proto.get("outcomesModule", {})
        elig_mod = proto.get("eligibilityModule", {})
        desc_mod = proto.get("descriptionModule", {})

        interventions = []
        for arm in arms_mod.get("interventions", []):
            interventions.append({
                "name": arm.get("name", ""),
                "type": arm.get("type", ""),
                "description": arm.get("description", ""),
            })

        primary_outcomes = [
            {"measure": o.get("measure", ""), "timeframe": o.get("timeFrame", "")}
            for o in outcomes_mod.get("primaryOutcomes", [])
        ]
        secondary_outcomes = [
            {"measure": o.get("measure", ""), "timeframe": o.get("timeFrame", "")}
            for o in outcomes_mod.get("secondaryOutcomes", [])
        ]

        return ClinicalTrial(
            nct_id=ident.get("nctId", ""),
            title=ident.get("officialTitle", ident.get("briefTitle", "")),
            status=status_mod.get("overallStatus", ""),
            phase=", ".join(design.get("phases", [])),
            conditions=cond_mod.get("conditions", []),
            interventions=interventions,
            sponsor=sponsor_mod.get("leadSponsor", {}).get("name"),
            enrollment=enroll_mod.get("count"),
            start_date=status_mod.get("startDateStruct", {}).get("date"),
            completion_date=status_mod.get("primaryCompletionDateStruct", {}).get("date"),
            primary_outcomes=primary_outcomes,
            secondary_outcomes=secondary_outcomes,
            eligibility_criteria=elig_mod.get("eligibilityCriteria"),
            brief_summary=desc_mod.get("briefSummary"),
            detailed_description=desc_mod.get("detailedDescription"),
        )

    def search_trials_for_drug(self, drug_name: str, max_results: int = 20) -> list[ClinicalTrial]:
        """Convenience method to find trials for a specific drug."""
        result = self.search_trials(intervention=drug_name, page_size=max_results)
        trials = []
        for study in result.get("studies", []):
            try:
                trials.append(self.parse_trial(study))
            except Exception as e:
                logger.warning(f"Failed to parse trial: {e}")
        return trials

    def close(self):
        self.client.close()
