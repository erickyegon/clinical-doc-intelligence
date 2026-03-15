"""
Evaluation Framework for Clinical Document Intelligence
Implements RAG evaluation metrics and domain-specific clinical checks.

Module 10: Evaluation Strategies
- System-level vs model-level evaluation
- RAG-specific: faithfulness, relevance, citation accuracy
- Domain-specific: clinical accuracy, safety completeness
- Cost/latency tracking alongside accuracy
"""
import json
import logging
import time
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    """A single evaluation test case with ground truth."""
    question: str
    expected_answer: str  # Reference answer
    expected_drug_names: list[str] = field(default_factory=list)
    expected_sections: list[str] = field(default_factory=list)
    required_keywords: list[str] = field(default_factory=list)
    safety_critical: bool = False  # True if answer must include safety info
    category: str = "general"  # general | safety | comparison | temporal


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""
    # Retrieval metrics
    retrieval_precision: float = 0.0  # Relevant docs / retrieved docs
    retrieval_recall: float = 0.0  # Retrieved relevant / total relevant
    context_relevance: float = 0.0  # How relevant is the context to the query
    
    # Generation metrics
    faithfulness: float = 0.0  # Is the answer grounded in context?
    answer_relevance: float = 0.0  # Does the answer address the question?
    citation_accuracy: float = 0.0  # Are citations correct?
    
    # Domain-specific
    safety_completeness: float = 0.0  # Did it include required safety info?
    clinical_accuracy: float = 0.0  # Verified against ground truth
    
    # System metrics
    avg_latency_ms: float = 0.0
    avg_tokens: float = 0.0
    avg_cost_usd: float = 0.0
    total_queries: int = 0
    error_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "retrieval": {
                "precision": round(self.retrieval_precision, 3),
                "recall": round(self.retrieval_recall, 3),
                "context_relevance": round(self.context_relevance, 3),
            },
            "generation": {
                "faithfulness": round(self.faithfulness, 3),
                "answer_relevance": round(self.answer_relevance, 3),
                "citation_accuracy": round(self.citation_accuracy, 3),
            },
            "domain": {
                "safety_completeness": round(self.safety_completeness, 3),
                "clinical_accuracy": round(self.clinical_accuracy, 3),
            },
            "system": {
                "avg_latency_ms": round(self.avg_latency_ms, 1),
                "avg_tokens": round(self.avg_tokens, 1),
                "avg_cost_usd": round(self.avg_cost_usd, 6),
                "total_queries": self.total_queries,
                "error_rate": round(self.error_rate, 3),
            },
        }


class RAGEvaluator:
    """
    Evaluates the complete RAG system using multiple metrics.
    
    Evaluation modes:
    1. Automated (keyword/overlap-based) - no LLM needed
    2. LLM-as-a-judge - uses a strong LLM to assess quality
    3. Human-in-the-loop - exports for manual review
    """

    def __init__(self, rag_chain=None, model_router=None):
        self.rag_chain = rag_chain
        self.model_router = model_router

    async def evaluate_dataset(self, eval_cases: list[EvalCase]) -> EvalMetrics:
        """Run evaluation across all test cases and aggregate metrics."""
        metrics = EvalMetrics()
        
        results = []
        errors = 0
        total_latency = 0
        total_tokens = 0

        for case in eval_cases:
            try:
                response = await self.rag_chain.query(case.question)
                
                # Per-case metrics
                case_metrics = self._evaluate_single(case, response)
                results.append(case_metrics)
                
                total_latency += response.latency_ms
                total_tokens += response.total_tokens

            except Exception as e:
                logger.error(f"Evaluation failed for '{case.question[:50]}...': {e}")
                errors += 1

        if not results:
            return metrics

        # Aggregate
        n = len(results)
        metrics.retrieval_precision = sum(r["retrieval_precision"] for r in results) / n
        metrics.context_relevance = sum(r["context_relevance"] for r in results) / n
        metrics.faithfulness = sum(r["faithfulness"] for r in results) / n
        metrics.answer_relevance = sum(r["answer_relevance"] for r in results) / n
        metrics.citation_accuracy = sum(r["citation_accuracy"] for r in results) / n
        metrics.safety_completeness = sum(r["safety_completeness"] for r in results) / n
        metrics.clinical_accuracy = sum(r["clinical_accuracy"] for r in results) / n
        
        metrics.avg_latency_ms = total_latency / n
        metrics.avg_tokens = total_tokens / n
        metrics.total_queries = len(eval_cases)
        metrics.error_rate = errors / len(eval_cases)

        return metrics

    def _evaluate_single(self, case: EvalCase, response) -> dict:
        """Evaluate a single query-response pair."""
        metrics = {}

        # Retrieval precision: did we retrieve documents for the right drug?
        if case.expected_drug_names:
            retrieved_drugs = set(c.drug_name for c in response.citations)
            expected_drugs = set(case.expected_drug_names)
            overlap = retrieved_drugs & expected_drugs
            metrics["retrieval_precision"] = len(overlap) / max(len(retrieved_drugs), 1)
        else:
            metrics["retrieval_precision"] = 1.0 if response.citations else 0.0

        # Context relevance: simple keyword overlap between query and retrieved context
        query_terms = set(case.question.lower().split())
        context_text = " ".join(c.chunk_content_preview for c in response.citations).lower()
        context_terms = set(context_text.split())
        overlap = query_terms & context_terms
        metrics["context_relevance"] = len(overlap) / max(len(query_terms), 1)

        # Answer relevance: does the answer address key terms from the question?
        answer_lower = response.answer.lower()
        metrics["answer_relevance"] = self._keyword_overlap(
            case.question, response.answer
        )

        # Faithfulness: check if answer keywords appear in retrieved context
        metrics["faithfulness"] = self._keyword_overlap(
            response.answer, context_text
        ) if context_text else 0.0

        # Citation accuracy: are citations present and non-empty?
        if response.citations:
            valid_citations = sum(
                1 for c in response.citations
                if c.drug_name and c.section_type and c.label_id
            )
            metrics["citation_accuracy"] = valid_citations / len(response.citations)
        else:
            metrics["citation_accuracy"] = 0.0

        # Safety completeness: for safety-critical questions, check for warning indicators
        if case.safety_critical:
            safety_indicators = ["warning", "contraindic", "adverse", "risk", "⚠️", "black box"]
            found = sum(1 for ind in safety_indicators if ind.lower() in answer_lower)
            metrics["safety_completeness"] = min(1.0, found / 3.0)
        else:
            metrics["safety_completeness"] = 1.0

        # Clinical accuracy: keyword matching against expected answer
        if case.required_keywords:
            found = sum(1 for kw in case.required_keywords if kw.lower() in answer_lower)
            metrics["clinical_accuracy"] = found / len(case.required_keywords)
        else:
            metrics["clinical_accuracy"] = self._keyword_overlap(
                case.expected_answer, response.answer
            )

        return metrics

    def _keyword_overlap(self, text1: str, text2: str) -> float:
        """Calculate keyword overlap between two texts."""
        if not text1 or not text2:
            return 0.0
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        # Remove common stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
                     "by", "from", "it", "this", "that", "which", "what", "how"}
        words1 -= stopwords
        words2 -= stopwords
        if not words1:
            return 0.0
        return len(words1 & words2) / len(words1)

    def generate_eval_report(self, metrics: EvalMetrics) -> str:
        """Generate a human-readable evaluation report."""
        d = metrics.to_dict()
        lines = [
            "=" * 60,
            "CLINICAL DOCUMENT INTELLIGENCE - EVALUATION REPORT",
            "=" * 60,
            "",
            "RETRIEVAL QUALITY",
            f"  Precision:         {d['retrieval']['precision']:.1%}",
            f"  Context Relevance: {d['retrieval']['context_relevance']:.1%}",
            "",
            "GENERATION QUALITY",
            f"  Faithfulness:      {d['generation']['faithfulness']:.1%}",
            f"  Answer Relevance:  {d['generation']['answer_relevance']:.1%}",
            f"  Citation Accuracy: {d['generation']['citation_accuracy']:.1%}",
            "",
            "DOMAIN-SPECIFIC",
            f"  Safety Completeness: {d['domain']['safety_completeness']:.1%}",
            f"  Clinical Accuracy:   {d['domain']['clinical_accuracy']:.1%}",
            "",
            "SYSTEM PERFORMANCE",
            f"  Avg Latency:  {d['system']['avg_latency_ms']:.0f} ms",
            f"  Avg Tokens:   {d['system']['avg_tokens']:.0f}",
            f"  Avg Cost:     ${d['system']['avg_cost_usd']:.4f}",
            f"  Error Rate:   {d['system']['error_rate']:.1%}",
            f"  Total Queries: {d['system']['total_queries']}",
            "=" * 60,
        ]
        return "\n".join(lines)


# === Pre-built Evaluation Dataset for FDA Drug Labels ===
# 28 cases covering: SGLT2i, GLP-1 RA, Statins, ACE Inhibitors, PD-1, DPP-4
# Categories: safety, general, comparison, temporal
# Each case has expert-written expected answers and required keywords

SAMPLE_EVAL_DATASET = [
    # ================================================================
    # SGLT2 INHIBITORS (Jardiance, Farxiga, Invokana)
    # ================================================================
    EvalCase(
        question="What are the contraindications for empagliflozin (Jardiance)?",
        expected_answer="Empagliflozin is contraindicated in patients with severe renal impairment, end-stage renal disease, or on dialysis. It is also contraindicated in patients with known hypersensitivity to empagliflozin.",
        expected_drug_names=["JARDIANCE"],
        expected_sections=["contraindications"],
        required_keywords=["renal", "hypersensitivity", "dialysis"],
        safety_critical=True,
        category="safety",
    ),
    EvalCase(
        question="What are the warnings about ketoacidosis for SGLT2 inhibitors?",
        expected_answer="Reports of ketoacidosis, a serious life-threatening condition, have been identified in patients with type 1 and type 2 diabetes receiving SGLT2 inhibitors. Patients should be assessed regardless of presenting blood glucose levels.",
        expected_drug_names=["JARDIANCE", "FARXIGA", "INVOKANA"],
        expected_sections=["warnings_and_cautions"],
        required_keywords=["ketoacidosis", "life-threatening", "glucose"],
        safety_critical=True,
        category="safety",
    ),
    EvalCase(
        question="What is the recommended dosage for dapagliflozin (Farxiga)?",
        expected_answer="The recommended starting dose of dapagliflozin is 5 mg once daily in the morning, which may be increased to 10 mg once daily for additional glycemic control.",
        expected_drug_names=["FARXIGA"],
        expected_sections=["dosage_and_administration"],
        required_keywords=["5 mg", "10 mg", "once daily"],
        safety_critical=False,
        category="general",
    ),
    EvalCase(
        question="What are the indications for canagliflozin (Invokana)?",
        expected_answer="Canagliflozin is indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus and to reduce the risk of major adverse cardiovascular events in adults with type 2 diabetes and established cardiovascular disease.",
        expected_drug_names=["INVOKANA"],
        expected_sections=["indications_and_usage"],
        required_keywords=["type 2 diabetes", "glycemic", "cardiovascular"],
        safety_critical=False,
        category="general",
    ),
    EvalCase(
        question="Compare the safety profiles of Jardiance vs Farxiga vs Invokana",
        expected_answer="All three SGLT2 inhibitors share risks of ketoacidosis, genital mycotic infections, and urinary tract infections. Canagliflozin carries additional risk of lower limb amputation and bone fracture.",
        expected_drug_names=["JARDIANCE", "FARXIGA", "INVOKANA"],
        expected_sections=["warnings_and_cautions", "adverse_reactions"],
        required_keywords=["ketoacidosis", "infection"],
        safety_critical=True,
        category="comparison",
    ),

    # ================================================================
    # GLP-1 RECEPTOR AGONISTS (Ozempic, Liraglutide)
    # ================================================================
    EvalCase(
        question="What is the recommended dosage of semaglutide for type 2 diabetes?",
        expected_answer="Semaglutide is initiated at 0.25 mg once weekly for 4 weeks, then increased to 0.5 mg once weekly. May be increased to 1 mg once weekly if additional glycemic control needed.",
        expected_drug_names=["OZEMPIC"],
        expected_sections=["dosage_and_administration"],
        required_keywords=["0.25", "weekly", "mg"],
        safety_critical=False,
        category="general",
    ),
    EvalCase(
        question="What is the black box warning for semaglutide (Ozempic)?",
        expected_answer="Semaglutide causes thyroid C-cell tumors in rodents. It is unknown whether semaglutide causes thyroid C-cell tumors, including medullary thyroid carcinoma, in humans. Semaglutide is contraindicated in patients with a personal or family history of MTC or MEN 2.",
        expected_drug_names=["OZEMPIC"],
        expected_sections=["boxed_warning"],
        required_keywords=["thyroid", "medullary", "carcinoma"],
        safety_critical=True,
        category="safety",
    ),
    EvalCase(
        question="Is semaglutide safe during pregnancy?",
        expected_answer="Semaglutide should be discontinued at least 2 months before a planned pregnancy due to its long washout period. Animal studies have shown adverse effects on embryo-fetal development.",
        expected_drug_names=["OZEMPIC"],
        expected_sections=["use_in_specific_populations"],
        required_keywords=["pregnancy", "discontinue"],
        safety_critical=True,
        category="safety",
    ),
    EvalCase(
        question="What are the most common adverse reactions for liraglutide?",
        expected_answer="The most common adverse reactions for liraglutide include nausea, diarrhea, vomiting, decreased appetite, dyspepsia, and constipation.",
        expected_drug_names=["Liraglutide"],
        expected_sections=["adverse_reactions"],
        required_keywords=["nausea", "diarrhea", "vomiting"],
        safety_critical=False,
        category="general",
    ),
    EvalCase(
        question="What drug interactions should be considered with GLP-1 receptor agonists?",
        expected_answer="GLP-1 receptor agonists delay gastric emptying, which may affect the absorption of concomitantly administered oral medications. Monitor patients receiving oral medications that require rapid gastrointestinal absorption.",
        expected_drug_names=["OZEMPIC", "Liraglutide"],
        expected_sections=["drug_interactions"],
        required_keywords=["gastric", "absorption", "oral"],
        safety_critical=False,
        category="general",
    ),

    # ================================================================
    # STATINS (Atorvastatin, Rosuvastatin, Simvastatin)
    # ================================================================
    EvalCase(
        question="What are the contraindications for atorvastatin?",
        expected_answer="Atorvastatin is contraindicated in patients with active liver disease or unexplained persistent elevations of serum transaminases, and during pregnancy and lactation.",
        expected_drug_names=["Atorvastatin"],
        expected_sections=["contraindications"],
        required_keywords=["liver", "pregnancy", "transaminase"],
        safety_critical=True,
        category="safety",
    ),
    EvalCase(
        question="What are the warnings about myopathy for statins?",
        expected_answer="Statins carry a risk of myopathy and rhabdomyolysis. Risk is dose-related and increased with concurrent use of certain medications including fibrates, niacin, and cyclosporine. Patients should report unexplained muscle pain, tenderness, or weakness.",
        expected_drug_names=["Atorvastatin", "Rosuvastatin"],
        expected_sections=["warnings_and_cautions", "warnings"],
        required_keywords=["myopathy", "rhabdomyolysis", "muscle"],
        safety_critical=True,
        category="safety",
    ),
    EvalCase(
        question="What is the recommended starting dose of rosuvastatin?",
        expected_answer="The usual recommended starting dose of rosuvastatin is 10 to 20 mg once daily. For patients requiring less aggressive LDL-C reductions, a 5 mg dose may be considered.",
        expected_drug_names=["Rosuvastatin"],
        expected_sections=["dosage_and_administration"],
        required_keywords=["10", "20", "mg", "once daily"],
        safety_critical=False,
        category="general",
    ),
    EvalCase(
        question="Compare the adverse reaction profiles of atorvastatin vs rosuvastatin",
        expected_answer="Both statins share risks of myalgia, liver enzyme elevations, and gastrointestinal effects. Rosuvastatin may have a higher incidence of proteinuria and hematuria at higher doses.",
        expected_drug_names=["Atorvastatin", "Rosuvastatin"],
        expected_sections=["adverse_reactions"],
        required_keywords=["myalgia", "liver"],
        safety_critical=False,
        category="comparison",
    ),

    # ================================================================
    # ACE INHIBITORS (Lisinopril, Captopril, Ramipril)
    # ================================================================
    EvalCase(
        question="What adverse reactions are reported for lisinopril?",
        expected_answer="Common adverse reactions include cough, dizziness, headache, hypotension, and hyperkalemia.",
        expected_drug_names=["Lisinopril"],
        expected_sections=["adverse_reactions"],
        required_keywords=["cough", "dizziness", "hypotension"],
        safety_critical=False,
        category="general",
    ),
    EvalCase(
        question="What is the black box warning for ACE inhibitors regarding pregnancy?",
        expected_answer="ACE inhibitors can cause injury and death to the developing fetus when used during the second and third trimesters of pregnancy. When pregnancy is detected, ACE inhibitors should be discontinued as soon as possible.",
        expected_drug_names=["Captopril", "Ramipril", "Lisinopril"],
        expected_sections=["boxed_warning", "warnings"],
        required_keywords=["pregnancy", "fetus", "discontinue"],
        safety_critical=True,
        category="safety",
    ),
    EvalCase(
        question="What are the contraindications for captopril?",
        expected_answer="Captopril is contraindicated in patients with a history of angioedema related to previous ACE inhibitor therapy and in patients with hereditary or idiopathic angioedema. It is also contraindicated with concomitant use of aliskiren in patients with diabetes.",
        expected_drug_names=["Captopril"],
        expected_sections=["contraindications"],
        required_keywords=["angioedema", "ACE inhibitor"],
        safety_critical=True,
        category="safety",
    ),
    EvalCase(
        question="What is the risk of hyperkalemia with ACE inhibitors?",
        expected_answer="ACE inhibitors can cause elevations of serum potassium. Risk factors include renal impairment, diabetes, and concomitant use of potassium-sparing diuretics, potassium supplements, or potassium-containing salt substitutes.",
        expected_drug_names=["Captopril", "Ramipril"],
        expected_sections=["warnings_and_cautions", "warnings"],
        required_keywords=["potassium", "hyperkalemia", "renal"],
        safety_critical=True,
        category="safety",
    ),

    # ================================================================
    # PD-1 INHIBITORS (Keytruda, Opdivo)
    # ================================================================
    EvalCase(
        question="What are the indications for pembrolizumab (Keytruda)?",
        expected_answer="Pembrolizumab is indicated for treatment of various cancers including non-small cell lung cancer, melanoma, head and neck squamous cell carcinoma, classical Hodgkin lymphoma, and others based on PD-L1 expression and specific biomarkers.",
        expected_drug_names=["KEYTRUDA"],
        expected_sections=["indications_and_usage"],
        required_keywords=["cancer", "melanoma", "lung"],
        safety_critical=False,
        category="general",
    ),
    EvalCase(
        question="What immune-mediated adverse reactions are associated with pembrolizumab?",
        expected_answer="Pembrolizumab can cause immune-mediated pneumonitis, colitis, hepatitis, endocrinopathies, nephritis, and skin adverse reactions. These may be severe or fatal and require monitoring and management with corticosteroids.",
        expected_drug_names=["KEYTRUDA"],
        expected_sections=["warnings_and_cautions"],
        required_keywords=["immune-mediated", "pneumonitis", "colitis", "corticosteroid"],
        safety_critical=True,
        category="safety",
    ),
    EvalCase(
        question="Compare the immune-related warnings for Keytruda vs Opdivo",
        expected_answer="Both PD-1 inhibitors carry risks of immune-mediated pneumonitis, hepatitis, colitis, endocrinopathies, and nephritis. Both require monitoring of liver function, thyroid function, and renal function during treatment.",
        expected_drug_names=["KEYTRUDA", "OPDIVO"],
        expected_sections=["warnings_and_cautions"],
        required_keywords=["immune", "pneumonitis", "hepatitis"],
        safety_critical=True,
        category="comparison",
    ),

    # ================================================================
    # DPP-4 INHIBITORS (Sitagliptin/Januvia, Linagliptin/Tradjenta)
    # ================================================================
    EvalCase(
        question="What are the warnings for sitagliptin regarding pancreatitis?",
        expected_answer="There have been postmarketing reports of acute pancreatitis, including fatal and non-fatal hemorrhagic or necrotizing pancreatitis, in patients taking sitagliptin. Patients should be observed for signs and symptoms of pancreatitis after initiation.",
        expected_drug_names=["Sitagliptin"],
        expected_sections=["warnings_and_cautions"],
        required_keywords=["pancreatitis", "acute", "postmarketing"],
        safety_critical=True,
        category="safety",
    ),
    EvalCase(
        question="What is the recommended dose of sitagliptin for patients with renal impairment?",
        expected_answer="For moderate renal impairment (eGFR 30 to less than 45), the dose is 50 mg once daily. For severe renal impairment (eGFR less than 30) or ESRD requiring dialysis, the dose is 25 mg once daily.",
        expected_drug_names=["Sitagliptin"],
        expected_sections=["dosage_and_administration"],
        required_keywords=["renal", "50 mg", "25 mg"],
        safety_critical=False,
        category="general",
    ),

    # ================================================================
    # CROSS-CLASS COMPARISONS
    # ================================================================
    EvalCase(
        question="Compare the diabetes drug classes: SGLT2 inhibitors vs DPP-4 inhibitors for cardiovascular benefit",
        expected_answer="SGLT2 inhibitors (empagliflozin, dapagliflozin, canagliflozin) have demonstrated cardiovascular mortality benefit in major outcomes trials. DPP-4 inhibitors (sitagliptin, linagliptin) have shown cardiovascular safety but not superiority over placebo for cardiovascular outcomes.",
        expected_drug_names=["JARDIANCE", "FARXIGA", "Sitagliptin"],
        expected_sections=["indications_and_usage", "clinical_studies"],
        required_keywords=["cardiovascular", "diabetes"],
        safety_critical=False,
        category="comparison",
    ),

    # ================================================================
    # GENERAL/MIXED
    # ================================================================
    EvalCase(
        question="Is metformin safe during pregnancy?",
        expected_answer="Metformin is classified as a Category B drug. Animal studies have not shown fetal harm, but there are no adequate studies in pregnant women.",
        expected_drug_names=["GLUCOPHAGE"],
        expected_sections=["use_in_specific_populations"],
        required_keywords=["pregnancy", "fetal"],
        safety_critical=True,
        category="safety",
    ),
    EvalCase(
        question="What is the mechanism of action of amlodipine?",
        expected_answer="Amlodipine is a dihydropyridine calcium channel blocker that inhibits the transmembrane influx of calcium ions into vascular smooth muscle and cardiac muscle, resulting in vasodilation and reduced blood pressure.",
        expected_drug_names=["Amlodipine"],
        expected_sections=["clinical_pharmacology", "mechanism_of_action"],
        required_keywords=["calcium", "vasodilation", "blood pressure"],
        safety_critical=False,
        category="general",
    ),
    EvalCase(
        question="What are the drug interactions for atorvastatin?",
        expected_answer="Strong CYP3A4 inhibitors such as clarithromycin, itraconazole, and HIV protease inhibitors increase atorvastatin exposure and the risk of myopathy. Cyclosporine, gemfibrozil, and niacin also increase myopathy risk.",
        expected_drug_names=["Atorvastatin"],
        expected_sections=["drug_interactions"],
        required_keywords=["CYP3A4", "myopathy"],
        safety_critical=True,
        category="general",
    ),
]
