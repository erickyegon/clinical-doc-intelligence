"""
Section-Aware Chunker for FDA Drug Labels
Chunks by label section boundaries first, then applies size-based splitting
only for sections that exceed the max chunk size.

Module 7: Chunking Strategies — "When to chunk, when NOT to chunk"
"""
import re
import hashlib
import logging
from typing import Optional

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class SectionAwareChunker:
    """
    Intelligent chunker that respects FDA label section boundaries.
    
    Strategy:
    - Short sections (< chunk_size): Keep as single chunk (preserve context)
    - Long sections: Split with overlap, preserving sentence boundaries
    - Black Box Warnings: NEVER split (safety-critical, must be complete)
    - Metadata: Inherited from parent label + section-level tags
    """

    # Sections that should never be split regardless of size
    NEVER_SPLIT = {"boxed_warning", "contraindications"}

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_size: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_label(self, label) -> list[dict]:
        """
        Chunk an entire DrugLabel into indexed documents.
        
        Returns list of dicts with keys: id, content, metadata
        """
        chunks = []

        for section_key, section_data in label.sections.items():
            content = section_data["content"]
            display_name = section_data["display_name"]

            if not content or len(content.strip()) < self.min_chunk_size:
                continue

            base_metadata = {
                "source_type": "fda_label",
                "label_id": label.label_id,
                "drug_name": label.drug_name,
                "generic_name": label.generic_name,
                "manufacturer": label.manufacturer,
                "section_type": section_key,
                "section_display_name": display_name,
                "approval_date": label.approval_date or "",
                "therapeutic_area": label.therapeutic_area or "",
            }

            # Safety-critical sections: never split
            if section_key in self.NEVER_SPLIT:
                chunk_id = self._make_id(label.label_id, section_key, 0)
                chunks.append({
                    "id": chunk_id,
                    "content": f"[{display_name}] {content}",
                    "metadata": {**base_metadata, "chunk_index": 0, "total_chunks": 1},
                })
                continue

            # Short sections: keep as single chunk
            if len(content) <= self.chunk_size:
                chunk_id = self._make_id(label.label_id, section_key, 0)
                chunks.append({
                    "id": chunk_id,
                    "content": f"[{display_name}] {content}",
                    "metadata": {**base_metadata, "chunk_index": 0, "total_chunks": 1},
                })
                continue

            # Long sections: split with sentence-boundary awareness
            section_chunks = self._split_with_overlap(content)
            for i, chunk_text in enumerate(section_chunks):
                chunk_id = self._make_id(label.label_id, section_key, i)
                chunks.append({
                    "id": chunk_id,
                    "content": f"[{display_name} ({i+1}/{len(section_chunks)})] {chunk_text}",
                    "metadata": {
                        **base_metadata,
                        "chunk_index": i,
                        "total_chunks": len(section_chunks),
                    },
                })

        logger.info(
            f"Chunked {label.drug_name} ({label.label_id}): "
            f"{len(label.sections)} sections → {len(chunks)} chunks"
        )
        return chunks

    def _split_with_overlap(self, text: str) -> list[str]:
        """
        Split text into chunks respecting sentence boundaries.
        Uses overlap to maintain context across chunk boundaries.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))

                # Calculate overlap: keep last N characters worth of sentences
                overlap_chunk = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_length += len(s) + 1
                    else:
                        break

                current_chunk = overlap_chunk
                current_length = overlap_length

            current_chunk.append(sentence)
            current_length += sentence_len + 1

        # Don't forget the last chunk
        if current_chunk:
            final = " ".join(current_chunk)
            if len(final) >= self.min_chunk_size:
                chunks.append(final)
            elif chunks:
                # Merge short final chunk with previous
                chunks[-1] += " " + final

        return chunks

    def _make_id(self, label_id: str, section: str, index: int) -> str:
        """Generate a deterministic chunk ID."""
        raw = f"{label_id}_{section}_{index}"
        return hashlib.md5(raw.encode()).hexdigest()

    def estimate_chunks(self, label) -> int:
        """Estimate chunk count without actually chunking (for planning)."""
        total = 0
        for section_key, section_data in label.sections.items():
            content = section_data["content"]
            if len(content) < self.min_chunk_size:
                continue
            if section_key in self.NEVER_SPLIT or len(content) <= self.chunk_size:
                total += 1
            else:
                total += max(1, len(content) // (self.chunk_size - self.chunk_overlap))
        return total
