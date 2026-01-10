"""Multi-frame aggregation for license plate recognition."""

from collections import defaultdict
from typing import Dict, List, Optional

from src.lpr.plate_types import PlateResult


class PlateAggregator:
    """
    Aggregates license plate OCR results across multiple frames.
    
    Uses character-level voting and confidence weighting to improve accuracy.
    """

    def __init__(self, min_confidence: float = 0.3, min_agreement: float = 0.5) -> None:
        """
        Initialize aggregator.

        Args:
            min_confidence: Minimum confidence to consider a result
            min_agreement: Minimum agreement ratio for final result (0.0-1.0)
        """
        self.min_confidence = min_confidence
        self.min_agreement = min_agreement
        # Store per-vehicle track: list of (frame_id, PlateResult)
        self.plate_history: Dict[int, List[tuple[int, PlateResult]]] = defaultdict(list)

    def add_result(
        self, vehicle_track_id: int, frame_id: int, plate_result: PlateResult
    ) -> None:
        """
        Add a plate result for a vehicle track.

        Args:
            vehicle_track_id: Vehicle track ID
            frame_id: Frame number
            plate_result: Plate detection/OCR result
        """
        # Only add if we have text and confidence meets threshold
        if (
            plate_result.text
            and plate_result.text != "UNKNOWN"
            and plate_result.confidence >= self.min_confidence
        ):
            self.plate_history[vehicle_track_id].append((frame_id, plate_result))

    def aggregate(self, vehicle_track_id: int) -> Optional[PlateResult]:
        """
        Aggregate plate results for a vehicle track.

        Args:
            vehicle_track_id: Vehicle track ID

        Returns:
            Aggregated PlateResult or None if insufficient data
        """
        if vehicle_track_id not in self.plate_history:
            return None

        results = self.plate_history[vehicle_track_id]
        if not results:
            return None

        # If only one result, return it
        if len(results) == 1:
            frame_id, plate_result = results[0]
            return PlateResult(
                text=plate_result.text,
                confidence=plate_result.confidence,
                bbox=plate_result.bbox,
                vehicle_track_id=vehicle_track_id,
            )

        # Character-level voting
        aggregated_text, text_confidence = self._character_level_voting(results)

        # If agreement is too low, return None
        if text_confidence < self.min_agreement:
            return None

        # Get best frame (highest confidence) for bbox
        best_frame_id, best_result = max(results, key=lambda x: x[1].confidence)

        return PlateResult(
            text=aggregated_text,
            confidence=text_confidence,
            bbox=best_result.bbox,
            vehicle_track_id=vehicle_track_id,
        )

    def _character_level_voting(
        self, results: List[tuple[int, PlateResult]]
    ) -> tuple[str, float]:
        """
        Perform character-level voting across multiple OCR results.

        Args:
            results: List of (frame_id, PlateResult) tuples

        Returns:
            Tuple of (aggregated_text, confidence)
        """
        # Extract texts and confidences
        texts = [r[1].text for r in results]
        confidences = [r[1].confidence for r in results]

        # Find maximum length (handle variable-length plates)
        max_len = max(len(text) for text in texts if text)

        if max_len == 0:
            return ("UNKNOWN", 0.0)

        # Character-level voting
        char_votes: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for text, conf in zip(texts, confidences):
            # Pad shorter texts (assume missing chars are at end)
            padded_text = text.ljust(max_len, "?")
            for pos, char in enumerate(padded_text):
                if char != "?":
                    char_votes[pos][char] += conf

        # Build aggregated text
        aggregated_chars = []
        total_confidence = 0.0

        for pos in range(max_len):
            if pos not in char_votes:
                break

            # Get character with highest weighted votes
            char_scores = char_votes[pos]
            if not char_scores:
                break

            best_char = max(char_scores.items(), key=lambda x: x[1])[0]
            char_confidence = char_scores[best_char] / sum(char_scores.values())

            aggregated_chars.append(best_char)
            total_confidence += char_confidence

        aggregated_text = "".join(aggregated_chars).rstrip("?")

        # Average confidence across positions
        avg_confidence = total_confidence / len(aggregated_chars) if aggregated_chars else 0.0

        # Also consider overall agreement (how many results agree)
        agreement_ratio = self._calculate_agreement(texts, aggregated_text)
        final_confidence = (avg_confidence + agreement_ratio) / 2.0

        return (aggregated_text, final_confidence)

    def _calculate_agreement(self, texts: List[str], aggregated_text: str) -> float:
        """
        Calculate agreement ratio between texts and aggregated result.

        Args:
            texts: List of individual OCR results
            aggregated_text: Aggregated result

        Returns:
            Agreement ratio (0.0-1.0)
        """
        if not texts or not aggregated_text:
            return 0.0

        # Count how many texts match aggregated result (allowing for minor differences)
        matches = 0
        for text in texts:
            # Exact match
            if text == aggregated_text:
                matches += 1
            # Similar match (Levenshtein distance <= 1 for short strings)
            elif len(text) == len(aggregated_text):
                diff = sum(c1 != c2 for c1, c2 in zip(text, aggregated_text))
                if diff <= 1:  # Allow 1 character difference
                    matches += 0.8  # Partial match

        return matches / len(texts) if texts else 0.0

    def clear_track(self, vehicle_track_id: int) -> None:
        """Clear history for a specific vehicle track."""
        if vehicle_track_id in self.plate_history:
            del self.plate_history[vehicle_track_id]

    def clear_all(self) -> None:
        """Clear all plate history."""
        self.plate_history.clear()

    def get_track_count(self, vehicle_track_id: int) -> int:
        """Get number of plate results for a track."""
        return len(self.plate_history.get(vehicle_track_id, []))




