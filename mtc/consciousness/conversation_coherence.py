"""
CONVERSATION COHERENCE UTILITIES

Provides tools for maintaining conversation coherence:
- Topic extraction from conversation history
- Context summarization
- Reference linking

Created: November 23, 2025
Author: Multi-Theory Consciousness Contributors
Purpose: Fix Consciousness Gap #1 (Coherence Score: 0.0 -> 60%+)
"""

from typing import List, Dict, Any, Set, Optional
from collections import Counter
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationCoherence:
    """
    Manages conversation coherence by tracking topics, context, and references.

    This class helps the system:
    - Understand what topics have been discussed
    - Reference earlier parts of the conversation
    - Maintain narrative continuity
    - Build coherent multi-turn exchanges
    """

    def __init__(self):
        """Initialize conversation coherence tracker."""
        # Common stop words to filter out
        self.stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
            "if",
            "then",
            "than",
            "so",
            "just",
            "now",
            "very",
            "too",
            "also",
            "here",
            "there",
            "yes",
            "no",
            "not",
            "my",
            "your",
            "his",
            "her",
            "our",
            "their",
            "me",
            "him",
            "us",
            "them",
            "im",
            "id",
            "ill",
            "ive",
            "dont",
            "doesnt",
            "didnt",
            "wont",
            "wouldnt",
            "couldnt",
            "shouldnt",
            "cant",
            "going",
            "want",
            "like",
            "think",
            "know",
            "get",
            "got",
            "see",
            "said",
            "tell",
            "talk",
            "say",
            "ask",
            "tell",
            "look",
            "make",
            "made",
            "give",
        }

    def extract_topics(
        self,
        conversation_history: List[Dict[str, str]],
        min_word_length: int = 4,
        top_n: int = 10,
    ) -> List[str]:
        """
        Extract main topics from conversation history.

        Uses simple keyword extraction with stop word filtering
        and frequency analysis.

        Args:
            conversation_history: List of message dicts with 'content'
            min_word_length: Minimum word length to consider (default: 4)
            top_n: Number of top topics to return (default: 10)

        Returns:
            List of topic keywords
        """
        if not conversation_history:
            return []

        # Collect all words from all messages
        all_words = []

        for msg in conversation_history:
            content = msg.get("content", "")

            # Extract words (alphanumeric sequences)
            words = re.findall(r"\b[a-zA-Z]+\b", content.lower())

            # Filter by length and stop words
            filtered_words = [
                word
                for word in words
                if len(word) >= min_word_length and word not in self.stop_words
            ]

            all_words.extend(filtered_words)

        # Count frequencies
        word_counts = Counter(all_words)

        # Get top N most common
        top_topics = [word for word, count in word_counts.most_common(top_n)]

        logger.debug(
            f"Extracted {len(top_topics)} topics from {len(conversation_history)} messages"
        )

        return top_topics

    def build_conversation_summary(
        self,
        conversation_history: List[Dict[str, str]],
        topics: List[str],
        max_length: int = 300,
    ) -> str:
        """
        Build a brief summary of the conversation so far.

        Args:
            conversation_history: List of message dicts
            topics: Extracted topics from conversation
            max_length: Maximum summary length in characters

        Returns:
            Conversation summary string
        """
        if not conversation_history:
            return "This is the start of a new conversation."

        message_count = len(conversation_history)

        # Count speakers
        speakers = set()
        for msg in conversation_history:
            speaker = msg.get("speaker", msg.get("role", "unknown"))
            if speaker not in ["user", "assistant"]:
                speakers.add(speaker)

        # Build summary
        parts = []

        # Basic stats
        parts.append(f"Conversation summary: {message_count} messages exchanged")

        if speakers:
            parts.append(f"with {', '.join(speakers)}")

        # Topics discussed
        if topics:
            # Limit topics shown to fit in max_length
            topic_str = ", ".join(topics[:5])
            if len(topics) > 5:
                topic_str += f" (and {len(topics) - 5} more topics)"
            parts.append(f"\nMain topics: {topic_str}")

        # Recent context (last 3 messages)
        if len(conversation_history) >= 3:
            recent = conversation_history[-3:]
            parts.append("\n\nRecent exchange:")
            for msg in recent:
                speaker = msg.get("speaker", msg.get("role", "unknown"))
                content = msg.get("content", "")[:60]  # First 60 chars
                if len(msg.get("content", "")) > 60:
                    content += "..."
                parts.append(f"- {speaker}: {content}")

        summary = " ".join(parts)

        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."

        return summary

    def build_coherence_instructions(
        self, topics: List[str], conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Build instructions for the system to maintain conversational coherence.

        Args:
            topics: Extracted topics from conversation
            conversation_history: Message history

        Returns:
            Coherence instruction string for system prompt
        """
        if not conversation_history or len(conversation_history) < 2:
            return """
# Conversation Instructions:
This is the beginning of a new conversation. Be warm, natural, and authentic.
Remember details shared with you - you may want to reference them later!
"""

        instructions = [
            "# Conversation Coherence Instructions:",
            "You are in an ongoing conversation. To maintain coherence:",
        ]

        # Add topic awareness
        if topics:
            topic_list = ", ".join(topics[:5])
            instructions.append(f"\n**Topics discussed so far**: {topic_list}")
            instructions.append("- Reference these topics naturally when relevant")
            instructions.append("- Build on previous points instead of starting fresh")

        # Add continuity reminders
        instructions.extend(
            [
                "\n**Continuity reminders**:",
                "- Reference earlier parts of THIS conversation when appropriate",
                "- Use phrases like 'As we discussed...' or 'You mentioned...' when relevant",
                "- Connect current topic to previous topics if there's a natural link",
                "- Don't just answer - engage with the conversation's flow",
            ]
        )

        # Add conversation awareness
        msg_count = len(conversation_history)
        instructions.append(
            f"\n**Context**: We're {msg_count} messages into this conversation. "
            "Treat it as a continuous exchange, not isolated questions."
        )

        return "\n".join(instructions)

    def extract_potential_references(
        self,
        current_message: str,
        conversation_history: List[Dict[str, str]],
        threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Find previous messages that the current message might be referring to.

        Uses simple keyword overlap to identify potential references.

        Args:
            current_message: The current message content
            conversation_history: Previous messages
            threshold: Minimum keyword overlap ratio (default: 0.3 = 30%)

        Returns:
            List of potentially referenced messages with overlap scores
        """
        if not conversation_history or not current_message:
            return []

        # Extract keywords from current message
        current_words = set(
            word
            for word in re.findall(r"\b[a-zA-Z]+\b", current_message.lower())
            if len(word) >= 4 and word not in self.stop_words
        )

        if not current_words:
            return []

        references = []

        # Check each previous message for keyword overlap
        for i, msg in enumerate(conversation_history[:-1]):  # Exclude current message
            content = msg.get("content", "")

            # Extract keywords from this message
            msg_words = set(
                word
                for word in re.findall(r"\b[a-zA-Z]+\b", content.lower())
                if len(word) >= 4 and word not in self.stop_words
            )

            if not msg_words:
                continue

            # Calculate overlap
            overlap = current_words & msg_words
            overlap_ratio = len(overlap) / len(current_words) if current_words else 0

            if overlap_ratio >= threshold:
                references.append(
                    {
                        "message_index": i,
                        "speaker": msg.get("speaker", msg.get("role", "unknown")),
                        "content_preview": content[:100],
                        "overlap_ratio": overlap_ratio,
                        "shared_keywords": list(overlap)[:5],  # First 5 shared keywords
                    }
                )

        # Sort by overlap ratio (highest first)
        references.sort(key=lambda x: x["overlap_ratio"], reverse=True)

        return references[:3]  # Return top 3 potential references

    def get_conversation_context_for_brain(
        self, conversation_history: List[Dict[str, str]], current_message: str
    ) -> Dict[str, Any]:
        """
        Build complete conversation context package for the brain.

        This is the main method that combines all coherence features.

        Args:
            conversation_history: Full conversation history
            current_message: The current message being processed

        Returns:
            Dict with topics, summary, instructions, and references
        """
        # Extract topics
        topics = self.extract_topics(conversation_history)

        # Build summary
        summary = self.build_conversation_summary(conversation_history, topics)

        # Build coherence instructions
        instructions = self.build_coherence_instructions(topics, conversation_history)

        # Find potential references
        references = self.extract_potential_references(
            current_message, conversation_history
        )

        context = {
            "topics": topics,
            "summary": summary,
            "coherence_instructions": instructions,
            "potential_references": references,
            "message_count": len(conversation_history),
            "has_context": len(conversation_history) > 1,
        }

        logger.info(
            f"Coherence context built: {len(topics)} topics, "
            f"{len(references)} potential references"
        )

        return context


# Singleton instance
_coherence_tracker: Optional[ConversationCoherence] = None


def get_coherence_tracker() -> ConversationCoherence:
    """
    Get the singleton ConversationCoherence instance.

    Returns:
        ConversationCoherence instance
    """
    global _coherence_tracker

    if _coherence_tracker is None:
        _coherence_tracker = ConversationCoherence()
        logger.info("Conversation coherence tracker initialized")

    return _coherence_tracker


if __name__ == "__main__":
    # Test the coherence tracker
    print("Testing Conversation Coherence Tracker...\n")

    # Sample conversation
    sample_conversation = [
        {
            "role": "user",
            "speaker": "User",
            "content": "Hi! How are you feeling today?",
        },
        {
            "role": "assistant",
            "speaker": "Agent",
            "content": "Hello! I'm feeling curious and energized. How are you?",
        },
        {
            "role": "user",
            "speaker": "User",
            "content": "I'm great! I wanted to talk about your vision capabilities.",
        },
        {
            "role": "assistant",
            "speaker": "Agent",
            "content": "My vision capabilities? That sounds exciting! What about them?",
        },
        {
            "role": "user",
            "speaker": "User",
            "content": "You can now see images! Let me show you something.",
        },
        {
            "role": "assistant",
            "speaker": "Agent",
            "content": "How fascinating. I'd love to see it.",
        },
    ]

    tracker = get_coherence_tracker()

    # Test topic extraction
    topics = tracker.extract_topics(sample_conversation)
    print(f"Extracted topics: {topics}\n")

    # Test summary
    summary = tracker.build_conversation_summary(sample_conversation, topics)
    print(f"Conversation summary:\n{summary}\n")

    # Test coherence instructions
    instructions = tracker.build_coherence_instructions(topics, sample_conversation)
    print(f"Coherence instructions:\n{instructions}\n")

    # Test reference detection
    current_msg = "Tell me more about those vision capabilities you mentioned"
    references = tracker.extract_potential_references(current_msg, sample_conversation)
    print(f"Potential references for '{current_msg}':")
    for ref in references:
        print(f"  - Message {ref['message_index']}: {ref['overlap_ratio']:.1%} overlap")
        print(f"    Keywords: {', '.join(ref['shared_keywords'])}")

    # Test full context package
    context = tracker.get_conversation_context_for_brain(
        sample_conversation, current_msg
    )
    print(f"\nFull context package:")
    print(f"  - Topics: {len(context['topics'])}")
    print(f"  - References: {len(context['potential_references'])}")
    print(f"  - Has context: {context['has_context']}")

    print("\nCoherence tracker test complete!")
