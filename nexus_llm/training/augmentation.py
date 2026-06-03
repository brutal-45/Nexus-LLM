"""Data augmentation: random deletion, swap, insertion, synonym replacement, back-translation."""

import random
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Common English synonyms for augmentation
_SYNONYM_MAP: Dict[str, List[str]] = {
    "good": ["great", "fine", "excellent", "nice", "wonderful"],
    "bad": ["poor", "terrible", "awful", "horrible", "dreadful"],
    "big": ["large", "huge", "enormous", "vast", "massive"],
    "small": ["tiny", "little", "miniature", "compact", "minute"],
    "fast": ["quick", "rapid", "swift", "speedy", "hasty"],
    "slow": ["sluggish", "unhurried", "gradual", "leisurely", "deliberate"],
    "happy": ["joyful", "cheerful", "delighted", "pleased", "content"],
    "sad": ["unhappy", "sorrowful", "melancholy", "gloomy", "downcast"],
    "important": ["significant", "crucial", "vital", "essential", "critical"],
    "easy": ["simple", "effortless", "straightforward", "uncomplicated", "painless"],
    "hard": ["difficult", "challenging", "tough", "demanding", "arduous"],
    "old": ["aged", "ancient", "elderly", "mature", "vintage"],
    "new": ["fresh", "novel", "recent", "modern", "contemporary"],
    "hot": ["warm", "scorching", "boiling", "heated", "fiery"],
    "cold": ["chilly", "freezing", "frigid", "icy", "frosty"],
    "beautiful": ["gorgeous", "stunning", "attractive", "lovely", "elegant"],
    "ugly": ["unsightly", "hideous", "unattractive", "grotesque", "plain"],
    "smart": ["intelligent", "clever", "brilliant", "bright", "sharp"],
    "strong": ["powerful", "mighty", "robust", "sturdy", "tough"],
    "weak": ["feeble", "frail", "fragile", "delicate", "flimsy"],
}


class TextAugmenter:
    """Applies various text augmentation techniques for training data augmentation."""

    def __init__(
        self,
        seed: Optional[int] = None,
        synonym_map: Optional[Dict[str, List[str]]] = None,
    ):
        self.rng = random.Random(seed)
        self.synonym_map = synonym_map or _SYNONYM_MAP

    def random_deletion(
        self,
        text: str,
        deletion_prob: float = 0.1,
    ) -> str:
        """Randomly delete words from the text.

        Args:
            text: Input text.
            deletion_prob: Probability of deleting each word.

        Returns:
            Augmented text with some words removed.
        """
        words = text.split()
        if len(words) <= 1:
            return text

        retained = [w for w in words if self.rng.random() > deletion_prob]

        if not retained:
            retained = [words[self.rng.randint(0, len(words) - 1)]]

        return " ".join(retained)

    def random_swap(
        self,
        text: str,
        num_swaps: Optional[int] = None,
    ) -> str:
        """Randomly swap two words in the text.

        Args:
            text: Input text.
            num_swaps: Number of swaps to perform. Defaults to len(words) // 2.

        Returns:
            Augmented text with words swapped.
        """
        words = text.split()
        if len(words) <= 1:
            return text

        num_swaps = num_swaps or max(1, len(words) // 2)

        for _ in range(num_swaps):
            idx1 = self.rng.randint(0, len(words) - 1)
            idx2 = self.rng.randint(0, len(words) - 1)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return " ".join(words)

    def random_insertion(
        self,
        text: str,
        num_insertions: Optional[int] = None,
    ) -> str:
        """Randomly insert synonyms of existing words into the text.

        Args:
            text: Input text.
            num_insertions: Number of insertions to perform.

        Returns:
            Augmented text with synonyms inserted.
        """
        words = text.split()
        if not words:
            return text

        num_insertions = num_insertions or max(1, len(words) // 4)

        for _ in range(num_insertions):
            word_idx = self.rng.randint(0, len(words) - 1)
            synonym = self._get_synonym(words[word_idx])
            if synonym:
                insert_pos = self.rng.randint(0, len(words))
                words.insert(insert_pos, synonym)

        return " ".join(words)

    def synonym_replacement(
        self,
        text: str,
        replacement_prob: float = 0.1,
    ) -> str:
        """Replace words with their synonyms.

        Args:
            text: Input text.
            replacement_prob: Probability of replacing each word with a synonym.

        Returns:
            Augmented text with some words replaced by synonyms.
        """
        words = text.split()
        new_words = []

        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in self.synonym_map and self.rng.random() < replacement_prob:
                synonym = self.rng.choice(self.synonym_map[clean_word])
                if word[0].isupper():
                    synonym = synonym.capitalize()
                new_words.append(synonym)
            else:
                new_words.append(word)

        return " ".join(new_words)

    def back_translation(
        self,
        text: str,
        intermediate_lang: str = "fr",
        translator: Optional[Any] = None,
    ) -> str:
        """Apply back-translation augmentation.

        Translate text to an intermediate language and back to the original.
        Requires a translator object with translate() method.

        Args:
            text: Input text.
            intermediate_lang: Intermediate language code.
            translator: Translator object with translate(text, dest, src) method.

        Returns:
            Back-translated text, or original if translator is unavailable.
        """
        if translator is None:
            logger.debug("No translator provided for back-translation. Returning original text.")
            return text

        try:
            translated = translator.translate(text, dest=intermediate_lang, src="en")
            back_translated = translator.translate(translated.text, dest="en", src=intermediate_lang)
            return back_translated.text
        except Exception as e:
            logger.warning(f"Back-translation failed: {e}")
            return text

    def _get_synonym(self, word: str) -> Optional[str]:
        """Get a random synonym for a word."""
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word in self.synonym_map:
            return self.rng.choice(self.synonym_map[clean_word])
        return None

    def augment(
        self,
        text: str,
        methods: Optional[List[str]] = None,
        num_augmented: int = 1,
        **kwargs,
    ) -> List[str]:
        """Apply multiple augmentation methods to generate augmented texts.

        Args:
            text: Input text.
            methods: List of augmentation methods to apply.
                    Options: "deletion", "swap", "insertion", "synonym", "back_translation"
            num_augmented: Number of augmented versions to generate.
            **kwargs: Additional arguments for specific methods.

        Returns:
            List of augmented text strings.
        """
        methods = methods or ["deletion", "swap", "synonym"]
        method_map = {
            "deletion": self.random_deletion,
            "swap": self.random_swap,
            "insertion": self.random_insertion,
            "synonym": self.synonym_replacement,
            "back_translation": self.back_translation,
        }

        augmented = []
        for _ in range(num_augmented):
            method = self.rng.choice(methods)
            if method in method_map:
                try:
                    result = method_map[method](text, **kwargs)
                    augmented.append(result)
                except Exception as e:
                    logger.warning(f"Augmentation method {method} failed: {e}")
                    augmented.append(text)

        return augmented

    def augment_batch(
        self,
        texts: List[str],
        methods: Optional[List[str]] = None,
        augmentation_factor: int = 2,
        **kwargs,
    ) -> List[str]:
        """Augment a batch of texts.

        Args:
            texts: List of input texts.
            methods: Augmentation methods to apply.
            augmentation_factor: Number of augmented versions per input text.
            **kwargs: Additional arguments.

        Returns:
            List containing original and augmented texts.
        """
        result = list(texts)
        for text in texts:
            augmented = self.augment(text, methods=methods, num_augmented=augmentation_factor, **kwargs)
            result.extend(augmented)
        return result

    def augment_dataset(
        self,
        data: List[Dict[str, Any]],
        text_field: str = "text",
        methods: Optional[List[str]] = None,
        augmentation_factor: int = 1,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Augment an entire dataset.

        Args:
            data: List of data dictionaries.
            text_field: Field containing the text to augment.
            methods: Augmentation methods to apply.
            augmentation_factor: Number of augmented copies per data point.
            **kwargs: Additional arguments.

        Returns:
            List containing original and augmented data.
        """
        result = list(data)
        for item in data:
            text = item.get(text_field, "")
            if not text:
                continue
            augmented_texts = self.augment(text, methods=methods, num_augmented=augmentation_factor, **kwargs)
            for aug_text in augmented_texts:
                new_item = dict(item)
                new_item[text_field] = aug_text
                new_item["augmented"] = True
                result.append(new_item)
        return result
