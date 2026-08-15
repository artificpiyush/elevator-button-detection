import re

class FloorSelector2:
    """
    Precision floor selector with visual confusion matching and ligature resolution.
    
    Matching cascade:
    1. Exact match (case & whitespace normalized)
    2. Verified elevator button synonyms / OCR ligatures (e.g. '11' -> 'M', '112' -> '12')
    3. Leading zero / multi-digit prefix resolution
    4. Visually-weighted Levenshtein distance with strict thresholds
    """

    # Visual character confusion sets
    CONFUSION_GROUPS = [
        {'0', 'O', 'D', 'Q'},
        {'1', 'I', 'L', 'l', '|', '!'},
        {'8', 'B'},
        {'5', 'S'},
        {'2', 'Z'},
        {'6', 'G'},
        {'4', 'A'},
        {'9', 'Q'}
    ]

    # Common OCR segmentations and ligatures for elevator buttons
    TOKEN_SYNONYMS = {
        "M": ["11", "1L", "1I", "1M", "N", "W"],
        "9M": ["1M", "GM", "0M", "9N", "9W"],
        "12": ["112", "I2", "L2"],
        "11": ["II", "LL", "1I", "I1"],
        "0": ["O", "D", "Q", "00"],
        "1": ["I", "L", "l", "|", "/"]
    }

    @classmethod
    def clean_text(cls, text):
        """
        Sanitizes text by stripping whitespace and non-alphanumeric noise.
        """
        if not text:
            return ""
        return re.sub(r'[^A-Za-z0-9]', '', str(text)).strip().upper()

    @classmethod
    def _char_substitution_cost(cls, c1, c2):
        if c1 == c2:
            return 0.0
        for group in cls.CONFUSION_GROUPS:
            if c1 in group and c2 in group:
                return 0.2
        return 1.0

    @classmethod
    def visual_edit_distance(cls, s1, s2):
        """
        Computes edit distance where visually similar characters have a lower substitution penalty.
        """
        s1 = cls.clean_text(s1)
        s2 = cls.clean_text(s2)

        if s1 == s2:
            return 0.0
        if not s1:
            return float(len(s2))
        if not s2:
            return float(len(s1))

        m, n = len(s1), len(s2)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = float(i)
        for j in range(n + 1):
            dp[0][j] = float(j)

        for i in range(1, m + 1):
            c1 = s1[i - 1]
            for j in range(1, n + 1):
                c2 = s2[j - 1]
                sub_cost = cls._char_substitution_cost(c1, c2)
                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,          # Deletion
                    dp[i][j - 1] + 1.0,          # Insertion
                    dp[i - 1][j - 1] + sub_cost     # Substitution
                )

        return dp[m][n]

    @classmethod
    def find_target_button(cls, buttons, target_floor):
        """
        Finds the button matching target_floor from a list of detected button dictionaries.
        Returns the matching button dict or None if no confident match is found.
        """
        if target_floor is None or not buttons:
            return None

        target_clean = cls.clean_text(str(target_floor))
        if not target_clean:
            return None

        # 1. Exact match
        for btn in buttons:
            btn_txt = cls.clean_text(btn.get('text', ''))
            if btn_txt == target_clean:
                return btn

        # 2. Token synonyms and ligatures
        synonyms = [cls.clean_text(s) for s in cls.TOKEN_SYNONYMS.get(target_clean, [])]
        for btn in buttons:
            btn_txt = cls.clean_text(btn.get('text', ''))
            if btn_txt in synonyms:
                return btn

        # 3. Leading zero / multi-digit prefix resolution
        for btn in buttons:
            btn_txt = cls.clean_text(btn.get('text', ''))
            if len(btn_txt) > len(target_clean):
                if btn_txt.startswith('0') and btn_txt.endswith(target_clean):
                    return btn
                if len(target_clean) >= 2 and btn_txt.startswith('1') and btn_txt.endswith(target_clean):
                    return btn

        # 4. Visually-weighted edit distance
        candidates = []
        for btn in buttons:
            btn_txt = cls.clean_text(btn.get('text', ''))
            if not btn_txt:
                continue

            if abs(len(btn_txt) - len(target_clean)) > 1:
                continue

            dist = cls.visual_edit_distance(target_clean, btn_txt)
            candidates.append((dist, btn))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            best_dist, best_btn = candidates[0]

            # Strict threshold to avoid false positive digit swaps
            threshold = 0.25 if len(target_clean) == 1 else 0.45
            if best_dist <= threshold:
                return best_btn

        return None
