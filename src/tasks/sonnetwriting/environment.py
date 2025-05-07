import random
from typing import Tuple, Set, Dict, Any

from src.typedefs import State

from .state import StateSonnetWriting
from ...typedefs import Environment, MAX_SEED

import re
import joblib
import pyphen
import syllables
import pronouncing

class EnvironmentSonnetWriting(Environment):

    @staticmethod
    def step(state: StateSonnetWriting, action: str) -> StateSonnetWriting:
        """
        Takes a step in the environment based on the given action.
        """
        random.seed(state.randomness)
        randomness = random.randint(0, MAX_SEED)

        state = StateSonnetWriting(
            puzzle=state.puzzle,
            current_state=action,
            steps=state.steps + [action],
            target=state.target,
            randomness=randomness
        )

    @staticmethod
    def is_valid(state: StateSonnetWriting, action: str) -> bool:
        """
        Checks if the action taken is valid.
        """
        pass

    @staticmethod
    def is_final(state: StateSonnetWriting) -> bool:
        """
        Checks if the current state is a final state.
        """
        pass

    @staticmethod
    def evaluate(state: StateSonnetWriting) -> Tuple[bool | float]:
        """
        Evaluates the current state.
        """
        current_state = state.current_state
        rhyme_scheme = state.target.split(", ")[0]

        try:
            errors = sonnet_errors(current_state, rhyme_scheme)
            if not errors:
                return True, 1.0
            return False, 0.0
        except Exception as e:
            return False, 0.0



#---Helper functions---#
# These function is from meta-prompting GH: https://github.com/suzgunmirac/meta-prompting/blob/main/utils/sonnet_eval.py
ALLOWED_SYLLABLES = {
    10,
    11,
}  # about 3-4% of legit lines have 11 syllables, so we allow it, > 99% have 10 or 11
NUM_REQUIRED_WORDS = 3

memory = joblib.Memory(
    ".cache", verbose=0
)  # use cache to speed up repeated rhyme/syllable calls


def sonnet_errors(poem: str, target: str, verbose=False) -> Dict[str, Any]:
    """
    Checks for sonnet errors with respect to target rhyme scheme (and optional required words)

    args:
        poem: the poem to check
        target: the rhyme scheme, e.g. "ABBA ABBA CDC DCD"
                optionally target can have a list of required words, like
                "ABBA ABBA CDC DCD, love train snail" each of these must be in the poem
        verbose: if True, print out more details
    """
    if ", " in target:
        scheme, rest = target.split(", ")
        required_words = rest.split()
    else:
        scheme = target
        required_words = []

    errors = scheme_errors(poem, scheme, verbose=verbose)
    assert isinstance(errors, dict)
    missing_words = [w for w in required_words if w.lower() not in poem.lower()]
    if any(missing_words):
        errors["missing words"] = missing_words

    syllable_errors = []
    for line in split_poem(poem):
        variations = syllable_variations(line)
        if not (variations & ALLOWED_SYLLABLES):
            syllable_errors.append((line, sorted(variations)))
    if syllable_errors:
        errors["syllable errors"] = syllable_errors

    return errors

def scheme_errors(poem: str, scheme: str, verbose=False):
    """Find errors with respect to a given rhyming scheme"""
    lines = split_poem(poem)
    scheme = scheme.replace(" ", "")

    if len(lines) != len(scheme):
        return {
            "line count": f"Poem has {len(lines)} != {len(scheme)} lines in pattern {scheme}"
        }

    last_words = [clean_word(l.replace("-", " ").split()[-1]) for l in lines]

    dictionary = pronouncing.cmudict.dict()  # we ignore words not in dictionary

    groups = []
    for chars in sorted(set(scheme)):
        groups.append(
            [w for w, p in zip(last_words, scheme) if p == chars and w in dictionary]
        )

    slant_sets = {w: set(slant_rhyming_parts(w)) for g in groups for w in g}

    scores = {}

    if verbose:
        print(groups)

    for g in groups:
        internal_words = set(g)
        external_words = {w for h in groups if h is not g for w in h}
        if len(internal_words) == 1:
            continue  # don't check rhymes if only word word in the group is in dictionary
        for w in g:
            rhymes = get_rhymes(w)
            scores[w] = []
            for comparisons in [internal_words, external_words]:
                m = dict(rhymes=[], slant_rhymes=[])
                scores[w].append(m)
                for v in comparisons:
                    if v == w:
                        continue
                    if v in rhymes:
                        m["rhymes"].append(v)
                    elif slant_sets[v] & slant_sets[w]:
                        m["slant_rhymes"].append(v)

    error_reasons = {}
    suspicious_reasons = {}

    for w in scores:
        internal, external = scores[w]

        if internal["rhymes"] or internal["slant_rhymes"]:
            pass  # ok if it rhymes (perfect or slant) with at least one other word in the group
        elif len(external["rhymes"]) >= 2:
            error_reasons[w] = "no internal rhymes, 2+ external perfect rhymes"
        elif external["rhymes"]:
            if len(external["slant_rhymes"]) >= 2:
                error_reasons[
                    w
                ] = "no internal rhymes, 1 external perfect rhyme, 2+ external slant rhymes"
            else:
                suspicious_reasons[
                    w
                ] = "no internal rhymes/slant rhymes, 1 external perfect rhymes"
        elif len(external["slant_rhymes"]) >= 3:
            error_reasons[
                w
            ] = "no internal rhymes/slant rhymes, 3+ external slant rhymes"
        if verbose:
            print(w, "internal:", internal, "external:", external)

    if len(error_reasons) + len(suspicious_reasons) >= 3:
        error_reasons.update(suspicious_reasons)

    return {
        w: {
            "reason": error_reasons[w],
            "internal": scores[w][0],
            "external": scores[w][1],
        }
        for w in error_reasons
    }

def syllable_variations(text, verbose=False) -> Set[int]:
    """
    Given a text, return the set of possible numbers of syllables. It's a set because some words like "caramel" can
    be pronounced with different numbers of syllables.
    """
    ans = {0}
    for word in re.split("[ -]+", text):
        word = clean_word(word)
        if not word:
            continue
        options = word_syllables(word)
        options = range(
            min(options), max(options) + 1
        )  # make it a range (so {2, 4} moves to [2, 3, 4])
        ans = {x + y for x in ans for y in options}
    return ans


@memory.cache
def word_syllables(word: str) -> Set[int]:
    assert word == clean_word(
        word
    ), "Word should be cleaned before hitting word_syllables cache"
    return SyllableCounters.count_word(word)


class SyllableCounters:
    """
    Simple class to count syllables in text.
    """

    _cmu_dict = None
    _pyphen_counter = None

    @staticmethod
    def cmu_dict():
        if not SyllableCounters._cmu_dict:
            SyllableCounters._cmu_dict = pronouncing.cmudict.dict()
        return SyllableCounters._cmu_dict

    def cmu(word):
        return {
            pronouncing.syllable_count(pro) for pro in pronouncing.phones_for_word(word)
        }

    @staticmethod
    def pyphen_counter():
        if not SyllableCounters._pyphen_counter:
            SyllableCounters._pyphen_counter = pyphen.Pyphen(lang="en")
        return SyllableCounters._pyphen_counter

    @staticmethod
    def count_word(word) -> Set[int]:
        if not word:
            return {0}

        cmu = SyllableCounters.cmu(word)

        pyph = SyllableCounters.pyphen_counter().inserted(word).count("-") + 1

        syll = syllables.estimate(word)

        ans = cmu | {pyph, syll}

        if 0 in ans and len(ans) > 1:
            ans.remove(0)

        return ans
    
def clean_word(text: str):
    return text.lower().strip(",.!?;: \"'[]()/")


def clean_line(line: str):
    """
    Clean a line from a poem.
    Check if line ends with (A) or (B) ... and remove it
    """
    line = re.sub(r"\s*\([A-Za-z]\)\s*$", "", line)
    return line.strip()


def split_poem(poem: str, min_line_len=3):
    ans = [clean_line(l) for l in poem.splitlines()]
    return [l for l in ans if len(l) > min_line_len]


@memory.cache
def slant_rhyming_parts(word: str):
    consonants = set("BCDFGHJKLMNPQRSTVWXYZ")
    ans = [
        "".join(
            ("R" if "R" in p else (p if p in consonants else "?"))
            for p in pronouncing.rhyming_part(ph).split()
        )
        for ph in pronouncing.phones_for_word(word)
    ]
    ans = [a for a in ans if not all(i == "?" for i in a)]
    ans = [a.replace("?", "") + ("?" if a.endswith("?") else "") for a in ans]
    return set(ans)


@memory.cache
def get_rhymes(w):
    return set(pronouncing.rhymes(w))