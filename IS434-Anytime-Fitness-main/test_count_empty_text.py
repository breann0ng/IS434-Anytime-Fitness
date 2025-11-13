"""Small local test for the "ratings-without-text" counting logic.
This script simulates minimal review elements and checks the logic used in the notebook:
- use find_elements for the review text class
- coerce missing/None to '' and strip whitespace
- count reviews where text == '' as "ratings only"

Run: python "c:\\wamp64\\www\\IS434-Anytime-Fitness\\test_count_empty_text.py"
"""

class FakeSubElem:
    def __init__(self, text):
        self.text = text

class FakeReview:
    def __init__(self, text_elems_texts):
        # text_elems_texts: list of strings or None to simulate subelements
        self._texts = text_elems_texts

    def find_elements(self, *args):
        # This mimics Selenium's find_elements signature; we check if the class name 'wiI7pd' appears
        # in the args and return subelements accordingly.
        for a in args:
            if isinstance(a, str) and 'wiI7pd' in a:
                return [FakeSubElem(t) for t in self._texts]
        # also handle (By.CLASS_NAME, 'wiI7pd') where args[1] == 'wiI7pd'
        if len(args) >= 2 and args[1] == 'wiI7pd':
            return [FakeSubElem(t) for t in self._texts]
        return []


def extract_review_text(review):
    # Mirrors the notebook logic: use find_elements and normalize
    text_elems = review.find_elements(None, 'wiI7pd')
    if text_elems:
        review_text = (text_elems[0].text or '') .strip()
    else:
        review_text = ''
    return review_text


def run_tests():
    cases = [
        (FakeReview(['Great place!']), False, 'Has normal text'),
        (FakeReview(['   ']), True, 'Whitespace-only text -> should count as empty'),
        (FakeReview([]), True, 'No text element -> should count as empty'),
        (FakeReview([None]), True, 'None text -> should count as empty'),
        (FakeReview(['Good', 'Extra']), False, 'Multiple subelements; first contains text'),
    ]

    empty_count = 0
    for review, expect_empty, note in cases:
        text = extract_review_text(review)
        is_empty = (text == '')
        print(f"Case: {note}\n  Extracted text: {repr(text)}\n  Recognized empty: {is_empty} (expected: {expect_empty})\n")
        if is_empty:
            empty_count += 1

    print(f"SUMMARY: Detected {empty_count} empty-text reviews (expected 3)")
    assert empty_count == 3, f"Test failed: expected 3 empty-text reviews, got {empty_count}"
    print("All tests passed.")


if __name__ == '__main__':
    run_tests()
