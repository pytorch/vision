import os

IN_CIRCLE_CI = os.getenv("CIRCLECI", False) == 'true'

def test_lol():
    if IN_CIRCLE_CI:
        print("WOOHOO")
    else:
        raise ValueError("Nope")
