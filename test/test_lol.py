from common_utils import IN_CIRCLE_CI

def test_lol():
    if IN_CIRCLE_CI:
        print("WOOHOO")
    else:
        raise ValueError("Nope")
