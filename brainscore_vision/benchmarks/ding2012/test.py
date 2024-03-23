from brainscore_vision import load_benchmark


def test_ding2012():
    ding2012 = load_benchmark("ding2012")
    print(ding2012.ceiling)

test_ding2012()