from brainscore.utils import recursive_dict_merge


class TestRecursiveDictMerge:
    def test_no_overlap(self):
        dict1 = {'a': 1}
        dict2 = {'b': 2}
        merged = recursive_dict_merge(dict1, dict2)
        assert {'a': 1, 'b': 2} == merged

    def test_overlap1(self):
        dict1 = {"foo": {"bar": 23, "blub": 42}, "flub": 17}
        dict2 = {"foo": {"bar": 100}, "flub": {"flub2": 10}, "more": {"stuff": 111}}
        merged = recursive_dict_merge(dict1, dict2)
        assert {'foo': {'bar': 100, 'blub': 42}, 'flub': {'flub2': 10}, 'more': {'stuff': 111}} == merged

    def test_overlap2(self):
        dict1 = {"foo": {"bar": 100}, "flub": {"flub2": 10}, "more": {"stuff": 111}}
        dict2 = {"foo": {"bar": 23, "blub": 42}, "flub": 17}
        merged = recursive_dict_merge(dict1, dict2)
        assert {'foo': {'bar': 23, 'blub': 42}, 'flub': 17, "more": {"stuff": 111}} == merged
