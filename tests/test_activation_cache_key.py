from brainscore_vision.model_helpers.activations.core import ActivationsExtractorHelper


def _helper(identifier="model", backbone_id=None, channel=None):
    return ActivationsExtractorHelper(
        get_activations=lambda _inputs, _layers: None,
        preprocessing=lambda inputs: inputs,
        identifier=identifier,
        backbone_id=backbone_id,
        channel=channel,
    )


def test_single_channel_cache_uses_legacy_stored_key(monkeypatch):
    helper = _helper(identifier="single-channel", backbone_id="shared-backbone")
    calls = []

    def old_stored(**kwargs):
        calls.append(("old", kwargs))
        return "old-cache"

    def channel_stored(**kwargs):
        calls.append(("channel", kwargs))
        return "channel-cache"

    monkeypatch.setattr(helper, "_from_paths_stored", old_stored)
    monkeypatch.setattr(helper, "_from_paths_stored_by_channel", channel_stored)

    result = helper.from_paths(
        stimuli_paths=["stimulus-a.png"],
        layers=["layer"],
        stimuli_identifier="stimuli",
        require_variance=False,
    )

    assert result == "old-cache"
    assert calls == [
        (
            "old",
            {
                "identifier": "shared-backbone",
                "stimuli_identifier": "stimuli",
                "require_variance": False,
                "layers": ["layer"],
                "stimuli_paths": ["stimulus-a.png"],
            },
        )
    ]


def test_shared_backbone_channels_use_separate_cache_keys(monkeypatch):
    calls = []

    def fail_old_stored(**_kwargs):
        raise AssertionError("shared-backbone channel calls must use channel-keyed storage")

    def make_channel_stored(return_value):
        def channel_stored(**kwargs):
            calls.append(kwargs)
            return return_value

        return channel_stored

    vision = _helper(
        identifier="vision-wrapper",
        backbone_id="shared-backbone",
        channel="vision",
    )
    text = _helper(
        identifier="text-wrapper",
        backbone_id="shared-backbone",
        channel="text",
    )
    monkeypatch.setattr(vision, "_from_paths_stored", fail_old_stored)
    monkeypatch.setattr(text, "_from_paths_stored", fail_old_stored)
    monkeypatch.setattr(
        vision, "_from_paths_stored_by_channel", make_channel_stored("vision-cache")
    )
    monkeypatch.setattr(
        text, "_from_paths_stored_by_channel", make_channel_stored("text-cache")
    )

    common_kwargs = dict(
        stimuli_paths=["same-stimulus"],
        layers=["shared-layer"],
        stimuli_identifier="same-stimuli",
        require_variance=False,
    )

    assert vision.from_paths(**common_kwargs) == "vision-cache"
    assert text.from_paths(**common_kwargs) == "text-cache"
    assert calls == [
        {
            "channel": "vision",
            "identifier": "shared-backbone",
            "stimuli_identifier": "same-stimuli",
            "require_variance": False,
            "layers": ["shared-layer"],
            "stimuli_paths": ["same-stimulus"],
        },
        {
            "channel": "text",
            "identifier": "shared-backbone",
            "stimuli_identifier": "same-stimuli",
            "require_variance": False,
            "layers": ["shared-layer"],
            "stimuli_paths": ["same-stimulus"],
        },
    ]
