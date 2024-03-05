A submission means the whole process of converting an ML model to a BrainModel, so that the model can be directly tested on brainscore benchmarks.

In wrapper_tutorial.py, you saw the process of wrapping an ML model to become a "base_model", which implements the inference procedure: *stimulus_set -> model_assembly*. Like the following:

```python
base_model = PytorchWrapper(
    identifier="my_model", 
    model=model, 
    preprocessing=transform_input, 
    fps=25, 
    layer_activation_format=layer_activation_format
)
...
model_assembly = base_model(video_paths, layers)
```

To further convert the base_model into a BrainModel is straightforward. You just have to pass this base_model to the ModelCommitment:
```python
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment

brain_model = ModelCommitment(identifier=identifier, activations_model=base_model, layers=layers, region_layer_map=region_layer_map)

# layers: the considered layers in a model
# region_layer_map: a dict(brain_region -> model_layer)
```

To register this brain_model to the brainscore model pool, you simply write the following code in the <ins>brainscore_vision/models/temporal_models/__init__.py</ins>
```python
from brainscore_vision import model_registry
from your_submodule import get_brain_model

model_registry['my_model'] = lambda: get_brain_model('my_model')
```

Here we used a lambda expression because we are supposed to pass a loader.

By doing the above, the 'my_model' will be registered and lazily loaded if needed. Caution: you have to write *model_registry['my_model']* explicitly. The following code will not work for brainscore:
```python
for name in names:
    model_registry[name] = lambda: get_brain_model(name)
    # will not work! Please write the name explicitly
```

For more details, see my submissions of torchvision and mmaction2 in <ins>brainscore_vision/models/temporal_models</ins>.

Finally, we can use our brain model just like a brain:
```python
from brainscore_vision import load_model

brain_model = load_model("my_model")
model.start_recording('IT', time_bins=[(100, 150), (150, 200), (200, 250)])
model_assembly = model.look_at(stimulus_set)  # NeuroidAssembly with dims: [presentation, time_bin, neuroid]
```
See a concrete example of using the brain model in <ins>brainscore_vision/models/temporal_models/test.py</ins>