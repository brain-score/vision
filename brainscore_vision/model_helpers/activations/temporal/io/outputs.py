import logging

logger = logging.getLogger(__name__)

class Activation(dict):
    def set(self, input, layer, data):
        if input not in self:
            self[input] = {}
        if layer in self[input]:
            logger.warning(f"Overwriting activation for input '{input}' and layer '{layer}'.")
        self[input][layer] = self._consume(data)

    def get(self, input, layer):
        return self[input][layer]
    
    # convert the data to xarray
    def _consume(self, data):
        import xarray as xr
        return xr.DataArray(data)
    
    # concatenate along the input dimension
    def to_compact(self):
        raise NotImplementedError()