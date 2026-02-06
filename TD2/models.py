from typing import Callable, List
import equinox as eqx
from jaxtyping import Float, Array
import jax

class MLP(eqx.Module):
    
    layers: list[eqx.nn.Linear]
    activation: Callable
    
    def __init__(self, hidden_layers: List[int], key: jax.Array, activation: Callable=jax.nn.tanh) -> None:
        if len(hidden_layers) < 2:
            raise Exception("MLP.hidden_layers should have at least 2 items (in and out feature dims)")
        
        keys = jax.random.split(key, len(hidden_layers))
        self.activation = activation
        self.layers = [
            eqx.nn.Linear(in_dim, out_dim, key=key) for in_dim, out_dim, key in zip(
                hidden_layers[:-1], 
                hidden_layers[1:], 
                keys
        )]
        
    def __call__(self, x: Float[Array, "in_dim"]) -> Float[Array, "out_dim"]:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x
    

class HardIC(eqx.Module):
    mlp: MLP
    ic: Callable[..., Array]
        
    def __call__(self, tx: Float[Array, "2"]) -> Float[Array, "1"]:
        t, x = tx
        o = t * self.mlp(tx) + (1 - t) * self.ic(x)
        return o
