"""The Constitional Controller's doubt density, learning how much to trust an agent's conditional capabilities."""

#
# Copyright (c) Simon Kohaut, Felix Divo, and contributors
#
# This file is part of CoCO and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with CoCO.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Standard Library
from collections.abc import Callable
from copy import deepcopy
from pickle import dump, load

# Third Party
import torch
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from numpy import array, mean
from numpy.typing import NDArray
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from promis.geo import Collection

class DoubtDensity:
    def __init__(
        self,
        doubt_space: dict[str, dict],
        number_of_states: int,
        number_of_hidden_features: int,
        number_of_layers: int,
    ):
        # Store doubt feature configuration
        self.doubt_space = doubt_space

        # Conditional Flow parameters
        self.number_of_states = number_of_states
        self.number_of_hidden_features = number_of_hidden_features
        self.number_of_layers = number_of_layers
        self.number_of_context_features = sum(
            [
                1
                if doubt_space[doubt_feature]["type"] == "continuous"
                else doubt_space[doubt_feature]["number_of_classes"]
                for doubt_feature in doubt_space.keys()
            ]
        )

        # Need to ensure even number of features for nflows library
        if self.number_of_context_features > 2 and self.number_of_context_features % 2 == 1:
            self.number_of_context_features += 1

        # Flow setup
        transforms = []
        for _ in range(self.number_of_layers):
            transforms.append(ReversePermutation(features=self.number_of_states))
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=self.number_of_states,
                    hidden_features=self.number_of_hidden_features,
                    context_features=self.number_of_context_features,
                )
            )

        self.flow = Flow(CompositeTransform(transforms), StandardNormal(shape=[self.number_of_states]))

    @staticmethod
    def load(path) -> "Collection":
        with open(path, "rb") as file:
            return load(file)

    def save(self, path: str):
        with open(path, "wb") as file:
            dump(self, file)

    def doubt_space_to_tensor(self, doubt_space: dict[str, dict]):
        # One hot encode categorical features
        # TODO: Apply chosen normalization to continuous features
        encoded_features = []
        for feature in doubt_space.keys():
            if doubt_space[feature]["type"] == "categorical":
                encoder = OneHotEncoder(sparse_output=False, dtype=int)
                encoder.fit(array(list(range(doubt_space[feature]["number_of_classes"]))).reshape(-1, 1))
                encoded_features.append(
                    torch.from_numpy(encoder.transform(doubt_space[feature]["values"][:].reshape(-1, 1)))
                )
            elif doubt_space[feature]["type"] == "continuous":
                encoded_features.append(doubt_space[feature]["values"][:].reshape(-1, 1))

        # Concatenate encoded features
        context = torch.hstack(encoded_features)

        # Ensure even number of columns for nflows library
        if context.shape[1] % 2 == 1:
            context = torch.hstack([context, torch.zeros((context.shape[0]), 1)])

        return context

    def sample(self, size: int, doubt_space: dict[str, dict]):
        return self.flow.sample(size, context=self.doubt_space_to_tensor(doubt_space))

    def log_prob(self, states: NDArray, doubt_space: dict[str, dict]):
        return self.flow.log_prob(states, context=self.doubt_space_to_tensor(doubt_space))

    def prob(self, states: NDArray, doubt_space: dict[str, dict]):
        return self.log_prob(states, doubt_space).exp()

    def fit(
        self, samples: NDArray, doubt_space: dict[str, dict], number_of_epochs: int, batch_size: int
    ) -> NDArray:
        # Setup optimizer
        optimizer = torch.optim.Adam(self.flow.parameters())
        dataloader = torch.utils.data.DataLoader(
            torch.hstack([samples, self.doubt_space_to_tensor(doubt_space)]),
            batch_size=batch_size,
            shuffle=True,
        )

        # Train on given samples and doubt features
        losses = []
        for epoch in tqdm(range(number_of_epochs), desc="Learning Doubt Density", unit="epoch"):
            for batch in dataloader:
                optimizer.zero_grad()
                loss = -self.flow.log_prob(
                    batch[:, : self.number_of_states], context=batch[:, self.number_of_states :]
                ).mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().numpy())

        return losses

