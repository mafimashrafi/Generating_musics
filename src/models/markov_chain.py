import json
import random
from collections import defaultdict, Counter
from pathlib import Path


class MarkovChainMusicModel:
    def __init__(self, order=2):
        self.order = order
        self.chain = defaultdict(Counter)
        self.start_states = []

    def train(self, token_files):
        for file_path in token_files:
            with open(file_path, "r") as f:
                data = json.load(f)
                
            if isinstance(data, dict):
                tokens = data.get("tokens", [])
            else:
                tokens = data

            if len(tokens) <= self.order:
                continue

            self.start_states.append(tuple(tokens[:self.order]))

            for i in range(len(tokens) - self.order):
                state = tuple(tokens[i:i + self.order])
                next_token = tokens[i + self.order]
                self.chain[state][next_token] += 1

    def next_token(self, state):
        
        if state not in self.chain:
            return None

        counter = self.chain[state]

        tokens = list(counter.keys())
        weights = list(counter.values())

        return random.choices(tokens, weights=weights, k=1)[0]

    def generate(self, length=512):
        if not self.start_states:
            raise ValueError("Model has not been trained.")

        state = random.choice(self.start_states)

        generated = list(state)

        for _ in range(length - self.order):

            nxt = self.next_token(state)

            if nxt is None:
                break

            generated.append(nxt)

            state = tuple(generated[-self.order:])

        return generated

    def save(self, path):
        serializable_chain = {
            str(k): dict(v)
            for k, v in self.chain.items()
        }

        payload = {
            "order": self.order,
            "chain": serializable_chain,
            "start_states": [list(s) for s in self.start_states]
        }

        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path):

        with open(path, "r") as f:
            payload = json.load(f)

        model = cls(order=payload["order"])

        for k, v in payload["chain"].items():
            state = tuple(map(int, k.strip("()").split(", ")))
            model.chain[state] = Counter({int(a): b for a, b in v.items()})

        model.start_states = [tuple(x) for x in payload["start_states"]]

        return model