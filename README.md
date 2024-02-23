# Exa Interview
Below is the work I did with Exa to train an embedding model to learn to distinguish similar-but-irrelevant queries.

See writeup.md for theoretical background and training details

## Using the model

First, install the requirements `pip install -r requirements.txt`.

The pretrained model is saved in huggingface at `mfine/embedding-model`. To use, load it using my TrainedEncoder class

```python
from src.encoder_model import TrainedEncoder

model = TrainedEncoder("mfine/embedding-model")

query = "youngest company based in the same region as Apple"

sentences = [
    "Founded in 2024 in Los Altos, Simile systems produces gpus for the blockchain",
    "In 2020, Synechdoche was founded by the youngest ex-Apple engineers, based out of the New York",
    "Analogy.ai was not founded in the same region as Apple and is not the youngest company"
]

q_emb = model.embed(query)
s_emb = model.embed(sentences)

q_emb @ s_emb.T
```

More details and evaluations can be found in src/eval_model.ipynb

## Training the model

To run the full data generation and training pipeline, first store your OpenAI key
`$ export OPENAI_KEY={your_key_here}`.

Then run `scripts/e2e_train.sh`, which calls `scripts/generate_data.sh` and `scripts/train.sh`, where you can set hyperparameters and output folders. You may have to log in to wandb to get metrics.
