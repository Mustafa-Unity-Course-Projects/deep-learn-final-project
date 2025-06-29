import torch

from train import model


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


def predict(model, observation: list[float]):
    print(observation, ": ", model(torch.tensor([observation], dtype=torch.float32, device=device)).max(1).indices.view(1, 1))


def main():
    predict(model, [0.0, 0.4, 10.0])
    predict(model, [0.0, 0.6, 10.0])
    predict(model, [0.0, 0.4, 35.0])
    predict(model, [0.0, 0.6, 35.0])
    predict(model, [1.0, 0.4, 10.0])
    predict(model, [1.0, 0.6, 10.0])
    predict(model, [1.0, 0.4, 35.0])
    predict(model, [1.0, 0.6, 35.0])
    torch.save(model.state_dict(), "model_weights")


if __name__ == "__main__":
    main()


