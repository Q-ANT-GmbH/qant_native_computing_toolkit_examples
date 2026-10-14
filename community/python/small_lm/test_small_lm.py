import generate
import numpy as np
import torch
import train
import utils


def test_train_and_generate(tmp_path):
    weights_path = tmp_path / "small_lm.pt"
    model, tokenizer, train_losses, _ = train.main(weights_path=weights_path)

    # Clearly better than random guessing (ln(65) ~ 4.17)
    np.testing.assert_array_less(train_losses[-1], 2.2)

    prompt = "ROMEO:"
    loaded_model, _ = generate.load_model(weights_path)
    text = generate.generate_text(loaded_model, tokenizer, prompt, 50, seed=1)
    assert text.startswith(prompt)
    assert len(text) == len(prompt) + 50
    assert set(text) <= set(tokenizer.chars)

    # bfloat16 inference on the NPU predicts (almost) the same characters as float32 on CPU
    data = torch.tensor(tokenizer.encode(utils.load_shakespeare()), dtype=torch.long)
    x, _ = utils.get_batch(data, model.block_size, 4, np.random.default_rng(seed=0))
    with torch.no_grad():
        model.train()  # PyTorch float32 reference
        logits_cpu, _ = model(x)
        model.eval()  # Q.ANT toolkit
        logits_npu, _ = model(x)
    agreement = (logits_cpu.argmax(-1) == logits_npu.argmax(-1)).float().mean().item()
    np.testing.assert_array_less(0.9, agreement)
