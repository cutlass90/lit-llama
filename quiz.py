import sys
import time
import warnings
from pathlib import Path
from typing import Optional
from typing import List

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup


@torch.no_grad()
def generate(
    model: LLaMA,
    idx: torch.Tensor,
    answers: List[torch.Tensor],
    *,
    temperature: float = 1.0,
) -> list:
    """Takes a question and few answers variants as input and return probability for each answer.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        answers: list of Tensors with indices of the answer sequences
        temperature: Scales the predicted logits by 1 / temperature
    """
    res = []
    for answer in answers:
        prob_list = []
        for i in range(len(answer)):
            x = torch.cat([idx, answer[:i]]).view(1, -1)
            # forward
            logits = model(x)
            logits = logits[0, -1] / temperature

            probs = torch.nn.functional.softmax(logits, dim=-1)
            prob_list.append(probs[answer[i]])
        res.append((sum(prob_list)/len(prob_list)).item())

    return res


def main(
    prompt: str = "What is the biggest planet in owr solar system?",
    answers: str = "Mars|Jupiter|Sun|Neptun|Earth|They are all the same size|The biggest planet is Jupiter",
    *,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    quantize: Optional[str] = None,
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
    """
    assert checkpoint_path.is_file(), checkpoint_path
    assert tokenizer_path.is_file(), tokenizer_path

    fabric = L.Fabric(devices=1)
    dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with lazy_load(checkpoint_path) as checkpoint:
        name = llama_model_lookup(checkpoint)

        with EmptyInitOnDevice(
                device=fabric.device, dtype=dtype, quantization_mode=quantize
        ):
            model = LLaMA.from_name(name)

        model.load_state_dict(checkpoint)
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=fabric.device)
    answers_idx = [tokenizer.encode(a, bos=False, eos=False, device=fabric.device) for a in answers.split('|')]
    L.seed_everything(1234)


    probs = generate(model, encoded, answers_idx, temperature=temperature)
    probs = [p*1/sum(probs) for p in probs]
    answers = sorted(zip(answers.split('|'), probs), key=lambda x: x[1], reverse=True)
    print(answers)

    model.reset_cache()


    if fabric.device.type == "cuda":
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    warnings.filterwarnings(
        # Triggered in bitsandbytes/autograd/_functions.py:298
        "ignore", 
        message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
    )
    CLI(main)
