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
    max_new_tokens: int,
    *,
    max_seq_length: Optional[int] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> list:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        eos_id: If specified, stop generating any more token once the <eos> token is triggered
    """
    res = []
    for answer in answers:
        # max_seq_length = idx.size(0) + len(answer)
        # device, dtype = idx.device, idx.dtype

        # generate max_new_tokens tokens
        prob_list = []
        for i in range(len(answer)):
            # input_pos = torch.arange(0, len(idx)+i, device=device)
            x = torch.cat([idx, answer[:i]]).view(1, -1)

            # forward
            logits = model(x)
            logits = logits[0, -1] / temperature

            probs = torch.nn.functional.softmax(logits, dim=-1)
            prob_list.append(probs[answer[i]])
        res.append((sum(prob_list)/len(prob_list)).item())

    return res


def main(
    prompt: str = "Hello, my name is",
    answers: str = "variant1|variant2|variant3",
    *,
    max_new_tokens: int = 50,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    quantize: Optional[str] = None,
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.

        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
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


    probs = generate(model, encoded, answers_idx, max_new_tokens, temperature=temperature, top_k=top_k)
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
