## Cognate Transformer for Automatic Phonological Reconstruction

Usage:
> `python run.py <method>`
>  Where `<method>` can be `reflex`, `proto-cog` or `proto-nmt`

Results are generated in `results/`

Pip dependencies are listed in `requirements.txt`

Additionally, the following line is to be added to `trainer_seq2seq.py` file from `transformers` library for NMT to work correctly.

> if "langs" in inputs: gen_kwargs["langs"] = inputs.get("langs", None)

The line number in the file would be at 189 (in `transformers v4.24.0`) and the file location would be like `.../anaconda3/lib/site-packages/transformers` or depends on the environment.