# train_pgdp_ocr

Python tools for training DocTR OCR models on PGDP Data

## Installation

Install the 'uv' tooling to manage project dependencies:

https://docs.astral.sh/uv/getting-started/installation/

I used: `pipx install uv` (you will need pipx to do this), upgrade with `pipx upgrade uv`

The Wget way is probably better.

Then run `uv venv` to create a venv.

Deactivate any current venv (`deactivate`), then activate the venv `source .venv/bin/activate`

Clone pd-book-tools from github in the parent directory of this repo.
gh repo clone ConcaveTrillion/pd-book-tools

Install dependencies.
`uv sync`

Check pre-commit
`pre-commit`

To get the model files, you need to use git lfs.
https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage

## Usage

### Labeler

`voila data-labeler.ipynb` will run the labeling notebook web server

Connect to http://localhost:8866/ to use the labeler

### Model Trainer

`jupyter model-trainer.ipynb` (or run in VS Code or your preferred notebook running tool)

To use the trainer, you have to pull down the doctr git repo as the scripts are not in the pypi doctr toolset.

Install it in the PARENT directory of this repo (`../doctr`)
e.g.
```
cd ..
gh repo clone mindee/doctr
```
or
```
cd ..
git clone https://github.com/mindee/doctr.git doctr
```

You need to modify one file in this repo to add logic to allow use of custom vocabulary:
In file: doctr/references/recognition/train_pytorch.py

Where you find
```
    vocab = VOCABS[args.vocab]
```

Change this to 
```
    if args.vocab.startswith("CUSTOM:"):
        # Custom vocab
        custom_vocab = args.vocab.split(":", 1)[1]
        if not custom_vocab:
            raise ValueError("Custom vocab cannot be empty")
        vocab = "".join(sorted(set([char for char in custom_vocab])))
    else:
        vocab = VOCABS[args.vocab]
```

Once you've done this, you can run the model training notebook.

## License

See LICENSE file.



