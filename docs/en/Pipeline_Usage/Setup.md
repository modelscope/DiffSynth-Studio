# Installing Dependencies

Install from source (recommended):

```
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

Install from PyPI (there may be delays in version updates; for latest features, install from source):

```
pip install diffsynth
```

If you encounter issues during installation, they may be caused by upstream dependency packages. Please refer to the documentation for these packages:

* [torch](https://pytorch.org/get-started/locally/)
* [sentencepiece](https://github.com/google/sentencepiece)
* [cmake](https://cmake.org)