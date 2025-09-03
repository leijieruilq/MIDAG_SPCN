# Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative



## Environment Setup
Linux with CUDA 12.X
```sh
conda create -n tats python=3.11.11
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install pandas scikit-learn patool tqdm sktime matplotlib reformer_pytorch transformers
```

## Execute TaTS (Default: iTransformer+GPT2, Environment dataset)
```sh
chmod +x ./scripts/main_forecast.sh
./scripts/main_forecast.sh
```


## Reference
If you find this repository useful in your research, please consider citing the following paper:

```
@article{DBLP:journals/corr/abs-2502-08942,
  author       = {Zihao Li and
                  Xiao Lin and
                  Zhining Liu and
                  Jiaru Zou and
                  Ziwei Wu and
                  Lecheng Zheng and
                  Dongqi Fu and
                  Yada Zhu and
                  Hendrik F. Hamann and
                  Hanghang Tong and
                  Jingrui He},
  title        = {Language in the Flow of Time: Time-Series-Paired Texts Weaved into
                  a Unified Temporal Narrative},
  journal      = {CoRR},
  volume       = {abs/2502.08942},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2502.08942},
  doi          = {10.48550/ARXIV.2502.08942},
  eprinttype    = {arXiv},
  eprint       = {2502.08942},
  timestamp    = {Fri, 14 Mar 2025 08:16:58 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2502-08942.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```