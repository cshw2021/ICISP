# A Lightweight Model for Perceptual Image Compression via Implicit Priors

[Hao Wei](https://scholar.google.com.hk/citations?user=hhNFVW0AAAAJ&hl=zh-CN), Yanhui Zhou, Yiwen Jia, Chenyang Ge, [Saeed Anwar](https://scholar.google.com.hk/citations?user=vPJIHywAAAAJ&hl=zh-CN), [Ajmal Mian](https://scholar.google.com.hk/citations?user=X589yaIAAAAJ&hl=zh-CN).

#### 🔥🔥🔥 News

- **2025-10-09:** This repo is released.

---
> **Abstract:** Perceptual image compression has shown strong potential for producing visually appealing results at low bitrates, surpassing classical standards and pixel-wise distortion-oriented neural methods.
However, existing methods typically improve compression performance by incorporating explicit semantic priors, such as segmentation maps and textual features, into the encoder or decoder, which increases model complexity by adding parameters and floating-point operations. This limits the model's practicality, as image compression often occurs on resource-limited mobile devices.
To alleviate this problem, we propose a lightweight perceptual **I**mage **C**ompression method using **I**mplicit **S**emantic **P**riors (**ICISP**). 
We first develop an enhanced visual state space block that exploits local and global spatial dependencies to reduce redundancy. 
Since different frequency information contributes unequally to compression, we develop a frequency decomposition modulation block to adaptively preserve or reduce the low-frequency and high-frequency information.
We establish the above blocks as the main modules of the encoder-decoder, and to further improve the perceptual quality of the reconstructed images, 
we develop a semantic-informed discriminator that uses implicit semantic priors from a pretrained DINOv2 encoder. 
Experiments on popular benchmarks show that our method achieves competitive compression performance and has significantly fewer network parameters and floating point operations than the existing state-of-the-art.
We will release the code and trained models.

## ⚒️ TODO

* [ ] Release code
* [ ] Upload the latest paper to Arxiv

## 🔗 Contents

- [x] [Datasets](#Datasets)
- [x] [Models](#Models)
- [ ] Testing
- [ ] Training
- [ ] [Results](#Results)
- [x] [Citation](#Citation)
- [ ] Acknowledgements

## <a name="datasets"></a>📊 Datasets
We train the ICISP on the [LSDIR](https://ofsoundof.github.io/lsdir-data/) dataset and evaluate it on the [Kodak](https://r0k.us/graphics/kodak/) and [CLIC_2020](https://clic2025.compression.cc/) datasets.

## <a name="models"></a>:dna:Models
| Rate Lambda | Link |
|--------|------|
|1   |  [model_1.pth](https://drive.google.com/drive/folders/1VIr_8j4gy69C2M4-gmtxGDaRGnGdnp0P)  |
|1.5 |  [model_1.5.pth](https://drive.google.com/drive/folders/1VIr_8j4gy69C2M4-gmtxGDaRGnGdnp0P)  |
|2.5 |  [model_2.5.pth](https://drive.google.com/drive/folders/1VIr_8j4gy69C2M4-gmtxGDaRGnGdnp0P)  |
|5 |  [model_5.pth](https://drive.google.com/drive/folders/1VIr_8j4gy69C2M4-gmtxGDaRGnGdnp0P)  |

## <a name='results'></a> 🔎 Results

## <a name="citation"></a>:smiley: Citation

If you find the code helpful in your research or work, please cite our work.

```
@article{wei2025lightweight,
  title={A Lightweight Model for Perceptual Image Compression via Implicit Priors},
  author={Wei, Hao and Zhou, Yanhui and Jia, Yiwen and Ge, Chenyang and Anwar, Saeed and Mian, Ajmal},
  journal={arXiv preprint arXiv:2502.13988},
  year={2025}
}
```

## <a name="acknowledgements"></a>💡 Acknowledgements

[TBD]
