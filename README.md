# HDR+ PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/martin-marek/hdr-plus-pytorch/blob/main/demo.ipynb)

This is a simplified PyTorch implementation of HDR+, the backbone of computational photography in Google Pixel phones, described in [Burst photography for high dynamic range and low-light imaging on mobile cameras](http://static.googleusercontent.com/media/www.hdrplusdata.org/en//hdrplus.pdf). Using a free Colab GPU, aligning 20MP RAW images takes ~200 ms/frame.

If you would like to use HDR+ in practice (rather than research), please check out my open-source Mac app [Burst Photo](https://burst.photo). It has a GUI, supports robust merge, and uses Adobe DNG SDK (instead of LibRaw), significantly improving image quality.

# Example

I took a burst of 35 images at ISO 12,800 on Sony RX100-V and boosted it by +2EV. Here's a comparison of a [single image](illustrations/burst_sample.jpg) from the burst vs. a [merge of all the images](illustrations/merged_image.jpg).

![alt text](illustrations/before_and_after.jpg)

# Usage

Here's a minimal example to align and merge a burst of raw images. For more, see the [Colab Notebook](https://colab.research.google.com/github/martin-marek/hdr-plus-pytorch/blob/main/demo.ipynb).

```python
import torch, align
images = torch.zeros([5, 1, 1000, 1000])
merged_image = align.align_and_merge(images)
```

# Implementation details

The implementation heavily relies on [first class dimensions](https://github.com/facebookresearch/torchdim), which allows for vectorized code that resembles the use of explicit loops. [Previous versions](https://github.com/martin-marek/hdr-plus-pytorch/blob/322c6039393074cefd9c5082006b509d5121aad1/align.py) of this repo used standard NumPy-style broadcasting but that was slower, harder to read, and required more loc.

# Features
- [x] jpeg support
- [x] RAW support
- [x] simple merge
- [ ] robust merge
- [x] tile comparison in pixel space
- [ ] tile comparison in Fourier space
- [x] CUDA support
- [x] CPU support (very slow)
- [ ] color post-processing
- [ ] automatic selection of the reference image

# Citation

```bibtex
@article{hasinoff2016burst,
  title={Burst photography for high dynamic range and low-light imaging on mobile cameras},
  author={Hasinoff, Samuel W and Sharlet, Dillon and Geiss, Ryan and Adams, Andrew and Barron, Jonathan T and Kainz, Florian and Chen, Jiawen and Levoy, Marc},
  journal={ACM Transactions on Graphics (ToG)},
  volume={35},
  number={6},
  pages={1--12},
  year={2016},
  publisher={ACM New York, NY, USA}
}
```
