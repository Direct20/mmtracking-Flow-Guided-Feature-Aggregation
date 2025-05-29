# Flow-guided Feature Aggregation for Video Object Detection

## Abstract

<!-- [ABSTRACT] -->

Extending state-of-the-art object detectors from image to video is challenging. The accuracy of detection suffers from degenerated object appearances in videos, e.g., motion blur, video defocus, rare poses, etc. Existing work attempts to exploit temporal information on box level, but such methods are not trained end-to-end. We present flowguided feature aggregation, an accurate and end-to-end learning framework for video object detection. It leverages temporal coherence on feature level instead. It improves the per-frame features by aggregation of nearby features along the motion paths, and thus improves the video recognition accuracy. Our method significantly improves upon strong single-frame baselines in ImageNet VID, especially for more challenging fast moving objects. Our framework is principled, and on par with the best engineered systems winning the ImageNet VID challenges 2016, without additional bells-and-whistles. The proposed method, together with Deep Feature Flow, powered the winning entry of ImageNet VID challenges 2017.

<!-- [IMAGE] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/34888372/142985575-4560a7c1-0402-428f-9094-ffb00d6b1e38.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```latex
@inproceedings{zhu2017flow,
  title={Flow-guided feature aggregation for video object detection},
  author={Zhu, Xizhou and Wang, Yujie and Dai, Jifeng and Yuan, Lu and Wei, Yichen},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={408--417},
  year={2017}
}
```
### This version of FGFA implementation supports batch size > 1! 
### Also, we solved the problem of mmtracking's flownet. It now behaves well on custom datasets.

