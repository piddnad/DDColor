## DDColor Model Zoo

| Model                 | Description          |  Note |
| ---------------------- | :------------------ | :-----|
| [ddcolor_paper.pth](https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_paper.pth)      | DDColor-L trained on ImageNet   | paper model, use it only if you want to reproduce some of the images in the paper.
| [ddcolor_modelscope.pth](https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_modelscope.pth) (***default***)  | DDColor-L trained on ImageNet   | We trained this model using the same data cleaning scheme as [BigColor](https://github.com/KIMGEONUNG/BigColor/issues/2#issuecomment-1196287574), so it can get the best qualitative results with little degrading FID performance. Use this model by default if you want to test images outside the ImageNet. It can also be easily downloaded through ModelScope [in this way](README.md#inference-with-modelscope-library).
| [ddcolor_artistic.pth](https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_artistic.pth) | DDColor-L trained on ImageNet + private data | We trained this model with an extended dataset containing many high-quality artistic images. Also, we didn't use colorfulness loss during training, so there may be fewer unreasonable color artifacts. Use this model if you want to try different colorization results.
| [ddcolor_paper_tiny.pth](https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_paper_tiny.pth) | DDColor-T trained on ImageNet   | The most lightweight version of ddcolor model, using the same training scheme as ddcolor_paper.

## Discussions

* About Colorfulness Loss (CL): CL can encourage more "colorful" results and help improve CF scores, however, it sometimes leads to the generation of unpleasant color blocks (eg. red color artifacts). If something goes wrong, I personally recommend trying to remove it during training.

