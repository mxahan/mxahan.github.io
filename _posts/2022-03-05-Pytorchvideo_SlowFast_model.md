# Slowfast Model tutorial


## Basics

Requirement:
- Python >3.7
- Pytorch: >=1.9
- Pytorchvideo
- Torchvision >= 0.10.0

### Bringing slowfast model

There is two ways we can bring a slowfast model with random weights initialised.

- **torch.load**: Will give us the model directly.

```
model_name = "slowfast_r50" # or "slowfast_r101"
model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=False)
```

To take the pretrained weights

```
model_name = "slowfast_r50" # or "slowfast_r101"
model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
```

We can verify the model by checking the model variable. We can use the following setup to count the parameter numbers.

```
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(model))
```

- **create_slowfast**

```
import pytorchvideo.models as models
from pytorchvideo.models import create_slowfast
model = create_slowfast(model_depth=50) # Alternative 18, 101, 152
```
The created model should have equal number of parameters as earlier. To load the pretrained weight we require to download the weights from the [link](https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html) and load the weights. We can use the followings to load the weights.

```
pretrain = 1
if pretrain == 1:
    wt = torch.load("../Directory/SLOWFAST_8x8_R50.pyth")
    model.load_state_dict(wt["model_state"])
```

However, to load the model zoo weight of slowfast 100 model do the followings

```
model = create_slowfast(model_depth=101, slowfast_fusion_conv_kernel_size = (5, 1, 1,))
```

### Inferring using the model

The pytorchvideo slowfast model takes list (formatted in a specific way) as input. It builds upon a class called *multiPathWayWithFuse* that takes list as input. Anyways, we require to do the following to infer using the model

```
#%% Defining functions
from pytorchvideo.transforms import  UniformTemporalSubsample
transform1=Compose(
    [UniformTemporalSubsample(num_frames),
        PackPathway()])

#%% Input processing
ipt_vid_frm = 64_consecutive_video_frame # shape of (3, 64, 224, 224)
f = transform1(ipt_vid_frm)
f = [_.to(device)[None, ...] for _ in f]
output = model(f)
```
**Interesting fact**: The model execution change the shape of input *f* since *f* is a list and when we process python list inside another function the list changes accordingly.


## Intermediate

### Batch Data processing

One issue I faces is to batching the data for the model to train/infer. Here's my code to solve the issue.

```
device = 'cuda:0' # 'cpu'
#%% Defining requirement
def sf_data_ready(vdat):

    transform1=Compose(
        [
            UniformTemporalSubsample(num_frames),
            # Lambda(lambda x: x/255.0),
            # NormalizeVideo(mean, std),
            # ShortSideScale(
            #     size=side_size
            # ),
            # CenterCropVideo(crop_size),
            PackPathway()
        ]
    )


    s2 =[[], []]
    for i in range(vdat.shape[0]):
        f = transform1(vdat[i])
        f = [_.to(device)[None, ...] for _ in f]
        s2[0].append(f[0])
        s2[1].append(f[1])

    for i in range(2):s2[i] = torch.cat((s2[i][0:]))

    return s2

#%% Implementation
video_batch= torch.rand(BS, 3, 64, 224, 224) # BS = Batch size
inputs = sf_data_ready(video_batch)
outputs  = model(inputs)
```
### Modifying Model Structure

We can build our custom slowfast model by providing different arguments choice for the create_slowfast model as enlisted in the [docs](https://pytorchvideo.readthedocs.io/en/latest/_modules/pytorchvideo/models/slowfast.html#create_slowfast). The list is exhaustive however, we will go over as much as we can,

- slowfast_channel_reduction_ratio: Corresponds to the inverse of the channel reduction ratio, beta between the Slow and Fast pathways. Default: (8, )
  - the Higher value would give model with lower params (fast channel reduces)

```
slowfast = create_slowfast(slowfast_channel_reduction_ratio= (v_r,)) # v_r <=64
```

- slowfast_conv_channel_fusion_ratio (int): Ratio of channel dimensions between  the Slow and Fast pathways. Default: 2
  - Note: the higher, the number of parameter increases (can go super high although should not)
- DEPRECATED slowfast_fusion_conv_kernel_size (tuple): the convolutional kernel size used for fusion. [self-explained]
- DEPRECATED slowfast_fusion_conv_stride (tuple): the convolutional stride size used for fusion. [self-explained]

Well, the idea is we can go over this and create a customized model. One drawback of such approach is that we might not have the pretrained weights for our custom model!

Here is some args to look at: **input_channel**: May allow color and gray scale simultaneously, **model_depth**: Model depth (18, 50, 101, 151), **head**: Do we want the projection after the final fusion.

## Advanced

The interesting fact that, SF model works with varying input size of (224, 224) or (256, 256). How is this possible even we have linear layer in the later section? Shouldn't we encounter matrix multiplication error??
  - Well, they use **torch.nn.AdaptiveAvgPool3d** to create uniform shape at the later stage (before linear layer).

### Modifying Model structure Heavily

We can also enter the layers and modify them as we suit as example below

```
model.blocks[6] = nn.Sequential(*list(model.blocks[6].children())[:-2], nn.AdaptiveAvgPool3d(1),
                                nn.Flatten(),
                                nn.Linear(in_features=2304, out_features=1000, bias=True), nn.ReLU()
                              )
```

We can only change the linear layer too.
```
model.blocks[6].proj = nn.Linear(in_features=2304, out_features=1000)
```

However, beware of the following [bug](https://discuss.pytorch.org/t/pytorch-is-allowing-incorrect-matrix-multiplication-when-using-cuda/130664) of pytorch 1.9 version.

Happy coding.
