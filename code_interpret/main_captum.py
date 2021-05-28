import captum
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from captum.attr import IntegratedGradients, Occlusion
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import GuidedBackprop, LayerActivation
from captum.attr import visualization as viz
from net.convnet import ConvNet
from config import device 

BATCH = 128
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='/datasets/CIFAR10', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False)
model = ConvNet()
model.load_state_dict(torch.load('pth/conv.pth'))
model.eval()

dataiter = iter(test_loader)
images, labels = dataiter.next()
outputs = model(images)
_, predicted = torch.max(outputs, 1)
ind = 3
inputs = images[ind].unsqueeze(0)
inputs.requires_grad = True

saliency = Saliency(model)
grads = saliency.attribute(inputs, target=labels[ind].item())
grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

original_image = np.transpose((images[ind].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

_ = viz.visualize_image_attr(None, original_image, method="original_image")
plt.savefig('captum_res/origin.png')
_ = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",show_colorbar=True)
plt.savefig('captum_res/saliency.png')

def attribute_image_features(algorithm, inputs, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(inputs,target=labels[ind],**kwargs)
    return tensor_attributions

ig = IntegratedGradients(model)
model.zero_grad()
attr_ig, delta = ig.attribute(inputs,target=labels[ind],baselines=inputs * 0, return_convergence_delta=True)
attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
_ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="absolute_value",show_colorbar=True)
plt.savefig('captum_res/inter_grad.png')

layer_act = LayerActivation(model, model.conv_block1)
#attribution = layer_act.attribute(input)
attr_ig, delta = ig.attribute(inputs,target=labels[ind],baselines=inputs * 0, return_convergence_delta=True)
attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
_ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="absolute_value",show_colorbar=True)
plt.savefig('captum_res/layer_act.png')

model.train()
gdp = GuidedBackprop(model)
attr_gdp = gdp.attribute(inputs, target=labels[ind])
attr_gdp = np.transpose(attr_gdp.squeeze().cpu().detach().numpy(), (1, 2, 0))
_ = viz.visualize_image_attr(attr_gdp, original_image, method="blended_heat_map",sign="absolute_value",show_colorbar=True)
plt.savefig('captum_res/gdp.png')

#换用一张图片探究occ的可视化
ind = 4
inputs = images[ind].unsqueeze(0)
original_image = np.transpose((images[ind].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
inputs.requires_grad = True
model.eval()
occlusion = Occlusion(model)
attributions_occ = occlusion.attribute(inputs,
                                       strides = (2, 2, 2),
                                       target=labels[ind],
                                       sliding_window_shapes=(2,2, 2),
                                       baselines=0)
_ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      original_image,
                                      ["original_image", "heat_map"],
                                      ["all", "absolute_value"],
                                      show_colorbar=True,
                                      outlier_perc=2,
)
plt.savefig('captum_res/occ_fog.png')

ind = 6
inputs = images[ind].unsqueeze(0)
original_image = np.transpose((images[ind].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
inputs.requires_grad = True
model.eval()
occlusion = Occlusion(model)
attributions_occ = occlusion.attribute(inputs,
                                       strides = (2, 2, 2),
                                       target=labels[ind],
                                       sliding_window_shapes=(2,2, 2),
                                       baselines=0)
_ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      original_image,
                                      ["original_image", "heat_map"],
                                      ["all", "absolute_value"],
                                      show_colorbar=True,
                                      outlier_perc=2,
)
plt.savefig('captum_res/occ.png')