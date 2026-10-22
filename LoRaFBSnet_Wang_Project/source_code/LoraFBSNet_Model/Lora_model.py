import torch.nn as nn
from spikingjelly.activation_based import base, layer, neuron, surrogate, functional
import torch
import torch.nn.functional as F

class FeedbackConnection(nn.Module):
    def __init__(self, source_channels, target_channels, source_stride, target_stride):
        super().__init__()
        self.last_output = None
        self.source_channels = source_channels
        self.target_channels = target_channels
        self.source_stride = source_stride
        self.target_stride = target_stride
        if source_stride % target_stride != 0:
            raise ValueError(
                f"Source Stride {source_stride} Must Be "
                f"Divisible by Target Stride {target_stride}"
            )
        scale_factor = source_stride // target_stride
        modules = []
        current_channels = source_channels
        if scale_factor > 1:
            if scale_factor & (scale_factor - 1) != 0:
                raise ValueError("Scale Factor must be a Power of 2")
            num_upsamples = int(torch.log2(torch.tensor(float(scale_factor))).item())
            for _ in range(num_upsamples):
                modules.append(
                    nn.ConvTranspose2d(
                        current_channels,
                        current_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        groups=current_channels,
                        bias=False
                    )
                )
                modules.append(layer.BatchNorm2d(current_channels))
                modules.append(neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True))
            if current_channels != target_channels:
                modules.append(
                    nn.Conv2d(
                        current_channels,
                        target_channels,
                        kernel_size=1,
                        bias=False
                    )
                )
                modules.append(neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True))
            self.feedback = nn.Sequential(*modules)

    def forward(self, x):
        out = self.feedback(x)
        self.last_output = out.detach()
        return out


class ConvRecurrentContainer(base.MemoryModule):
    def __init__(self, sub_module, in_channels, out_channels, stride, step_mode='s', recurrent_enabled=True,
                 recurrent_scale=1.0, external_feedback_channels=None):
        super().__init__()
        self.step_mode = step_mode
        assert (
                not hasattr(sub_module, "step_mode")
                or sub_module.step_mode == "s"
        )
        self.recurrent_enabled = recurrent_enabled
        self.recurrent_scale = recurrent_scale
        self.sub_module_out_channels = out_channels
        self.sub_module = sub_module
        if external_feedback_channels is None:
            self.external_feedback_channels = []
        elif isinstance(external_feedback_channels, (list, tuple)):
            self.external_feedback_channels = list(external_feedback_channels)
        else:
            self.external_feedback_channels = [external_feedback_channels]
        self.total_external_feedback_channels = int(sum(self.external_feedback_channels))

        if stride > 1:
            self.dwconv = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=1,
                groups=out_channels,
                bias=False
            )

            self.bn1 = layer.BatchNorm2d(out_channels)

            self.sn1 = neuron.LIFNode(
                surrogate_function=surrogate.ATan(),
                detach_reset=True
            )
        else:
            self.dwconv = None
        total_input_channels = in_channels + out_channels + self.total_external_feedback_channels
        self.pwconv = nn.Conv2d(total_input_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.bn = layer.BatchNorm2d(in_channels)

        self.sn = neuron.LIFNode(
            surrogate_function=surrogate.ATan(),
            detach_reset=True
        )
        self.last_recurrent = None
        self.last_output = None
        self.last_external_feedback = None
        self.register_memory("y", None)

    def set_recurrent(self, enabled=None, scale=None):
        if enabled is not None:
            self.recurrent_enabled = enabled

        if scale is not None:
            self.recurrent_scale = scale

    def reset_recurrent_state(self):
        self.y = None

    def _prepare_recurrent_state(self, x):
        B, _, H, W = x.shape
        if self.y is None:
            return torch.zeros(B, self.sub_module_out_channels, H, W, device=x.device, dtype=x.dtype)
        y = self.y.to(device=x.device, dtype=x.dtype)
        if self.dwconv is not None:
            y = self.dwconv(y)
            y = self.bn1(y)
            y = self.sn1(y)
        if y.shape[-2:] != (H, W):
            y = F.interpolate(
                y,
                size=(H, W),
                mode='nearest'
            )
        return y

    def single_step_forward(self, x, external_feedback=None):
        B, _, H, W = x.shape
        fusion_inputs = [x]
        if self.recurrent_enabled:
            recurrent = self._prepare_recurrent_state(x)
            recurrent = (recurrent * self.recurrent_scale)

        else:
            recurrent = torch.zeros((B, self.sub_module_out_channels, H, W), device=x.device, dtype=x.dtype)
        self.last_recurrent = recurrent.detach()
        fusion_inputs.append(recurrent)

        # 2. External Feedback
        if self.total_external_feedback_channels > 0:
            if external_feedback is None:
                external_feedback = torch.zeros(
                    (B, self.total_external_feedback_channels, H, W),
                    device=x.device,
                    dtype=x.dtype
                )
            else:
                if external_feedback.shape[-2:] != (H, W):
                    external_feedback = F.interpolate(external_feedback, size=(H, W), mode='nearest')
                if external_feedback.shape[1] != self.total_external_feedback_channels:
                    raise RuntimeError(
                        f"External Feedback Channels Mismatch: "
                        f"Expected "
                        f"{self.total_external_feedback_channels}"
                        f"got "
                        f"{external_feedback.shape[1]}"
                    )
            self.last_external_feedback = external_feedback.detach()
            fusion_inputs.append(external_feedback)
        else:
            self.last_external_feedback = None
        out = torch.cat(fusion_inputs, dim=1)
        out = self.bn(self.pwconv(out))
        out = self.sn(out)
        self.y = self.sub_module(out)
        self.last_output = self.y.detach()
        return self.y


def sew_function(x, y, cnf):
    if cnf == "ADD":
        return x + y
    elif cnf == "AND":
        return x * y
    elif cnf == "OR":
        return x + y - x * y
    else:
        raise NotImplementedError


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return layer.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    return layer.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self, in_planes, planes, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None, cnf=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 "
                "and base_width=64"
            )
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)

        self.stride = stride
        self.cnf = cnf

    def forward(self, x):
        identity = x
        out = self.sn1(self.bn1(self.conv1(x)))
        out = self.sn2(self.bn2(self.conv2(out)))
        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = sew_function(out, identity, self.cnf)
        return out


# ============================================================
# Bottleneck
# ============================================================

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None,
                 cnf=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.sn2 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.downsample = downsample

        if downsample is not None:
            self.downsample_sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)

        self.stride = stride
        self.cnf = cnf

    def forward(self, x):
        identity = x
        out = self.sn1(self.bn1(self.conv1(x)))
        out = self.sn2(self.bn2(self.conv2(out)))
        out = self.sn3(self.bn3(self.conv3(out)))

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = sew_function(out, identity, self.cnf)
        return out


class LoRaFBSNet(nn.Module):
    def __init__(self, block, layers, num_classes=3, groups=1, width_per_groups=64, norm_layer=None, cnf="ADD",
                 zero_init_residual=False, local_recurrence=True, recurrent_scale=1, feedback_range=0):
        super().__init__()
        if feedback_range < 0 or feedback_range > 3:
            raise ValueError("Feedback Range must be between 0 to 3")
        self.feedback_range = feedback_range
        self.local_recurrence = local_recurrence
        self.feedback_enabled = (feedback_range > 0)
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self.norm_layer = norm_layer
        self.in_planes = 64
        self.groups = groups
        self.base_width = width_per_groups
        self.conv1 = layer.Conv2d(1, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_layer(self.in_planes)
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion,
                               512 * block.expansion]
        self.stage_strides = [1, 2, 4, 8]
        layer1 = self._make_layer(block, 64, layers[0], cnf=cnf)
        layer2 = self._make_layer(block, 128, layers[1], stride=2, cnf=cnf)
        layer3 = self._make_layer(block, 256, layers[2], stride=2, cnf=cnf)
        layer4 = self._make_layer(block, 512, layers[3], stride=2, cnf=cnf)
        self.feedback_graph = self._build_feedback_graph(feedback_range)
        incoming_feedback_channels = {
            0: [],
            1: [],
            2: [],
            3: []
        }
        for source_idx, target_idx in self.feedback_graph:
            incoming_feedback_channels[target_idx].append(self.stage_channels[target_idx])

        # ----------------------------------------------------
        # Recurrent containers
        # ----------------------------------------------------

        self.recurrent_layer1 = ConvRecurrentContainer(
            layer1,
            in_channels=64,
            out_channels=self.stage_channels[0],
            stride=1,
            recurrent_enabled=local_recurrence,
            recurrent_scale=recurrent_scale,
            external_feedback_channels=incoming_feedback_channels[0]
        )

        self.recurrent_layer2 = ConvRecurrentContainer(
            layer2,
            in_channels=self.stage_channels[0],
            out_channels=self.stage_channels[1],
            stride=2,
            recurrent_enabled=local_recurrence,
            recurrent_scale=recurrent_scale,
            external_feedback_channels=incoming_feedback_channels[1]
        )

        self.recurrent_layer3 = ConvRecurrentContainer(
            layer3,
            in_channels=self.stage_channels[1],
            out_channels=self.stage_channels[2],
            stride=2,
            recurrent_enabled=local_recurrence,
            recurrent_scale=recurrent_scale,
            external_feedback_channels=incoming_feedback_channels[2]
        )

        self.recurrent_layer4 = ConvRecurrentContainer(
            layer4,
            in_channels=self.stage_channels[2],
            out_channels=self.stage_channels[3],
            stride=2,
            recurrent_enabled=local_recurrence,
            recurrent_scale=recurrent_scale,
            external_feedback_channels=incoming_feedback_channels[3]
        )

        # ----------------------------------------------------
        # Feedback modules
        # ----------------------------------------------------

        self.feedback_connections = nn.ModuleDict()
        for source_idx, target_idx in self.feedback_graph:
            name = f"{source_idx + 1}_to_{target_idx + 1}"
            self.feedback_connections[name] = (
                FeedbackConnection(
                    source_channels=self.stage_channels[source_idx],
                    target_channels=self.stage_channels[target_idx],
                    source_stride=self.stage_strides[source_idx],
                    target_stride=self.stage_strides[target_idx]
                )
            )
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.stage_channels[3], num_classes)
        for m in self.modules():
            if isinstance(m, (layer.Conv2d, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # ========================================================
    # Build Feedback Graph
    # ========================================================

    def _build_feedback_graph(self, feedback_range):
        graph = []
        for source in range(4):
            for distance in range(1, feedback_range + 1):
                target = source - distance
                if target >= 0:
                    graph.append((source, target))
        return graph

    # ========================================================
    # Make ResNet Layer
    # ========================================================

    def _make_layer(self, block, planes, num_blocks, stride=1, cnf=None):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                layer.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion)
            )
        layers_list = [
            block(self.in_planes, planes, stride, downsample, self.groups, self.base_width, self.norm_layer, cnf)
        ]

        self.in_planes = (planes * block.expansion)

        for _ in range(1, num_blocks):
            layers_list.append(
                block(self.in_planes, planes, groups=self.groups, base_width=self.base_width,
                      norm_layer=self.norm_layer, cnf=cnf))

        return nn.Sequential(*layers_list)

    # ========================================================
    # Feedback API
    # ========================================================

    def set_local_recurrence(self, enabled=None, scale=None, layers=None):

        recurrent_layers = {
            1: self.recurrent_layer1,
            2: self.recurrent_layer2,
            3: self.recurrent_layer3,
            4: self.recurrent_layer4
        }

        if layers is None:
            layers = [1, 2, 3, 4]

        for layer_idx in layers:
            recurrent_layers[layer_idx].set_recurrent(enabled=enabled, scale=scale)

    def set_feedback_enabled(self, enabled=True):
        self.feedback_enabled = enabled

    # ========================================================
    # Get External Feedback
    # ========================================================

    def _compute_feedback(self, target_idx, previous_outputs):
        if not self.feedback_enabled:
            return None
        feedback_tensors = []

        for source_idx, graph_target_idx in self.feedback_graph:
            if graph_target_idx != target_idx:
                continue
            prev_y = previous_outputs[source_idx]
            if prev_y is None:
                continue
            name = f"{source_idx + 1}_to_{target_idx + 1}"
            feedback = self.feedback_connections[name](prev_y)
            feedback = feedback.to(
                device=prev_y.device,
                dtype=prev_y.dtype
            )
            feedback_tensors.append(feedback)
        if not feedback_tensors:
            return None
        return torch.cat(feedback_tensors, dim=1)

    # ========================================================
    # Reset State
    # ========================================================
    def reset_state(self):

        self.recurrent_layer1.reset_recurrent_state()
        self.recurrent_layer2.reset_recurrent_state()
        self.recurrent_layer3.reset_recurrent_state()
        self.recurrent_layer4.reset_recurrent_state()
        functional.reset_net(self)

    # ========================================================
    # Forward Method
    # ========================================================

    def forward(self, x):
        self.reset_state()
        T, B, C, H, W = x.shape
        x_seq = []
        for t in range(T):
            xt = self.conv1(x[t])
            xt = self.bn1(xt)
            xt = self.sn1(xt)
            xt = self.maxpool(xt)
            x_seq.append(xt)
        x = torch.stack(x_seq, dim=0)
        out_seq = []
        for t in range(T):
            xt = x[t]
            previous_outputs = [
                self.recurrent_layer1.y,
                self.recurrent_layer2.y,
                self.recurrent_layer3.y,
                self.recurrent_layer4.y
            ]
            feedback_l1 = self._compute_feedback(target_idx=0, previous_outputs=previous_outputs)
            x1 = self.recurrent_layer1.single_step_forward(xt, external_feedback=feedback_l1)

            feedback_l2 = self._compute_feedback(target_idx=1, previous_outputs=previous_outputs)
            x2 = self.recurrent_layer2.single_step_forward(x1, external_feedback=feedback_l2)

            feedback_l3 = self._compute_feedback(target_idx=2, previous_outputs=previous_outputs)
            x3 = self.recurrent_layer3.single_step_forward(x2, external_feedback=feedback_l3)

            feedback_l4 = self._compute_feedback(target_idx=3, previous_outputs=previous_outputs)
            x4 = self.recurrent_layer4.single_step_forward(x3, external_feedback=feedback_l4)
            x_model = self.avgpool(x4)
            x_model = torch.flatten(x_model, 1)
            x_model = self.fc(x_model)
            out_seq.append(x_model)
        out = torch.stack(out_seq, dim=0)
        return out


def lorafb_snet18(**kwargs):
    return LoRaFBSNet(BasicBlock, [2, 2, 2, 2], **kwargs)