import models.resnet
import models.densenet
import models.googlenet
import models.vgg
import models.mobilenetv2
import models.inceptionv3
import models.inceptionv4
import models.inception_resnetv2

from bag_of_local_features_models.bagnets.pytorchnet import bagnet9, bagnet17, bagnet33


from timm.models import create_model


def get_model(args):
    if args.model == 'resnet50':
        model = models.resnet.resnet50(pretrained=args.pretrained)
    elif args.model == 'resnet18':
        model = models.resnet.resnet18(pretrained=args.pretrained)
    elif args.model == 'resnet34':
        model = models.resnet.resnet34(pretrained=args.pretrained)
    elif args.model == 'resnet101':
        model = models.resnet.resnet101(pretrained=args.pretrained)
    elif args.model == 'resnet152':
        model = models.resnet.resnet152(pretrained=args.pretrained)
    elif args.model == 'wide_resnet50_2':
        model = models.resnet.wide_resnet50_2(pretrained=args.pretrained)
    elif args.model == 'wide_resnet101_2':
        model = models.resnet.wide_resnet101_2(pretrained=args.pretrained)

    elif args.model == 'googlenet':
        model = models.googlenet.googlenet(pretrained=args.pretrained)
    elif args.model == 'vgg16':
        model = models.vgg.vgg16(pretrained=args.pretrained)
    elif args.model == 'mobilenet_v2':
        model = models.mobilenetv2.mobilenet_v2(pretrained=args.pretrained)
    elif args.model == 'inceptionv3':
        model = models.inceptionv3.inception_v3(pretrained=args.pretrained)
    elif args.model == 'inceptionv4':
        model = models.inceptionv4.inceptionv4(pretrained="imagenet")
    elif args.model == 'inceptionresnetv2':
        model = models.inception_resnetv2.inceptionresnetv2(pretrained="imagenet")

    elif args.model == 'densenet121':
        model = models.densenet.densenet121(pretrained=args.pretrained)
    elif args.model == 'densenet161':
        model = models.densenet.densenet161(pretrained=args.pretrained)
    elif args.model == 'densenet169':
        model = models.densenet.densenet169(pretrained=args.pretrained)
    elif args.model == 'densenet201':
        model = models.densenet.densenet201(pretrained=args.pretrained)

    elif args.model == "bagnet9":
        model = bagnet9(pretrained=args.pretrained)
    elif args.model == "bagnet17":
        model = bagnet17(pretrained=args.pretrained)
    elif args.model == "bagnet33":
        model = bagnet33(pretrained=args.pretrained)

    elif args.model.split('_') [0] == 'vit':
        model = create_model(
            args.model,
            pretrained=True,
            num_classes=1000,
            in_chans=3,
            global_pool=args.gp,
            scriptable=args.torchscript)
    else:
        raise ValueError(f"Model {args.model} not supported.")
    return model
