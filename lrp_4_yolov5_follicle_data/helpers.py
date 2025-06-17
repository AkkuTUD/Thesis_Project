from matplotlib import pyplot as plt
import matplotlib as mpl
import torch
import torchvision
from PIL import Image
import numpy as np

_ = torch.hub.load('ultralytics/yolov5', "yolov5s", pretrained=True)

from utils.general import xywh2xyxy
from models.common import Conv, Bottleneck

from matplotlib.colors import ListedColormap
import types



def get_image_tensor(img):
    
    tile_tensor = torchvision.transforms.ToTensor()(img)

    if tile_tensor.ndimension() == 3:
        tile_tensor = tile_tensor.unsqueeze(0)
    
    return tile_tensor

def load_image(img_path):
    return Image.open(img_path).convert("RGB")

def load_image(img_path):
    return Image.open(img_path).convert("RGB")

class meta_model():
    def __init__(self, 
                 model, 
                 detect_thresh, 
                 nms_thresh, 
                 conf_thres=0.25, 
                 iou_thres=0.45, 
                 classes=None, 
                 agnostic=False,
                 multi_label=False,
                 labels=(),
                 max_det=300):
        self.model = model
        self.detect_thresh = detect_thresh
        self.nms_thresh = nms_thresh
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic = agnostic
        self.multi_label = multi_label
        self.labels = labels
        self.max_det = max_det
        self.lrp_ = None

    def predict(self, instance, grad=False):
        if grad:
            preprocessed_x = get_image_tensor(instance['image'])
            preprocessed_x.requires_grad_(True)
            preprocessed_x.grad = None

            raw_prediction = self.model(preprocessed_x)[0]

            return self.non_max_suppression_modified(raw_prediction)
        else:
            prediction = self.model(get_image_tensor(instance['image']))[0]
            #print("prediction shape:", prediction.shape)
            # NMS
            prediction, _, _ = self.non_max_suppression_modified(prediction)
            #print('pred after nms', prediction)
            # pred: xyxy, conf, class
            if isinstance(prediction, list):
                if len(prediction) == 1:
                    return prediction[0]
                else:
                    raise NotImplementedError
            else:
                return prediction

    # We override `non_max_suppression` to make it compatible with .backward(_).

    def non_max_suppression_modified(self, prediction 
                                    #conf_thres = conf_thres, 
                                    #ou_thres = self.ou_thres, 
                                    #classes= self.classes, 
                                    #agnostic=False, 
                                    #multi_label=False,
                                    #labels=(),
                                    #max_det=300
                                    ):
        """Runs Non-Maximum Suppression (NMS) on inference results
        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > self.conf_thres  # candidates

        # Checks
        assert 0 <= self.conf_thres <= 1, f'Invalid Confidence threshold {self.conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= self.iou_thres <= 1, f'Invalid IoU {self.iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        self.multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        # t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        arr_good_raw_predictions = [None] * prediction.shape[0]
        for xi, raw_prediction in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            raw_prediction = raw_prediction[xc[xi]]  # confidence


            # Cat apriori labels if autolabelling
            if self.labels and len(self.labels[xi]):
                raise SystemExit("We haven't implemented this")
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=raw_prediction.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                raw_prediction = torch.cat((raw_prediction, v), 0)

            # If none remain process next image
            if not raw_prediction.shape[0]:
                continue


            class_obj_conf = raw_prediction[:, 5:] * raw_prediction[:, 4:5]  # conf = cls_conf * obj_conf
            assert class_obj_conf.shape[0] == raw_prediction.shape[0], "Sanity check number of boxes"

            # pprint(f"class_obj_conf's shape: {class_obj_conf.shape}")
            # note: class_obj_conf's shape: torch.Size([_, 80])

            box_parameters = raw_prediction[:, :4]

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(box_parameters)

            # Detections matrix nx6 (xyxy, conf, cls)
            if self.multi_label:
                raise SystemExit("[pat]: The current implementation doesn't support this branch")
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = class_obj_conf.max(1, keepdim=True)
                ix_conf_exceeds_threshold = conf.view(-1) > self.conf_thres
                x = torch.cat((box, conf, j.float()), 1)[ix_conf_exceeds_threshold]
                # pprint("----")
                # pprint(f"box.shape: {box.shape}")
                # pprint(f"conf.shape: {conf.shape}")
                # pprint(f"j.shape: {j.shape}")
                # pprint(f"sum(ix_conf_exceeds_threshold): {ix_conf_exceeds_threshold.sum()}")

            good_raw_predictions = raw_prediction[ix_conf_exceeds_threshold]
            assert good_raw_predictions.shape[0] == ix_conf_exceeds_threshold.sum()

            # Pat's note: x's shape (number good candidates, 4 (box params)+ 1 (class_conf) + 1 (class_conf))

            # Filter by class
            if self.classes is not None:
                raise SystemExit("[pat]: The current implementation doesn't support this branch")
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            # pprint(f"Number of boxes: {n}")
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                # todo: get indices from this
                sorted_ix = x[:, 4].argsort(descending=True)[:max_nms]

                x = x[sorted_ix]  # sort by confidence
                good_raw_predictions = good_raw_predictions[sorted_ix]

            # Batched NMS
            # pprint(f"agnostic={agnostic}")
            c = x[:, 5:6] * (0 if self.agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            # pprint(f"boxes'shape: {boxes.shape}")

            assert good_raw_predictions.shape[0] == x.shape[0]

            i = torchvision.ops.nms(boxes, scores, self.iou_thres)  # NMS
            if i.shape[0] > self.max_det:  # limit detections
                i = i[:self.max_det]

            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                raise SystemExit("[pat] we haven't implemented this")
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy


            
            output[xi] = x[i]
            arr_good_raw_predictions[xi] = good_raw_predictions[i]

            assert output[xi].shape[0] == arr_good_raw_predictions[xi].shape[0]
            
        # print("arr_good_raw_predictions",arr_good_raw_predictions)
        return output, arr_good_raw_predictions, xc
    
    ## for LRP
    # This recursive function allows us to query modules belonging to a specific type.
    def select(self, module, target_type, level=0):
        if type(module) == target_type:
            return [module]
        elif hasattr(module, "children"):
            modules = []
            for child in module.children():
                modules.extend(self.select(child, target_type, level=level+1))

        if level == 0:
            print(f"{target_type}: {len(modules)}")
        return modules
    
    # This module trivially represents the last operator in NFNets
    class Summation(torch.nn.Module):
        def forward(self, a, s):
            return a + s
        
    def bottleneck_overriden_forward(self, x):
        return self.__summation(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
    
    def get_summations_in_bottleneck(self):
        all_bn_modules = self.select(self.model.model, Bottleneck)
        print(f"We have {len(all_bn_modules)} Bottlenecks")

        all_bn_modules_with_add = list(filter(lambda m: m.add, all_bn_modules))
        print(f"and {len(all_bn_modules_with_add)} of them with add=True")

        summations = []
        for bn in all_bn_modules_with_add:
            # https://github.com/ultralytics/yolov5/blob/master/models/common.py#L103
            # the module uses this condition whether it will add the input
            # if yes, we override the forward pass with summation module
            if bn.add:
                mod_sum = self.Summation()
                bn.__summation = mod_sum

                bn.forward = types.MethodType(self.bottleneck_overriden_forward, bn)

                summations.append(mod_sum)  

        assert len(summations) == len(all_bn_modules_with_add)
        return summations    

    
    def prepare_exp(self):

        detector = list(self.model.model.children())[24]
        # We disable inplace operators to be able to run backward().
        detector.inplace = False

        arr_yolo_convs = self.select(self.model.model, Conv)
        focus_conv = arr_yolo_convs[0]

        assert focus_conv == self.model.model[0].conv, "We select the correct conv for the Focus module"

        arr_inside_yolo_convs = arr_yolo_convs[1:]
        assert len(arr_inside_yolo_convs) == len(arr_yolo_convs) - 1, "We correctly filter out the first layer Conv."
        assert focus_conv not in arr_inside_yolo_convs, "Sanity check that Module Focus's Conv is excluded from the list of inside Convs."

        arr_detection_convs = self.select(self.model.model[24], torch.nn.Conv2d)

        arr_summation_shortcuts = self.get_summations_in_bottleneck()

        self.lrp_ = [focus_conv, arr_yolo_convs, arr_inside_yolo_convs, arr_detection_convs, arr_summation_shortcuts] 


def plot_grid(size):
    h, w = size
    for ix, gx in enumerate(range(0, h, h // 8)):  
        alpha = 0.4 if ix % 2 == 0 else 0.1
        plt.axhline(gx, ls="-", lw=1, alpha=alpha, color="k")
        
    for ix, gx in enumerate(range(0, w, w // 8)):    
        alpha = 0.4 if ix % 2 == 0 else 0.1
        plt.axvline(gx, ls="-", lw=1, alpha=alpha, color="k")

def plot_prediction(img, model_prediction, image_filename):
    plt.imshow(img)
    
    ax = plt.gca()
    
    # grid
    plot_grid(img.size)
    
    n = len(model_prediction)

    filename_slug = image_filename.split("/")[-1]
    plt.title(f"{filename_slug}: {n} Detections(s)")
    for ix in range(n):
        
        prediction = model_prediction[ix].detach().cpu().numpy().reshape(-1)

        pred_conf, cls_ix = prediction[4], prediction[5]
        cls_ix = int(cls_ix)
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = bb = prediction[:4]

        box = mpl.patches.Rectangle(
            (top_left_x, top_left_y),
            (bottom_right_x - top_left_x),
            (bottom_right_y - top_left_y),
            linewidth=5,
            edgecolor="r",
            facecolor="none"
        )
        ax.add_patch(box)
        ax.text(
          bottom_right_x + 5, top_left_y + 0.5*(bottom_right_y - top_left_y) ,
          f"{ix}",
          color="white",
          bbox=dict(facecolor="black", alpha=0.7),
          verticalalignment="center",
          fontsize=15
        )

def plot_heatmap(heatmap):
    """
    args:
        heatmap np.array(h, w):
        reference_heatmap (np.array(h, w), optional): used for calculating normalization values. Defaults to None.
        total_score (float, optional): used for normalizing scores. Defaults to None.
    """


    b = 10 * ((np.abs(heatmap) ** 3.0).mean() ** (1.0 / 3))

    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)

    sum_Ri = np.sum(heatmap)
  
    txt = r"$\sum R_i=%.4f$" % (sum_Ri)

    plt.xticks([]); plt.yticks([])

    plt.imshow(heatmap, cmap=my_cmap, vmin=-b, vmax=b)

    h = heatmap.shape[0]      


def show_explanation(exp, img, pred_ix, model_prediction, class_names, gamma, grad_of):

    prediction = model_prediction[pred_ix].detach().cpu().reshape(-1).tolist()
    cls_ix = int(prediction[5])
    print(prediction)
    plt.figure(figsize=(7*4, 7))
    
    conf = prediction[4]
    
    plt.subplot(1, 4, 1)
    ax = plt.gca()
    plt.title(f"Site {pred_ix}: {class_names[cls_ix]} (cls_ix={cls_ix}, conf: {conf:.4f})")
    plt.imshow(img)
    plot_grid(img.size)
    
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = bb = prediction[:4]

    box = mpl.patches.Rectangle(
            (top_left_x, top_left_y),
            (bottom_right_x - top_left_x),
            (bottom_right_y - top_left_y),
            linewidth=5,
            edgecolor="r",
            facecolor="none"
    )
    ax.add_patch(box)
        
    rel, prob_obj, prob_cls = exp.get_relevance_for(pred_ix)
    #rel, prob_obj, prob_cls = get_relevance_for2(pred_ix)
    
    plt.subplot(1, 4, 2)
    plot_grid(rel.shape)
    plot_heatmap(rel)
    plt.title("[Site %d] LRP$_{\gamma=%.4f}$ w.r.t. %s)" % (pred_ix, gamma, grad_of))
    
    # todo: this `prediction` need to return from img
    tx, ty, bx, by = list(map(lambda x: int(x), prediction[:4]))

    plt.subplot(1, 4, 3)
    
    np_img = np.array(img)[ty:by, tx:bx, :]
    size = np_img.shape[:2]
    plt.imshow(np_img)
    plot_grid(size)
    
    plt.subplot(1, 4, 4)
    bb_rel = rel[ty:by, tx:bx]
    plot_heatmap(bb_rel)
    plot_grid(size)


def display_exp_image(exp, img, model_prediction, dev = True):
    """
    the full heatmap is put together from the objects 
    rel: heatmap for an object
    """
    n_p = model_prediction.shape[0]
    plt.figure(figsize=(7*n_p, 7))
    heatmap_all = np.zeros((np.array(img).shape[0],np.array(img).shape[1]))
    for plot_ix in range(0,n_p+1):#, pred in enumerate(model_prediction):
        #print(pred_ix)
        # first row: raw imgages
        # first index: original image
        # index
        # 1: image, 2: pred_0, 3: pred_1, ...
        # 1+n_p: hm_image, 2+n_p: hm_pred_0, 3+n_p: hm_pred_1, ...
        if plot_ix == 0:
            # first row: plot total image            
            plt.subplot(2,n_p+1,plot_ix+1)
            for pred_ix, _ in enumerate(model_prediction):
                prediction = model_prediction[pred_ix]
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = bb = prediction[:4]
                ax = plt.gca()
                box = mpl.patches.Rectangle(
                        (top_left_x, top_left_y),
                        (bottom_right_x - top_left_x),
                        (bottom_right_y - top_left_y),
                        linewidth=3,
                        edgecolor="r",
                        facecolor="none",
                        label = 'rnnr'
                )
                ax.add_patch(box)
                pred_class = int(prediction[-1].detach().item())
                pred_index = 'id ' + str(int(pred_ix))
                ax.annotate(pred_index, (top_left_x,top_left_y))
            plt.imshow(img)
            # second row: plot sum of rel (at the end)
        else:
            #explain each object prediction 
            if dev:
                rel = np.array(img)[:,:,0]
            else: 
                rel, prob_obj, prob_cls = exp.get_relevance_for(plot_ix-1)
            heatmap_all = heatmap_all + rel
            # first row: crop object from raw image
            prediction = model_prediction[plot_ix-1].detach().cpu().reshape(-1).tolist() # correct index for the image at the beginning
            #top_left_x, top_left_y, bottom_right_x, bottom_right_y = bb = prediction[:4]
            tx, ty, bx, by = list(map(lambda x: int(x), prediction[:4]))
            plt.subplot(2,n_p+1,plot_ix+1)
            plt.title('id ' + str(plot_ix - 1) + ', tc: ' + str(0) + ', pc: ' + str(int(prediction[-1])))
            np_img = np.array(img)[ty:by, tx:bx, :]
            size = np_img.shape[:2]
            plt.imshow(np_img)
            plot_grid(size)
                
            #plt.imshow(img)
            # second row: explanations
            plt.subplot(2,n_p+1,plot_ix+2+n_p)
            bb_rel = rel[ty:by, tx:bx]
            plot_heatmap(bb_rel)
            plot_grid(size)
            #plt.imshow(np_img)

        #plot_heatmap(rel)
    plt.subplot(2,n_p+1,n_p+2)

    plot_heatmap(heatmap_all)
    plt.show()




# bb in image
# true and predicted classes
