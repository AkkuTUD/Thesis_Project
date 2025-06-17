from matplotlib import pyplot as plt
import matplotlib as mpl
import torch
from functools import partial
import copy
from io import BytesIO
from PIL import Image
from helpers import plot_heatmap, plot_grid
from IPython.display import display
import pandas as pd

_ = torch.hub.load('ultralytics/yolov5', "yolov5s", pretrained=True)

from models.common import Conv, Bottleneck
import numpy as np
from helpers import get_image_tensor


# This recursive function allows us to query modules belonging to a specific type.
## copied to model class!!! remove here
def select(module, target_type, level=0):
  if type(module) == target_type:
    return [module]
  elif hasattr(module, "children"):
    modules = []
    for child in module.children():
      modules.extend(select(child, target_type, level=level+1))

  if level == 0:
    print(f"{target_type}: {len(modules)}")
  return modules


import types

#copied
# This module trivially represents the last operator in NFNets
class Summation(torch.nn.Module):
  def forward(self, a, s):
    return a + s

# copied
def bottleneck_overriden_forward(self, x):
    return self.__summation(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))

## copied to model class
def get_summations_in_bottleneck(model):
    all_bn_modules = select(model.model, Bottleneck)
    print(f"We have {len(all_bn_modules)} Bottlenecks")

    all_bn_modules_with_add = list(filter(lambda m: m.add, all_bn_modules))
    print(f"and {len(all_bn_modules_with_add)} of them with add=True")

    summations = []
    for bn in all_bn_modules_with_add:
        # https://github.com/ultralytics/yolov5/blob/master/models/common.py#L103
        # the module uses this condition whether it will add the input
        # if yes, we override the forward pass with summation module
        if bn.add:
            mod_sum = Summation()
            bn.__summation = mod_sum

            bn.forward = types.MethodType(bottleneck_overriden_forward, bn)

            summations.append(mod_sum)  

    assert len(summations) == len(all_bn_modules_with_add)
    return summations

def prepare_for_lrp(m_model):
    detector = list(m_model.model.model.children())[24]
    # We disable inplace operators to be able to run backward().
    detector.inplace = False

    arr_yolo_convs = select(m_model.model.model, Conv)
    focus_conv = arr_yolo_convs[0]

    assert focus_conv == m_model.model.model[0].conv, "We select the correct conv for the Focus module"

    arr_inside_yolo_convs = arr_yolo_convs[1:]
    assert len(arr_inside_yolo_convs) == len(arr_yolo_convs) - 1, "We correctly filter out the first layer Conv."
    assert focus_conv not in arr_inside_yolo_convs, "Sanity check that Module Focus's Conv is excluded from the list of inside Convs."

    arr_detection_convs = select(m_model.model.model[24], torch.nn.Conv2d)

    #ano()

    arr_summation_shortcuts = get_summations_in_bottleneck(m_model.model)

    return [focus_conv, arr_yolo_convs, arr_inside_yolo_convs, arr_detection_convs, arr_summation_shortcuts]

def label(grad_of):
    if grad_of == "conf":
        return "Conf"
    elif grad_of == "logit_class":
        return "Inverted Prob. Class"
    elif grad_of == "logit_obj":
        return "Inverted Prob. Obj"


def get_prob_class_obj(raw_prediction, class_ix):
    # raw_prediction: first five entries [w, h, x, y, prob. objectives] + [prob. for each class] 
    # print("classix", class_ix)
    prob_obj = raw_prediction[4]
    # print("prob_obj",prob_obj)
    prob_class = raw_prediction[5 + class_ix]
    # print("prob_class",prob_class)
    return prob_obj, prob_class


# This function is based on heatmapping.org/tutorial.
# It is a helper function to clone a module.
def newlayer(layer, g, without_bias=False):

    #print(layer)
    layer = copy.deepcopy(layer)
    #layer = layer.detach().clone()

    layer._forward_hooks = None

    try:
        layer.weight = torch.nn.Parameter(g(layer.weight))
        layer.weight.requires_grad_(False)
    except AttributeError:
        pass
    
    if without_bias:
        layer.bias = torch.nn.Parameter(torch.zeros_like(layer.bias))
        layer.bias.requires_grad_(False)
    else:
        try: layer.bias   = torch.nn.Parameter(g(layer.bias))
        except AttributeError:
            pass

    return layer


# to copy convolution layers
def get_modified_conv_layer(conv_layer, gamma):
    # We extract the positive and negative weights to the convolutional layers
    def _rho_p(g, w):
        return w + g * w.clamp(min=0)

    def _rho_n(g, w):
        return w + g * w.clamp(max=0)

    rho_p = partial(_rho_p, gamma)
    rho_n = partial(_rho_n, gamma)

    return (
        newlayer(conv_layer, rho_p,),
        newlayer(conv_layer, rho_p, without_bias=True),
        newlayer(conv_layer, rho_n),
        newlayer(conv_layer, rho_n, without_bias=True),
    )


# we use this to convert probablities to logit when explaining
def inverse_sigmoid(y):
  return torch.log(y / (1-y))


# This function is a stable way to make the devisions for LRP in forward hooks more stable.
def lrp_div_long(nom, denom, context):
    output = context.detach()

    # todo: for some reason, torch automatically remove the batch axis of context
    # could this be PyTorch's bug?
    if nom.shape[0] == 1 and len(nom.shape) == 4 and len(output.shape) == 3:
        output = output.unsqueeze(0)


    # this trick combats gettig nan from backprop of x/0.
    # see https://github.com/pytorch/pytorch/issues/4132
    nonzero_ix = (denom != 0).detach()

    output[nonzero_ix] *= (nom[nonzero_ix] / denom[nonzero_ix].detach())

    output[~nonzero_ix] = 0

    return output


# An helper to register the same forward hook for different modules
def register_forwardhook_for_modules(modules, hook):
    return list(
      map(lambda m: m.register_forward_hook(hook), modules)
    )


def inverse_alternate_slicing_and_concat(x, original_shape):
    """
      This is the invese of the alternating slicing and torch.cat
      implemented at
      https://github.com/ultralytics/yolov5/blob/master/models/common.py.

      Recall that the step is 
      torch.cat(
          [
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
          ], 
          1
      )
      This halves the spatial demensions and increases
      the channel dimension by 4, i.e.,
      (1, 3, h, w) → (1, 3⨉4, h/2, w/2)
    """
    input = np.zeros(original_shape)

    input[:, :, 0::2, 0::2] = x[:, 0:3]
    input[:, :, 1::2, 0::2] = x[:, 3:6]
    input[:, :, 0::2, 1::2] = x[:, 6:9]
    input[:, :, 1::2, 1::2] = x[:, 9:12]

    return input

def ano():

    for size in [(224, 550), (200, 50)]:
        x = torch.from_numpy(np.random.random(size=(1, 3, *size)))
        output = torch.cat(
          [
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
          ], 1
        )

    np.testing.assert_allclose(
        inverse_alternate_slicing_and_concat(output, x.shape),
        x.detach().numpy(),
        verbose=True,
        err_msg="Inverse Alternate Slicing is wrong"
    )
    print("Inverting Alternate Slicing: Sanity check Passed!")


class test_deepcopy():
    def __init__(self, model):
        self.model = model
        #self.image_raw = image_raw
    
    def test_copy(self):
        print('start copying')
        for instance in self.model:
            model_copy = copy.deepcopy(instance)

    
# 2do: have instance defined as in notebook: image data + label information
class explainer():
    def __init__(self, 
                 model,    
                 instance,             
                 detect_thresh, 
                 nms_thresh,
                 grad_of,
                 gamma,
                 n_max,
                 input_lv,
                 input_hv,
                 focus_conv,
                 arr_yolo_convs,
                 arr_inside_yolo_convs,
                 arr_detection_convs,
                 arr_summation_shortcuts
                 ):
        self.model = model
        self.instance = instance
        self.detect_thresh = detect_thresh
        self.nms_thresh = nms_thresh
        self.grad_of = grad_of
        self.gamma = gamma
        self.n_max = n_max
        self.input_lv = input_lv
        self.input_hv = input_hv
        self.focus_conv = focus_conv
        self.arr_yolo_convs= arr_yolo_convs,
        self.arr_inside_yolo_convs= arr_inside_yolo_convs,
        self.arr_detection_convs= arr_detection_convs,
        self.arr_summation_shortcuts= arr_summation_shortcuts

    # Case: YOLOv5's `Conv` and `torch.nn.Conv2d`
    def fh_conv(self, module, input, output): 
        if type(module) is torch.nn.Conv2d:
            actual_conv_module  = module
        elif type(module) is Conv:
            actual_conv_module  = module.conv
        else:
            raise ValueError("Something wrong happens!")

        pconv, pconvnb, nconv, nconvnb = get_modified_conv_layer(
        actual_conv_module,
        self.gamma
        )

        aj, = input
        # for YOLOv5's Conv: ak is the output of SiLU
        # for torch.nn.Conv2d: ak is raw activation, i.e. zk.
        ak = output

        def get_pos_neg_part(aj):
            aj_p = aj.clamp(min=0)
            aj_n = aj.clamp(max=0)

            # Positive case (z_k > 0)
            # This is the cases that the signs of a_j and w_{jk} align.
            zk_pp = pconv(aj_p)
            # We exluce bias here because we wil bias to be a neuron with weight 1.
            # Effectively, it means that  aj^- = 0 => aj^- x b = 0.
            zk_nn = nconvnb(aj_n)

            # Negative case (z_k < 0)
            zk_np = nconv(aj_p)
            zk_pn = pconvnb(aj_n)

            pos_part = (zk_pp + zk_nn) 
            assert not torch.isnan(pos_part).any(), "pos_part is nan"

            neg_part = (zk_np + zk_pn) 
            assert not torch.isnan(neg_part).any(), "neg_part is nan"

            return pos_part, neg_part

        pos_part, neg_part = get_pos_neg_part(aj)


        overriden_ak = lrp_div_long(pos_part, pos_part, ak.clamp(min=0)) \
        + lrp_div_long(neg_part, neg_part, ak.clamp(max=0))

        assert torch.allclose(overriden_ak, ak), "overrided output == original output"

        np.testing.assert_allclose(
        overriden_ak.shape,
        output.shape,
        err_msg="output is the same",
        verbose=True
        )

        return overriden_ak
    
    # Case: Input layer
    def fh_input_layer(self, module_focus, input, output): 
        aj, = input 
        ak = output 
        
        aj.retain_grad()
        lv = (torch.ones_like(aj) * self.input_lv).requires_grad_(True)
        hv = (torch.ones_like(aj) * self.input_hv).requires_grad_(True)

        mod_self = newlayer(module_focus.conv, lambda w: w)
        mod_lv = newlayer(module_focus.conv, lambda w: w.clamp(min=0))
        mod_hv = newlayer(module_focus.conv, lambda w: w.clamp(max=0))
        
        zk = mod_self(aj)
        zk_hv = mod_hv(hv)

        # in our setting, we can just actually remove this because lv=0.
        zk_lv = mod_lv(lv)

        z = zk - (zk_lv + zk_hv)

        assert not torch.isnan(z).any(), "output contain nan"

        overriden_ak = lrp_div_long(z, z, ak)

        assert torch.allclose(overriden_ak, ak), "the overrided zk is the same as the original"

        # this is variable will be used to get final explanation
        module_focus.__act_for_lrp_beta = (aj, lv, hv)

        return overriden_ak
    
    # Case: Residue Connection
    def fh_residue_connection(self, module, input, output):
        aj, sj = input
        zk = output

        ajp = aj.clamp(min=0)
        sjp = sj.clamp(min=0)

        asp_0 = (ajp + sjp)
        asp_gamma = asp_0 * (1 + self.gamma)

        ajn = aj.clamp(max=0)
        sjn = sj.clamp(max=0)

        asn_0 = (ajn + sjn)
        asn_gamma = asn_0 * (1 + self.gamma)

        pos_part = asp_gamma + asn_0
        neg_part = asn_gamma + asp_0

        overriden_zk = lrp_div_long(pos_part, pos_part, zk.clamp(min=0)) \
        + lrp_div_long(neg_part, neg_part, zk.clamp(max=0))

        assert torch.allclose(overriden_zk, zk), \
        "the overrided zk is the same as the original"

        return overriden_zk
    
    # explaining
    # 2do: classes as attribute...
    def get_relevance_for(self, data_ix, class_ix):
        """
        returns heatmap for an object regarding the predicted class
        data_ix: object index
        """
        
        try:
            hooks = [
                *register_forwardhook_for_modules([self.focus_conv], self.fh_input_layer),
                *register_forwardhook_for_modules(self.arr_inside_yolo_convs[0], self.fh_conv),
                *register_forwardhook_for_modules(self.arr_summation_shortcuts, self.fh_residue_connection)
            ]

            preprocessed_x = get_image_tensor(self.instance['image'])
            preprocessed_x.requires_grad_(True)
            preprocessed_x.grad = None
            
            
            results, arr_good_raw_predictions, conf_selected = self.model.predict(self.instance, grad = True)
            # prediction's shape = (first 4 bounding box params, 1 conf, 1 class_ix)
            prediction = results[0][data_ix].reshape(-1)

            class_ix = torch.tensor(class_ix)
            conf = prediction[4]
            prob_obj, prob_class = get_prob_class_obj(arr_good_raw_predictions[0][data_ix], class_ix)


            if self.grad_of == "conf":
                conf.backward()
            elif self.grad_of == "logit_class":
                logit = inverse_sigmoid(prob_class)
                logit.backward(retain_graph=True)
            elif self.grad_of == "logit_obj":
                logit = inverse_sigmoid(prob_obj)
                logit.backward()
            else:
                raise ValueError(f"{self.grad_of} is not valid.")

            # LRP-beta variables
            aj, aj_lb, aj_hb = self.focus_conv.__act_for_lrp_beta

            relevance_before_alternate_slicing = aj * aj.grad + aj_lb * aj_lb.grad + aj_hb * aj_hb.grad

            relevance_before_alternate_slicing = relevance_before_alternate_slicing.detach().numpy()

            # This is the invese of the alternating slicing and torch.cat(_) 
            input_relevance = inverse_alternate_slicing_and_concat(
            relevance_before_alternate_slicing,
            preprocessed_x.shape
            ).squeeze().sum(axis=0)
        except Exception as e:
            print(e)
            raise e
        finally:
            for h in hooks:
                h.remove()

            if hasattr(self.focus_conv, "__act_for_lrp_beta"):
                del self.focus_conv.__act_for_lrp_beta

        return input_relevance, prob_obj, prob_class
    
    def image_to_binary(self, image, instance, counter):
        buffer = BytesIO()
        image.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
        buffer.seek(0)
        binary_heatmap = buffer.read()    
        plt.show()
        plt.close()
        return binary_heatmap
    
    def get_heatmaps_for_instance(self, exp, instance, img, class_names):
        """
        function returns a heatmap for each potetial class for one image
        classes can in the future be taken from a model object
        """

        heatmap_all = np.zeros((np.array(img).shape[0],np.array(img).shape[1]))

        # model prediction
        model_prediction = self.model.predict(instance)
        for i in model_prediction:
            i[i<0]=0.0
        print("model preds", model_prediction)
        # number of objects
        n_p = model_prediction.shape[0]
        print("n_p", n_p)
        # number of labels
        n_l = len(instance['label'])
        print("labels:", instance['label'])
        print("n_l", n_l)
        # number of classes
        n_c = 9 #2

        # dataframe for all objects in an image
        df = pd.DataFrame() 
        if n_p >= n_l:
            for object_index in range(1,n_p+1):    
                print("object_index",object_index)
                
                prediction = model_prediction[object_index-1].detach().cpu().reshape(-1).tolist()
                tx, ty, bx, by = list(map(lambda x: int(x), prediction[:4]))
                # --> store rel somewhere (rel: heatmap for the object for the class)                   
                np_img = np.array(img)[ty:by, tx:bx, :]

                size = np_img.shape[:2]
                    
                df.loc[object_index-1, 'pred_cls'] = str(int(prediction[-1]))

                # crop objects from full heatmap
                fig = plt.figure(figsize=(10,10))
                plt.subplot(4,n_p+3,object_index+1) 
                if object_index > n_l:
                    plt.title('id ' + str(object_index - 1) + ', tc: ' + 'BG' + ', pc: ' + str(int(prediction[-1])))
                    df.loc[object_index-1, 'org_cls'] = str('BG')
                    df.loc[object_index-1, 'outcome'] = str('FP')
                else:
                    plt.title('id ' + str(object_index - 1) + ', tc: ' + str(int(instance['label'][object_index-1][0])) + ', pc: ' + str(int(prediction[-1])))
                    df.loc[object_index-1, 'org_cls'] = str(int(instance['label'][object_index-1][0]))
                    if str(int(instance['label'][object_index-1][0])) == str(int(prediction[-1])):
                        df.loc[object_index-1, 'outcome'] = str('TP')
                    else:
                        df.loc[object_index-1, 'outcome'] = str('Missclassified')
                plt.imshow(np_img)
                plot_grid(size)
                df.loc[object_index-1, 'binary_img'] = self.image_to_binary(plt, instance, counter=1)
                
                for clss in range(0,n_c):
                    fig = plt.figure(figsize=(10,10))
                    rel, prob_obj, prob_cls = self.get_relevance_for(object_index-1, class_ix=int(clss))
                    heatmap_all = heatmap_all + rel
                    if clss==int(prediction[-1]):
                        df.loc[object_index-1, 'conf'] = str(round(prob_cls.item(), 5))
                        df.loc[object_index-1, 'obj'] = str(round(prob_obj.item(), 5))
                    plt.subplot(4,n_p+3,clss+3)
                    plt.title(str(class_names[clss]), fontsize=8)
                    bb_rel = rel[ty:by, tx:bx]
                    plot_heatmap(bb_rel)
                    plot_grid(size)    
                    df.loc[object_index-1, 'heatmap_'+str(clss)] = self.image_to_binary(plt, instance, counter=2)
                df.loc[object_index-1, 'classes']='{0,1,2,3,4,5,6,7,8}' #'{0,1}'   
            display(df)
            return df
        else:
            print("FN")
            for object_index in range(1,n_l+1):
                if object_index<=n_p:
                    print("TP")
                    prediction = model_prediction[object_index-1].detach().cpu().reshape(-1).tolist()
                    tx, ty, bx, by = list(map(lambda x: int(x), prediction[:4]))
                    # --> store rel somewhere (rel: heatmap for the object for the class)                   
                    np_img = np.array(img)[ty:by, tx:bx, :]
                    size = np_img.shape[:2]
                        
                    df.loc[object_index-1, 'pred_cls'] = str(int(prediction[-1]))

                    # crop objects from full heatmap
                    fig = plt.figure(figsize=(10,10))
                    plt.subplot(4,n_p+3,object_index+1) 
                    
                    plt.title('id ' + str(object_index - 1) + ', tc: ' + str(int(instance['label'][object_index-1][0])) + ', pc: ' + str(int(prediction[-1])))
                    df.loc[object_index-1, 'org_cls'] = str(int(instance['label'][object_index-1][0]))
                    if str(int(instance['label'][object_index-1][0])) == str(int(prediction[-1])):
                        df.loc[object_index-1, 'outcome'] = str('TP')
                    else:
                        df.loc[object_index-1, 'outcome'] = str('Missclassified')

                    plt.imshow(np_img)
                    plot_grid(size)
                    df.loc[object_index-1, 'binary_img'] = self.image_to_binary(plt, instance, counter=1)
                    
                    for clss in range(0,n_c):
                        fig = plt.figure(figsize=(10,10))
                        rel, prob_obj, prob_cls = self.get_relevance_for(object_index-1, class_ix=int(clss))
                        heatmap_all = heatmap_all + rel
                        if clss==int(prediction[-1]):
                            df.loc[object_index-1, 'conf'] = str(round(prob_cls.item(), 5))
                            df.loc[object_index-1, 'obj'] = str(round(prob_obj.item(), 5))
                        plt.subplot(4,n_p+3,clss+3)
                        plt.title(str(class_names[clss]), fontsize=8)
                        bb_rel = rel[ty:by, tx:bx]
                        plot_heatmap(bb_rel)
                        plot_grid(size)    
                        df.loc[object_index-1, 'heatmap_'+str(clss)] = self.image_to_binary(plt, instance, counter=2)
                else:
                    print("FN")
                    ax, ay, wx, wy = list(map(lambda x: x, instance['label'][object_index-1][1:5]))
                    print("labels box", ax, ay, wx, wy)

                    #calculate initial value of label # tx, ty, bx, by
                    bx = int(((wx*1280)+(ax*1280*2))/2)   
                    by = int(((wy*1280)+(ay*1280*2))/2)
                    tx = int(((ax*1280)*2) - bx)
                    ty = int(((ay*1280)*2) - by)

                    print("labels box", tx, ty, bx, by)
                    
                    np_img = np.array(img)[ty:by, tx:bx, :]
                    size = np_img.shape[:2]
                        
                    df.loc[object_index-1, 'pred_cls'] = str(int(instance['label'][object_index-1][0]))

                    # crop objects from full heatmap
                    fig = plt.figure(figsize=(10,10))
                    plt.subplot(4,n_p+3,object_index+1) 
                    
                    plt.title('id ' + str(object_index - 1) + ', tc: ' + str(int(instance['label'][object_index-1][0])) + ', pc: ' + 'BG')
                    df.loc[object_index-1, 'org_cls'] = str(int(instance['label'][object_index-1][0]))

                    plt.imshow(np_img)
                    plot_grid(size)
                    df.loc[object_index-1, 'binary_img'] = self.image_to_binary(plt, instance, counter=1)
                    df.loc[object_index-1, 'outcome'] = str('FN')
                    df.loc[object_index-1, 'conf'] = 'NaN'
                    df.loc[object_index-1, 'obj'] = 'NaN'
                    for clss in range(0,n_c):    
                        df.loc[object_index-1, 'heatmap_'+str(clss)] = self.image_to_binary(plt, instance, counter=2)
            plt.show()
            df.loc[object_index-1, 'classes']='{0,1,2,3,4,5,6,7,8}'
            display(df)
            return df
    
