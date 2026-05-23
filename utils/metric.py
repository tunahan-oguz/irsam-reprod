import numpy as np
import torch.nn as nn
import torch
from skimage import measure
import numpy


class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass, bins):  # bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)

        recall = self.tp_arr / (self.pos_arr + 0.001)
        precision = self.tp_arr / (self.class_pos + 0.001)

        return tp_rates, fp_rates, recall, precision

    def reset(self):
        self.tp_arr = np.zeros([11])
        self.pos_arr = np.zeros([11])
        self.fp_arr = np.zeros([11])
        self.neg_arr = np.zeros([11])
        self.class_pos = np.zeros([11])


class PD_FA():
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)
        self.FA_denominator = 0.0

    def update(self, preds, labels):
        # preds: [B, C, H, W] or [B, H, W], labels: [B, C, H, W] or [B, H, W]
        if len(preds.shape) == 4:
            preds = preds.squeeze(1)
        if len(labels.shape) == 4:
            labels = labels.squeeze(1)

        B, H, W = preds.shape
        self.FA_denominator += B * H * W

        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        for b in range(B):
            pred_sample = preds_np[b]
            label_sample = labels_np[b]

            for iBin in range(self.bins + 1):
                score_thresh = iBin * (255 / self.bins)
                predits = (pred_sample > score_thresh).astype('int64')
                labelss = label_sample.astype('int64')

                image = measure.label(predits, connectivity=2)
                coord_image = measure.regionprops(image)
                label = measure.label(labelss, connectivity=2)
                coord_label = measure.regionprops(label)

                self.target[iBin] += len(coord_label)
                image_area_total = []
                image_area_match = []
                distance_match = []

                for K in range(len(coord_image)):
                    area_image = np.array(coord_image[K].area)
                    image_area_total.append(area_image)

                for i in range(len(coord_label)):
                    centroid_label = np.array(list(coord_label[i].centroid))
                    for m in range(len(coord_image)):
                        centroid_image = np.array(list(coord_image[m].centroid))
                        distance = np.linalg.norm(centroid_image - centroid_label)
                        area_image = np.array(coord_image[m].area)
                        if distance < 3:
                            distance_match.append(distance)
                            image_area_match.append(area_image)
                            del coord_image[m]
                            break

                dismatch = [x for x in image_area_total if x not in image_area_match]
                self.FA[iBin] += np.sum(dismatch)
                self.PD[iBin] += len(distance_match)

    def get(self, img_num):
        denom = self.FA_denominator if self.FA_denominator > 0 else (512 * 512) * img_num
        Final_FA = self.FA / denom
        Final_PD = self.PD / (self.target + 1e-8)

        return Final_FA, Final_PD

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])
        self.target = np.zeros(self.bins + 1)


class mIoU():

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * (predict == target)

    tp = intersection.sum()
    fp = (predict * (predict != target)).sum()
    tn = ((1 - predict) * (predict == target)).sum()
    fn = ((predict != target) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos = tp + fp

    return tp, pos, fp, neg, class_pos


def batch_pix_accuracy(output, target):
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float()) * ((target > 0)).float()).sum()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _ = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union
