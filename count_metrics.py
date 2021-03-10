import sklearn.metrics as metric
from PIL import Image
import numpy as np
import os
import logging
logging.basicConfig(filename='result_oa.log', level=logging.INFO)

image_list = os.listdir('network_result_1D')

OA=np.array([0])
F1=np.array([0, 0, 0, 0, 0])
PRECISION=np.array([0, 0, 0, 0, 0])
RECALL=np.array([0, 0, 0, 0, 0])

"test onstride 384_72"
for name in image_list:
    print(name)
    logging.info('name:{}'.format(name))
    prediction = Image.open('network_result_1D/' + name)
    prediction = np.array(prediction)
    prediction = np.reshape(prediction, (6000*6000))
    label = Image.open('complete_size_label' + name)
    label = np.array(label)
    label = np.reshape(label, (6000*6000))

    oa = metric.accuracy_score(label, prediction)
    print(oa)
    logging.info('oa:{}'.format(oa))
    OA = OA + oa

    f1 = metric.f1_score(label, prediction, average=None)
    print(f1)
    logging.info('f1:{}'.format(f1))
    f1 = f1[0:5]
    F1 = F1 + f1

    precision = metric.precision_score(label, prediction, average=None)
    print(precision)
    logging.info('precision:{}'.format(precision))
    precision = precision[0:5]
    PRECISION = PRECISION + precision

    recall = metric.recall_score(label, prediction, average=None)
    print(recall)
    logging.info('recall:{}'.format(recall))
    recall = recall[0:5]
    RECALL = RECALL + recall
    "test onstride 384_72"

OA = OA / 14
F1 = F1 / 14
PRECISION = PRECISION/14
RECALL = RECALL/14
OA_average = np.average(OA)
F1_average = np.average(F1)
PRECISION_average = np.average(PRECISION)
RECALL_average = np.average(RECALL)

print('OA:', OA_average, OA)
print('F1:', F1_average, F1)
print('precision:', PRECISION_average, PRECISION)
print('recall:', RECALL_average, RECALL)
logging.info('OA:{}{}'.format(OA_average, OA)+'\n'+'F1:{}{}'.format(F1_average, F1)+'\n'+'precision:{}{}'.format(PRECISION_average, PRECISION)+'\n'+'recall:{}{}'.format(RECALL_average, RECALL))