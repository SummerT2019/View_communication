from __future__ import print_function

from numpy import*
from torch import nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim
import BCPP.write_data_bcpp as write_tool
import BCPP.config_bcpp as config
from BCPP.read_data_validation import bcppValidation_two
from sklearn import metrics
import numpy as np

class SharedAndSpecificLoss(nn.Module):
    def __init__(self, ):
        super(SharedAndSpecificLoss, self).__init__()

    # Should be orthogonal
    @staticmethod
    def orthogonal_loss(shared, specific):
        shared = torch.sigmoid(shared)
        specific = torch.sigmoid(specific)
        shared = F.normalize(shared, p=2, dim=1)
        specific = F.normalize(specific, p=2, dim=1)
        correlation_matrix = shared.mul(specific)
        cost = correlation_matrix.mean()
        cost = F.relu(cost)
        return cost


    # should be big
    # dot product
    @staticmethod
    def dot_product_normalize(shared_1, shared_2):
        assert (shared_1.dim() == 2)
        assert (shared_2.dim() == 2)
        num_of_samples = shared_1.size(0)
        shared_1 = shared_1 - shared_1.mean()
        shared_2 = shared_2 - shared_2.mean()
        shared_1 = F.normalize(shared_1, p=2, dim=1)  # 40*16
        shared_2 = F.normalize(shared_2, p=2, dim=1)

        # Dot product
        match_map = torch.bmm(shared_1.view(num_of_samples, 1, -1), shared_2.view(num_of_samples, -1, 1)) # 40*1*16
        mean = match_map.mean()  # 求矩阵平均值 同一个量綱内 可以度量两个向量的相似度
        return mean

    def forward(self, level1_output, level2_output, classification_output, target):
        level1_view1_specific, level1_view1_shared, level1_view2_specific, level1_view2_shared = level1_output

        level2_view1_specific, level2_view1_shared, level2_view2_specific, level2_view2_shared = level2_output
        # Similarity Loss
        similarity_loss_1 = - self.dot_product_normalize(level1_view1_shared, level1_view2_shared)
        similarity_loss_2 = - self.dot_product_normalize(level2_view1_shared, level2_view2_shared)
        similarity_loss = similarity_loss_1 + similarity_loss_2

        # orthogonal restrict
        orthogonal_loss1 = self.orthogonal_loss(level1_view1_shared, level1_view1_specific)
        orthogonal_loss2 = self.orthogonal_loss(level1_view2_shared, level1_view2_specific)
        orthogonal_loss3 = self.orthogonal_loss(level2_view1_shared, level2_view1_specific)
        orthogonal_loss4 = self.orthogonal_loss(level2_view2_shared, level2_view2_specific)

        orthogonal_loss = orthogonal_loss1 + orthogonal_loss2 + orthogonal_loss3 + orthogonal_loss4

        # Classification Loss
        classification_loss = F.cross_entropy(classification_output, target)

        loss = orthogonal_loss * 0.2 + similarity_loss * 0.2 + classification_loss
        return loss


# net
class SharedAndSpecificClassifier(nn.Module):
    def __init__(self, view_size=[3000, 1840], n_units=[128, 64], out_size=32, c_n_units=[64, 32], n_class=2):
        super(SharedAndSpecificClassifier, self).__init__()

        # LEVEL 1
        # View1
        self.level1_shared1_l1 = nn.Linear(view_size[0], n_units[0])
        self.level1_shared1_l2 = nn.Linear(n_units[0], n_units[1])
        self.level1_shared1_l3 = nn.Linear(n_units[1], out_size)

        self.level1_specific1_l1 = nn.Linear(view_size[0], n_units[0])
        self.level1_specific1_l2 = nn.Linear(n_units[0], n_units[1])
        self.level1_specific1_l3 = nn.Linear(n_units[1], out_size)

        # View2
        self.level1_shared2_l1 = nn.Linear(view_size[1], n_units[0])
        self.level1_shared2_l2 = nn.Linear(n_units[0], n_units[1])
        self.level1_shared2_l3 = nn.Linear(n_units[1], out_size)

        self.level1_specific2_l1 = nn.Linear(view_size[1], n_units[0])
        self.level1_specific2_l2 = nn.Linear(n_units[0], n_units[1])
        self.level1_specific2_l3 = nn.Linear(n_units[1], out_size)

        # LEVEL 2
        # View1
        self.level2_shared1_l1 = nn.Linear(view_size[1] + out_size, n_units[0])
        self.level2_shared1_l2 = nn.Linear(n_units[0], n_units[1])
        self.level2_shared1_l3 = nn.Linear(n_units[1], out_size)

        self.level2_specific1_l1 = nn.Linear(view_size[1] + out_size, n_units[0])
        self.level2_specific1_l2 = nn.Linear(n_units[0], n_units[1])
        self.level2_specific1_l3 = nn.Linear(n_units[1], out_size)

        # View2
        self.level2_shared2_l1 = nn.Linear(view_size[0] + out_size, n_units[0])
        self.level2_shared2_l2 = nn.Linear(n_units[0], n_units[1])
        self.level2_shared2_l3 = nn.Linear(n_units[1], out_size)

        self.level2_specific2_l1 = nn.Linear(view_size[0] + out_size, n_units[0])
        self.level2_specific2_l2 = nn.Linear(n_units[0], n_units[1])
        self.level2_specific2_l3 = nn.Linear(n_units[1], out_size)

        # Level 2 fusion
        # Fusion View1 + Specific2
        self.level2_fusion1_l1 = nn.Linear(view_size[0] + 2 * out_size, n_units[0])
        self.level2_fusion1_l2 = nn.Linear(n_units[0], n_units[1])
        self.level2_fusion1_l3 = nn.Linear(n_units[1], out_size)

        # Fusion View2 + Specific1
        self.level2_fusion2_l1 = nn.Linear(view_size[1] + 2 * out_size, n_units[0])
        self.level2_fusion2_l2 = nn.Linear(n_units[0], n_units[1])
        self.level2_fusion2_l3 = nn.Linear(n_units[1], out_size)

        # Classification
        self.classification_l1 = nn.Linear(out_size * 3, c_n_units[0])
        self.classification_l2 = nn.Linear(c_n_units[0], c_n_units[1])
        self.classification_l3 = nn.Linear(c_n_units[1], n_class)

    # init
    def init_params(self):
        # Level 1 view 1
        init.kaiming_normal(self.level1_shared1_l1.weight)
        init.kaiming_normal(self.level1_shared1_l2.weight)
        init.kaiming_normal(self.level1_shared1_l3.weight)

        init.kaiming_normal(self.level1_specific1_l1.weight)
        init.kaiming_normal(self.level1_specific1_l2.weight)
        init.kaiming_normal(self.level1_specific1_l3.weight)
        # Level 1 View 2
        init.kaiming_normal(self.level1_shared2_l1.weight)
        init.kaiming_normal(self.level1_shared2_l2.weight)
        init.kaiming_normal(self.level1_shared2_l3.weight)

        init.kaiming_normal(self.level1_specific2_l1.weight)
        init.kaiming_normal(self.level1_specific2_l2.weight)
        init.kaiming_normal(self.level1_specific2_l3.weight)

        # Level 2

        # Level 2 view 1
        init.kaiming_normal(self.level2_shared1_l1.weight)
        init.kaiming_normal(self.level2_shared1_l2.weight)
        init.kaiming_normal(self.level2_shared1_l3.weight)

        init.kaiming_normal(self.level2_specific1_l1.weight)
        init.kaiming_normal(self.level2_specific1_l2.weight)
        init.kaiming_normal(self.level2_specific1_l3.weight)
        # Level 2 View 2
        init.kaiming_normal(self.level2_shared2_l1.weight)
        init.kaiming_normal(self.level2_shared2_l2.weight)
        init.kaiming_normal(self.level2_shared2_l3.weight)

        init.kaiming_normal(self.level2_specific2_l1.weight)
        init.kaiming_normal(self.level2_specific2_l2.weight)
        init.kaiming_normal(self.level2_specific2_l3.weight)

        # Fusion
        init.kaiming_normal(self.level2_fusion1_l1.weight)
        init.kaiming_normal(self.level2_fusion1_l2.weight)
        init.kaiming_normal(self.level2_fusion1_l3.weight)

        init.kaiming_normal(self.level2_fusion2_l1.weight)
        init.kaiming_normal(self.level2_fusion2_l2.weight)
        init.kaiming_normal(self.level2_fusion2_l3.weight)

        # Classification
        init.kaiming_normal(self.classification_l1.weight)
        init.kaiming_normal(self.classification_l2.weight)
        init.kaiming_normal(self.classification_l3.weight)

    def forward(self, view1_input, view2_input):
        # LEVEL1
        # View1
        level1_view1_specific = F.relu(self.level1_specific1_l1(view1_input))
        level1_view1_specific = F.relu(self.level1_specific1_l2(level1_view1_specific))
        level1_view1_specific = F.relu(self.level1_specific1_l3(level1_view1_specific))

        level1_view1_shared = F.relu(self.level1_shared1_l1(view1_input))
        level1_view1_shared = F.relu(self.level1_shared1_l2(level1_view1_shared))
        level1_view1_shared = F.relu(self.level1_shared1_l3(level1_view1_shared))

        # View2
        level1_view2_specific = F.relu(self.level1_specific2_l1(view2_input))
        level1_view2_specific = F.relu(self.level1_specific2_l2(level1_view2_specific))
        level1_view2_specific = F.relu(self.level1_specific2_l3(level1_view2_specific))

        level1_view2_shared = F.relu(self.level1_shared2_l1(view2_input))
        level1_view2_shared = F.relu(self.level1_shared2_l2(level1_view2_shared))
        level1_view2_shared = F.relu(self.level1_shared2_l3(level1_view2_shared))

        # LEVEL 2
        level2_view1_input = torch.cat([level1_view1_specific, view2_input], dim=1)
        level2_view2_input = torch.cat([level1_view2_specific, view1_input], dim=1)

        # View1
        level2_view1_specific = F.relu(self.level2_specific1_l1(level2_view1_input))
        level2_view1_specific = F.relu(self.level2_specific1_l2(level2_view1_specific))
        level2_view1_specific = F.relu(self.level2_specific1_l3(level2_view1_specific))

        level2_view1_shared = F.relu(self.level2_shared1_l1(level2_view1_input))
        level2_view1_shared = F.relu(self.level2_shared1_l2(level2_view1_shared))
        level2_view1_shared = F.relu(self.level2_shared1_l3(level2_view1_shared))

        # View2
        level2_view2_specific = F.relu(self.level2_specific2_l1(level2_view2_input))
        level2_view2_specific = F.relu(self.level2_specific2_l2(level2_view2_specific))
        level2_view2_specific = F.relu(self.level2_specific2_l3(level2_view2_specific))

        level2_view2_shared = F.relu(self.level2_shared2_l1(level2_view2_input))
        level2_view2_shared = F.relu(self.level2_shared2_l2(level2_view2_shared))
        level2_view2_shared = F.relu(self.level2_shared2_l3(level2_view2_shared))

        # Fusion View1
        level2_fusion1_input = torch.cat([level2_view1_specific, level2_view2_input], dim=1)
        level2_fusion1_output = F.relu(self.level2_fusion1_l1(level2_fusion1_input))
        level2_fusion1_output = F.relu(self.level2_fusion1_l2(level2_fusion1_output))
        level2_fusion1_output = F.relu(self.level2_fusion1_l3(level2_fusion1_output))

        # Fusion View2
        level2_fusion2_input = torch.cat([level2_view2_specific, level2_view1_input], dim=1)
        level2_fusion2_output = F.relu(self.level2_fusion2_l1(level2_fusion2_input))
        level2_fusion2_output = F.relu(self.level2_fusion2_l2(level2_fusion2_output))
        level2_fusion2_output = F.relu(self.level2_fusion2_l3(level2_fusion2_output))

        common = (level2_view1_shared + level2_view2_shared) / 2  # 将level2的部分输出参数取平均后，作为分类层的输入
        # Classification
        classification_input = torch.cat([level2_fusion1_output, common, level2_fusion2_output], dim=1)
        classification_output = F.relu(self.classification_l1(F.dropout(classification_input)))
        classification_output = F.relu(self.classification_l2(F.dropout(classification_output)))
        classification_output = self.classification_l3(classification_output)

        level1_output = [level1_view1_specific, level1_view1_shared, level1_view2_specific, level1_view2_shared]
        level2_output = [level2_view1_specific, level2_view1_shared, level2_view2_specific, level2_view2_shared]
        return level1_output, level2_output, classification_output

#softmax
def softmax_prob(array):
    result = []       # softmax result
    for i in range(array.shape[0]):   # search each row
        array[i] = array[i] - max(array[i])
        result.append(np.exp(array[i]) / np.sum(       # cal softmax
                np.exp(array[i]), axis=0))
    result = np.array(result)  # to array
    result = np.max(result, axis=1)  # get the positive prob
    return result


def main():
    # Hyper Parameters
    EPOCH = 1200
    BATCH_SIZE = 256
    USE_GPU = False
    train_size = config.TRAIN_SIZE

    # Load datasets
    test_size = 1 - train_size
    write_tool.write_bbcp_3to2(train_size=train_size, test_size=test_size, validation_size=0.1, seed=43)
    train_data = bcppValidation_two(set_name='train')
    test_data = bcppValidation_two(set_name='test')
    # Build Model
    model = SharedAndSpecificClassifier(view_size=[config.VIEW1_SIZE_TWO, config.VIEW2_SIZE], n_units=[128, 64],
                                        out_size=32, c_n_units=[64, 64], n_class=config.CLASS_NUM)
    model.init_params()
    print(model)
    if USE_GPU:
        model = model.cuda()

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    loss_function = SharedAndSpecificLoss()

    print("Training...")

    train_loss_ = []

    train_acc_ = []
    test_acc_ = []

    train_f1_ = []
    test_f1_ = []

    # Data Loader for easy mini-batch return in training
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE*10, shuffle=True)
    for epoch in range(EPOCH):

        # training epoch
        total_loss = 0.0
        total_acc = 0.0
        total_f1 = 0.0
        total = 0.0   # sample num
        count = 0.0   # batch num

        for iter, traindata in enumerate(train_loader):
            clinical_exp_input, cnv_input, train_labels = traindata

            train_labels = torch.squeeze(train_labels)

            if USE_GPU:
                clinical_exp_input, cnv_input, train_labels = Variable(clinical_exp_input.cuda()), Variable(
                    cnv_input.cuda()), \
                                                              train_labels.cuda()
            else:
                clinical_exp_input = Variable(clinical_exp_input).type(torch.FloatTensor)
                cnv_input = Variable(cnv_input).type(torch.FloatTensor)
                train_labels = Variable(train_labels).type(torch.LongTensor)

            optimizer.zero_grad()
            level1_output, level2_output, classification_output = model(clinical_exp_input, cnv_input)
            loss = loss_function(level1_output=level1_output, level2_output=level2_output,
                                 classification_output=classification_output, target=train_labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(classification_output.data, 1)

            total_acc += metrics.accuracy_score(train_labels.numpy(), predicted.numpy())
            total_f1 += metrics.f1_score(train_labels.numpy(), predicted.numpy())

            total += len(train_labels)
            count += 1
            total_loss += loss.item()

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc / count)
        train_f1_.append(total_f1 / count)

        # testing epoch ========================================================================================
        total_acc = 0.0
        total_f1 = 0.0
        count = 0.0
        for iter, testdata in enumerate(test_loader):
            test_clinical_exp_inputs, test_cnv_inputs, test_labels = testdata
            test_labels = torch.squeeze(test_labels)

            if USE_GPU:
                test_clinical_exp_inputs, test_cnv_inputs, test_labels = Variable(test_clinical_exp_inputs.cuda()), \
                                                                         Variable(
                                                                             test_cnv_inputs.cuda()), test_labels.cuda()
            else:
                test_clinical_exp_inputs = Variable(test_clinical_exp_inputs).type(torch.FloatTensor)
                test_cnv_inputs = Variable(test_cnv_inputs).type(torch.FloatTensor)
                test_labels = Variable(test_labels).type(torch.LongTensor)

            optimizer.zero_grad()
            level1_output, level2_output, classification_output = model(test_clinical_exp_inputs, test_cnv_inputs)
            loss = loss_function(level1_output=level1_output, level2_output=level2_output,
                                 classification_output=classification_output, target=test_labels)
            loss.backward()
            optimizer.step()

            # calc testing acc
            _, predicted = torch.max(classification_output.data, 1)
            total_acc += metrics.accuracy_score(test_labels.numpy(), predicted.numpy())
            total_f1 += metrics.f1_score(test_labels.numpy(), predicted.numpy())
            count += 1

        test_acc_.append(total_acc / count)
        test_f1_.append(total_f1 / count)

        print('[Epoch: %3d/%3d] Training Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f, Testing F1: %.3f'
              % (epoch+1, EPOCH, train_loss_[epoch], train_acc_[epoch], test_acc_[epoch], test_f1_[epoch]))
    return train_loss_[-1], train_acc_[-1], test_acc_[-1], train_f1_[-1], test_f1_[-1]


if __name__ == "__main__":
    # main()
    train_loss = []
    train_acc = []
    test_acc = []
    train_f1 = []
    test_f1 = []
    MAX_ITER = config.MAX_ITER
    for i in range(MAX_ITER):
        train_loss_, train_prec_, test_prec_, train_acc_, test_acc_, train_auc_, test_auc_, train_f1_, test_f1_ = main()
        train_loss.append(train_loss_)
        train_acc.append(train_acc_)
        test_acc.append(test_acc_)
        train_f1.append(train_f1_)
        test_f1.append(test_f1_)
    print(mean(train_loss), mean(train_acc), mean(test_acc), mean(test_f1))

