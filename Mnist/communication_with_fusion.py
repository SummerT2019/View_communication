from __future__ import print_function
from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim
import torch.nn.init as init
import sklearn.metrics as metrics
import numpy as np
import Mnist.write_data_mnist as write_tool
from Mnist.read_data_validation import MNISTValidation
import Mnist.config_mnist as config

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class SharedAndSpecificLoss(nn.Module):
    def __init__(self, ):
        super(SharedAndSpecificLoss, self).__init__()

    # Should be orthogonal
    @staticmethod
    def orthogonal_loss(shared, specific):
        shared = torch.sigmod(shared)
        specific = torch.sigmod(specific)
        shared = F.normalize(shared, p=2, dim=1)
        specific = F.normalize(specific, p=2, dim=1)
        correlation_matrix = shared.mul(specific)
        cost = correlation_matrix.mean()
        cost = F.relu(cost)
        return cost

    # should be big
    @staticmethod
    def dot_product_similarity(shared_1, shared_2):
        assert (shared_1.dim() == 2)
        assert (shared_2.dim() == 2)
        num_of_samples = shared_1.size(0)
        shared_1 = torch.sigmoid(shared_1)
        shared_2 = torch.sigmoid(shared_2)
        # Dot product
        match_map = torch.bmm(shared_1.view(num_of_samples, 1, -1), shared_2.view(num_of_samples, -1, 1))
        mean = match_map.mean()
        return mean

    def forward(self, level_output, representation_output, classification_output, target):
        # level_output = view1_specific, view2_specific, view1_shared, view2_shared
        # Similarity Loss
        similarity_loss = 0.0
        similarity_loss_list = [[] for i in range(len(level_output))]
        for i in range(len(level_output)):
            similarity_loss_list[i] = - self.dot_product_similarity(level_output[i][2], level_output[i][3])

        for i in range(len(similarity_loss_list)):
            similarity_loss += similarity_loss_list[i]

        # orthogonal restrict
        orthogonal_loss = 0.0
        orthogonal_loss_list = [[] for i in range(len(level_output) * 2)]
        for i in range(len(level_output)):
            orthogonal_loss_list[i * 2] = self.orthogonal_loss(level_output[i][0], level_output[i][2])  # view1
            orthogonal_loss_list[i * 2 + 1] = self.orthogonal_loss(level_output[i][1], level_output[i][3])  # view2

        for i in range(len(orthogonal_loss_list)):
            orthogonal_loss += orthogonal_loss_list[i]

        # Representation loss
        view1_specific, view2_specific, view1_shared, view2_shared = representation_output
        similarity_loss += self.dot_product_similarity(view1_shared, view2_shared)
        orthogonal_loss += self.orthogonal_loss(view1_specific, view1_shared)
        orthogonal_loss += self.orthogonal_loss(view2_specific, view2_shared)
        # Classification Loss
        classification_loss = F.cross_entropy(classification_output, target)

        loss = orthogonal_loss * 0.2 + similarity_loss * 0.5 + classification_loss
        return loss


# generate index for sub modules
class AttrProxy(object):
    """Translates index lookups into attribute lookups."""

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, index):
        return getattr(self.module, self.prefix + str(index))


# One round Communication
class ResCommunicationBlock(nn.Module):
    def __init__(self, view_size=[64, 64], n_units=[128, 64], feature_size=64):
        super(ResCommunicationBlock, self).__init__()
        # View1
        self.shared1_l1 = nn.Linear(view_size[0], n_units[0])
        self.shared1_l2 = nn.Linear(n_units[0], n_units[1])
        self.shared1_l3 = nn.Linear(n_units[1], feature_size)

        self.specific1_l1 = nn.Linear(view_size[0], n_units[0])
        self.specific1_l2 = nn.Linear(n_units[0], n_units[1])
        self.specific1_l3 = nn.Linear(n_units[1], feature_size)

        # View2
        self.shared2_l1 = nn.Linear(view_size[1], n_units[0])
        self.shared2_l2 = nn.Linear(n_units[0], n_units[1])
        self.shared2_l3 = nn.Linear(n_units[1], feature_size)

        self.specific2_l1 = nn.Linear(view_size[1], n_units[0])
        self.specific2_l2 = nn.Linear(n_units[0], n_units[1])
        self.specific2_l3 = nn.Linear(n_units[1], feature_size)

        # Fusion
        # Fusion View1 + Specific2
        self.fusion1_l1 = nn.Linear(view_size[0] + feature_size, n_units[0])
        self.fusion1_l2 = nn.Linear(n_units[0], n_units[1])
        self.fusion1_l3 = nn.Linear(n_units[1], view_size[0])

        # Fusion View2 + Specific1
        self.fusion2_l1 = nn.Linear(view_size[1] + feature_size, n_units[0])
        self.fusion2_l2 = nn.Linear(n_units[0], n_units[1])
        self.fusion2_l3 = nn.Linear(n_units[1], view_size[1])

    def init_params(self):
        # view 1
        init.kaiming_normal_(self.shared1_l1.weight)
        init.kaiming_normal_(self.shared1_l2.weight)
        init.kaiming_normal_(self.shared1_l3.weight)

        init.kaiming_normal_(self.specific1_l1.weight)
        init.kaiming_normal_(self.specific1_l2.weight)
        init.kaiming_normal_(self.specific1_l3.weight)

        # view 2
        init.kaiming_normal_(self.shared2_l1.weight)
        init.kaiming_normal_(self.shared2_l2.weight)
        init.kaiming_normal_(self.shared2_l3.weight)

        init.kaiming_normal_(self.specific2_l1.weight)
        init.kaiming_normal_(self.specific2_l2.weight)
        init.kaiming_normal_(self.specific2_l3.weight)

        # Fusion
        init.kaiming_normal_(self.fusion1_l1.weight)
        init.kaiming_normal_(self.fusion1_l2.weight)
        init.kaiming_normal_(self.fusion1_l3.weight)

        init.kaiming_normal_(self.fusion2_l1.weight)
        init.kaiming_normal_(self.fusion2_l2.weight)
        init.kaiming_normal_(self.fusion2_l3.weight)

    def forward(self, view1_input, view2_input):
        # View1
        view1_specific = F.relu(self.specific1_l1(view1_input))
        view1_specific = F.relu(self.specific1_l2(view1_specific))
        view1_specific = F.relu(self.specific1_l3(view1_specific))

        view1_shared = F.relu(self.shared1_l1(view1_input))
        view1_shared = F.relu(self.shared1_l2(view1_shared))
        view1_shared = F.relu(self.shared1_l3(view1_shared))

        # View2
        view2_specific = F.relu(self.specific2_l1(view2_input))
        view2_specific = F.relu(self.specific2_l2(view2_specific))
        view2_specific = F.relu(self.specific2_l3(view2_specific))

        view2_shared = F.relu(self.shared2_l1(view2_input))
        view2_shared = F.relu(self.shared2_l2(view2_shared))
        view2_shared = F.relu(self.shared2_l3(view2_shared))

        # Fusion
        fusion1_input = torch.cat([view1_input, view2_specific], dim=1)
        fusion2_input = torch.cat([view2_input, view1_specific], dim=1)

        # Fusion View1
        fusion1_output = F.relu(self.fusion1_l1(fusion1_input))
        fusion1_output = F.relu(self.fusion1_l2(fusion1_output))
        fusion1_output = F.relu(self.fusion1_l3(fusion1_output))

        # Fusion View2
        fusion2_output = F.relu(self.fusion2_l1(fusion2_input))
        fusion2_output = F.relu(self.fusion2_l2(fusion2_output))
        fusion2_output = F.relu(self.fusion2_l3(fusion2_output))

        # Fusion
        view1_new = view1_input + fusion1_output
        view2_new = view2_input + fusion2_output

        return view1_new, view2_new, view1_specific, view2_specific, view1_shared, view2_shared


# whole module
class MultipleRoundsCommunication(nn.Module):
    def __init__(self, level_num=3, original_view_size=[3000, 1840], view_size=[64, 64], feature_size=64,
                 n_units=[128, 64], c_n_units=[64, 32], class_num=2):

        super(MultipleRoundsCommunication, self).__init__()

        # View1 Input
        self.input1_l1 = nn.Linear(original_view_size[0], n_units[0])
        self.input1_l2 = nn.Linear(n_units[0], n_units[1])
        self.input1_l3 = nn.Linear(n_units[1], view_size[0])

        # View2 Input
        self.input2_l1 = nn.Linear(original_view_size[1], n_units[0])
        self.input2_l2 = nn.Linear(n_units[0], n_units[1])
        self.input2_l3 = nn.Linear(n_units[1], view_size[1])

        # Communication
        self.level_num = level_num
        for i_th in range(self.level_num):
            basic_model = ResCommunicationBlock(view_size=[view_size[0], view_size[1]], n_units=[128, 64],
                                                feature_size=feature_size)
            self.add_module('level_' + str(i_th), basic_model)
        self.levels = AttrProxy(self, 'level_')

        # Representation learning network
        # View1
        self.shared1_l1 = nn.Linear(view_size[0], n_units[0])
        self.shared1_l2 = nn.Linear(n_units[0], n_units[1])
        self.shared1_l3 = nn.Linear(n_units[1], feature_size)

        self.specific1_l1 = nn.Linear(view_size[0], n_units[0])
        self.specific1_l2 = nn.Linear(n_units[0], n_units[1])
        self.specific1_l3 = nn.Linear(n_units[1], feature_size)

        # View2
        self.shared2_l1 = nn.Linear(view_size[1], n_units[0])
        self.shared2_l2 = nn.Linear(n_units[0], n_units[1])
        self.shared2_l3 = nn.Linear(n_units[1], feature_size)

        self.specific2_l1 = nn.Linear(view_size[1], n_units[0])
        self.specific2_l2 = nn.Linear(n_units[0], n_units[1])
        self.specific2_l3 = nn.Linear(n_units[1], feature_size)

        # Classification
        self.classification_l1 = nn.Linear(feature_size*3, c_n_units[0])
        self.classification_l2 = nn.Linear(c_n_units[0], c_n_units[1])
        self.classification_l3 = nn.Linear(c_n_units[1], class_num)

    def init_params(self):

        # Input init
        init.kaiming_normal_(self.input1_l1.weight)
        init.kaiming_normal_(self.input1_l2.weight)
        init.kaiming_normal_(self.input1_l3.weight)

        init.kaiming_normal_(self.input2_l1.weight)
        init.kaiming_normal_(self.input2_l2.weight)
        init.kaiming_normal_(self.input2_l3.weight)

        # init module
        for i_th in range(self.level_num):
            name = 'level_' + str(i_th)
            level = self._modules[name]
            level.init_params()

        # Representation init
        init.kaiming_normal_(self.shared1_l1.weight)
        init.kaiming_normal_(self.shared1_l2.weight)
        init.kaiming_normal_(self.shared1_l3.weight)

        init.kaiming_normal_(self.shared2_l1.weight)
        init.kaiming_normal_(self.shared2_l2.weight)
        init.kaiming_normal_(self.shared2_l3.weight)

        init.kaiming_normal_(self.specific1_l1.weight)
        init.kaiming_normal_(self.specific1_l2.weight)
        init.kaiming_normal_(self.specific1_l3.weight)

        init.kaiming_normal_(self.specific2_l1.weight)
        init.kaiming_normal_(self.specific2_l2.weight)
        init.kaiming_normal_(self.specific2_l3.weight)

        # Classification init
        init.kaiming_normal_(self.classification_l1.weight)
        init.kaiming_normal_(self.classification_l2.weight)
        init.kaiming_normal_(self.classification_l3.weight)

    def forward(self, original_view1_input, original_view2_input):
        # Input View1
        input1_output = F.relu(self.input1_l1(original_view1_input))
        input1_output = F.relu(self.input1_l2(input1_output))
        input1 = F.relu(self.input1_l3(input1_output))

        # Input View2
        input2_output = F.relu(self.input2_l1(original_view2_input))
        input2_output = F.relu(self.input2_l2(input2_output))
        input2 = F.relu(self.input2_l3(input2_output))

        # output list record result for each level
        level_output = [[] for j in range(self.level_num)]
        for i_th in range(self.level_num):
            name = 'level_' + str(i_th)
            # get the module of this level
            level = self._modules[name]
            input1, input2, view1_specific, view2_specific, view1_shared, view2_shared = level(input1, input2)
            level_output[i_th] = [view1_specific, view2_specific, view1_shared, view2_shared]

        # Representation
        # View1
        view1_specific = F.relu(self.specific1_l1(input1))
        view1_specific = F.relu(self.specific1_l2(view1_specific))
        view1_specific = F.relu(self.specific1_l3(view1_specific))

        view1_shared = F.relu(self.shared1_l1(input1))
        view1_shared = F.relu(self.shared1_l2(view1_shared))
        view1_shared = F.relu(self.shared1_l3(view1_shared))

        # View2
        view2_specific = F.relu(self.specific2_l1(input2))
        view2_specific = F.relu(self.specific2_l2(view2_specific))
        view2_specific = F.relu(self.specific2_l3(view2_specific))

        view2_shared = F.relu(self.shared2_l1(input2))
        view2_shared = F.relu(self.shared2_l2(view2_shared))
        view2_shared = F.relu(self.shared2_l3(view2_shared))

        representation_output = [view1_specific, view2_specific, view1_shared, view2_shared]

        common = (view1_shared + view2_shared)/2

        # Shared and specific classification
        classification_input = torch.cat([view1_specific, common, view2_specific], dim=1)
        classification_output = F.relu(self.classification_l1(F.dropout(classification_input)))
        classification_output = F.relu(self.classification_l2(F.dropout(classification_output)))
        classification_output = self.classification_l3(classification_output)

        return level_output, representation_output, classification_output


def main(seed=43):
    # Hyper Parameters
    EPOCH = config.MAX_EPOCH
    BATCH_SIZE = config.BATCH_SIZE
    USE_GPU = config.USE_GPU
    LEVEL_NUM = 3

    # Load data
    write_tool.write_data()
    train_data = MNISTValidation(set_name='train')
    test_data = MNISTValidation(set_name='test')

    # Build Model
    model = MultipleRoundsCommunication(level_num=LEVEL_NUM, original_view_size=[config.VIEW1_SIZE, config.VIEW2_SIZE],
                                        view_size=[64, 64], feature_size=64, n_units=[128, 64], c_n_units=[64, 32],
                                        class_num=config.CLASS_NUM)
    print(model)

    # module init
    model.init_params()

    if USE_GPU:
        model = model.cuda()

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    loss_function = SharedAndSpecificLoss()

    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    print("Training...")

    train_loss_ = []
    train_acc_ = []
    test_acc_ = []
    test_precision_ = []
    test_f1_ = []

    for epoch in range(EPOCH):
        # testing epoch ========================================================================================
        train_total_acc = 0.0
        train_total_loss = 0.0
        train_total = 0.0
        for train_iteration_index, train_data in enumerate(train_loader):
            page_input, link_input, train_labels = train_data
            train_labels = torch.squeeze(train_labels)

            if USE_GPU:
                page_input = Variable(page_input.cuda())
                link_input = Variable(link_input.cuda())
                train_labels = train_labels.type(torch.LongTensor).cuda()
            else:
                page_input = Variable(page_input).type(torch.FloatTensor)
                link_input = Variable(link_input).type(torch.FloatTensor)
                train_labels = Variable(train_labels).type(torch.LongTensor)

            optimizer.zero_grad()
            level_output, representation_output, classification_output = model(page_input, link_input)
            loss = loss_function(level_output=level_output, representation_output=representation_output, classification_output=classification_output,
                                 target=train_labels)
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(classification_output.data, 1)
            train_total_acc += (predicted == train_labels.data).sum()
            train_total += len(train_labels)
            train_total_loss += loss.item()

        train_loss_.append(train_total_loss / train_total)

        if config.USE_GPU:
            train_acc_.append(train_total_acc.cpu().numpy() / train_total)
        else:
            train_acc_.append(train_total_acc.numpy() / train_total)

        # testing epoch ========================================================================================
        test_predict_labels = []
        test_ground_truth = []
        for iteration_index, test_data in enumerate(test_loader):
            test_page_inputs, test_link_inputs, test_labels = test_data
            test_labels = torch.squeeze(test_labels)

            if USE_GPU:
                test_page_inputs = Variable(test_page_inputs.cuda())
                test_link_inputs = Variable(test_link_inputs.cuda())
                test_labels = test_labels.cuda()

                level_output, representation_output, classification_output = \
                    model(test_page_inputs, test_link_inputs)

                _, predicted = torch.max(classification_output.data, 1)
                test_predict_labels.extend(list(predicted.cpu().numpy()))
                test_ground_truth.extend(list(test_labels.data.cpu().numpy()))
            else:
                test_page_inputs = Variable(test_page_inputs).type(torch.FloatTensor)
                test_link_inputs = Variable(test_link_inputs).type(torch.FloatTensor)
                test_labels = Variable(test_labels).type(torch.LongTensor)

                level_output, representation_output, classification_output = \
                    model(test_page_inputs, test_link_inputs)

                _, predicted = torch.max(classification_output.data, 1)
                test_predict_labels.extend(list(predicted.numpy()))
                test_ground_truth.extend(list(test_labels.data.numpy()))

        # calculate acc and f1
        a_acc = metrics.accuracy_score(test_ground_truth, test_predict_labels)
        test_acc_.append(a_acc)
        result = metrics.classification_report(test_ground_truth, test_predict_labels, digits=5, output_dict=True)
        a_precision = result['weighted avg']['precision']
        a_f1 = result['weighted avg']['f1-score']
        test_precision_.append(a_precision)
        test_f1_.append(a_f1)

        print('[Epoch: %3d/%3d] Training Loss: %.3f, Training Acc: %.5f, '
              'Testing Precision: %.5f, Testing Acc: %.5f, Testing F1-score: %.5f'
              % (epoch + 1, EPOCH, train_loss_[epoch], train_acc_[epoch],
                 test_precision_[epoch], test_acc_[epoch], test_f1_[epoch]))

    final_acc = np.mean(np.sort(test_acc_)[-2:])
    final_f1 = np.mean(np.sort(test_f1_)[-2:])
    return final_acc, final_f1


if __name__ == "__main__":
    acc_list = []
    f1_list = []
    for i in range(config.ITERATIONS):
        acc, f1 = main(seed=i)
        acc_list.append(acc)
        f1_list.append(f1)
        print("In this Run, Acc = ", acc, ", F1 = ", f1)
    # Print result
    print("===================== ACC =====================")
    for i in range(len(acc_list)):
        print(acc_list[i])
    print("=================== F1-score ==================")
    for i in range(len(f1_list)):
        print(f1_list[i])
