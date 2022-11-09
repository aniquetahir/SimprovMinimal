# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os.path

import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
import copy

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument('--deepness', type=int, default=6)
parser.add_argument('--num_confirmations', type=int, default=10)
parser.add_argument('--filter_ratio', type=float, default=0.3)
parser.add_argument('--dropout', type=float, default=0.5)
flags = parser.parse_args()

GLOBAL_DROPOUT_RATE = flags.dropout

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

final_train_accs = []
final_test_accs = []


def self_distill(trained_model, train_envs, test_envs, steps, lr, deepness=6, num_confirmations=10, filter_ratio=0.3, restart=0):
    num_classes = 2.
    random_chance_accuracy = 1./num_classes
    # define the current model
    current_model = trained_model
    for d in range(deepness):
        pred_history = []
        pred_logits_history = []
        x = [y for x in test_envs for y in x['images']]
        y = [y for x in test_envs for y in x['labels']]
        x = torch.stack(x)
        y = torch.stack(y)

        # TODO incorporate test data

        # define the distilled model
        distilled_model = MLP().to(device)
        optimizer = optim.Adam(distilled_model.parameters(), lr=lr)

        # pseudo label the test data
        for i in range(num_confirmations):
            with torch.no_grad():
                pred_logits = current_model(x).detach()
                pred_history.append((pred_logits > 0.).float())
                pred_logits_history.append((pred_logits))

        # filter the top confidence samples from the test data
        pred_history = torch.stack(pred_history)
        pred_logits_history = torch.stack(pred_logits_history)
        pred_variation = pred_history.var(axis=0)

        mean_prediction = torch.mean(pred_logits_history, dim = 0)
        modal_prediction = torch.mode(pred_history, dim=0).values

        hci = torch.sort(pred_variation.flatten()).indices
        num_filtered_samples = int(len(hci) * filter_ratio)

        new_x = x[hci[:num_filtered_samples]]
        new_y = modal_prediction[hci[:num_filtered_samples]]
        new_gt_y = y[hci[:num_filtered_samples]]

        # train on the top confidence samples
        previous_model_state = copy.deepcopy(distilled_model.state_dict())
        previous_model_rchance_diff = 0.

        for step in range(steps):

            logits = distilled_model(new_x)
            nll = mean_nll(logits, new_y)
            optimizer.zero_grad()
            nll.backward()
            optimizer.step()
            if step % 10 == 0:
                # keep a copy of the models parameters
                # current_model_state = copy.deepcopy(distilled_model.state_dict())
                # get the training env accuracy for the current model
                distill_train_accuracy_hist = []
                for env in train_envs:
                    xtr = env['images']
                    ytr = env['labels']
                    with torch.no_grad():
                        distill_train_accuracy_hist.append(abs(mean_accuracy(distilled_model(xtr), ytr).item()-random_chance_accuracy))
                distill_train_accuracy_diff = np.max(distill_train_accuracy_hist)
                print(distill_train_accuracy_diff, end=' ')
                # compare the current model with the previous model on the training data
                # if the current model is near random chance
                if distill_train_accuracy_diff < previous_model_rchance_diff:
                    # Revert the model
                    distilled_model.load_state_dict(previous_model_state)
                    # decrease the learning rate
                    for group in optimizer.param_groups:
                        # pass
                        group['lr'] = group['lr'] * 0.95
                else:
                    for group in optimizer.param_groups:
                        # According to observation, learning seldom goes in the correct direction once it starts to fail
                        # pass
                        group['lr'] = group['lr'] * 1.05

                previous_model_rchance_diff = distill_train_accuracy_diff
                previous_model_state = copy.deepcopy(distilled_model.state_dict())
            if step % 300 == 0:
                print('')
                # print('=' * 10)
                with torch.no_grad():
                    distilled_model.eval()
                    print(f'--> Test Accuracy: {mean_accuracy(distilled_model(x).detach(), y)}')
                    distilled_model.train()
                    # How does the distilled model perform on the training data
                    for i, env in enumerate(train_envs):
                        xtr = env['images']
                        ytr = env['labels']
                        # print(f'--> Train{i} Accuracy: {mean_accuracy(distilled_model(xtr), ytr)}')


        # update the current model
        current_model = distilled_model
        f_acc = float(mean_accuracy(distilled_model(x).detach(), y))
        print(f'Final Test Accuracy: {f_acc}')
        print('=' * 20)
        pass


    # Get final rchance diff
    current_model.eval()
    current_train_accuracy_hist = []
    for env in train_envs:
        xtr = env['images']
        ytr = env['labels']
        with torch.no_grad():
            current_train_accuracy_hist.append(
                abs(mean_accuracy(current_model(xtr), ytr).item() - random_chance_accuracy))
    current_train_accuracy_diff = np.max(current_train_accuracy_hist)

    # Save the model
    filename = f"simprov_cmnist_acc_{f_acc}_deep_{flags.deepness}_conf_{flags.num_confirmations}_ratio_{flags.filter_ratio}_dropout_{flags.dropout}_r{restart}.pth"
    filename = os.path.join('saves', filename)
    torch.save(current_model, filename)
    with open('history.csv', 'a') as history_file:
        history_file.write(filename + f",{current_train_accuracy_diff}\n")
    return current_model


for restart in range(flags.n_restarts):
    print("Restart", restart)

    # Load MNIST, make train/val splits, and shuffle train set examples

    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())


    # Build environments

    def make_environment(images, labels, e):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def torch_xor(a, b):
            return (a-b).abs() # Assumes both inputs are either 0 or 1
        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
        return {
            'images': (images.float() / 255.).to(device),
            'labels': labels[:, None].to(device)
        }

    envs = [
        make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
        make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
        make_environment(mnist_val[0], mnist_val[1], 0.9)
    ]


    train_envs = [envs[0], envs[1]]
    test_envs = [envs[2]]


    # Define and instantiate the model
    # TODO Add dropout to the model to try out the uncertainty idea
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            if flags.grayscale_model:
                lin1 = nn.Linear(14 * 14, flags.hidden_dim)
            else:
                lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
            lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
            lin3 = nn.Linear(flags.hidden_dim, 1)
            for lin in [lin1, lin2, lin3]:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            self._main = nn.Sequential(lin1, nn.ReLU(True), nn.Dropout(GLOBAL_DROPOUT_RATE), lin2, nn.ReLU(True),
                                       nn.Dropout(GLOBAL_DROPOUT_RATE), lin3)

        def forward(self, input):
            if flags.grayscale_model:
                out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
            else:
                out = input.view(input.shape[0], 2 * 14 * 14)
            out = self._main(out)
            return out


    mlp = MLP().to(device)
    mlp_final = MLP().to(device)


    # Define loss function helpers

    def mean_nll(logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)


    def mean_accuracy(logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()


    def penalty(logits, y):
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss = mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)


    # Train loop

    def pretty_print(*values):
        col_width = 13

        def format_val(v):
            if not isinstance(v, str):
                v = np.array2string(v, precision=5, floatmode='fixed')
            return v.ljust(col_width)

        str_values = [format_val(v) for v in values]
        print("   ".join(str_values))


    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)
    optimizer_final = optim.Adam(mlp_final.parameters(), lr=flags.lr)

    pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

    # TODO find the difference between predicitons over different dropout inferences
    for step in range(flags.steps):
        for env in envs:
            logits = mlp(env['images'])
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            env['penalty'] = penalty(logits, env['labels'])

        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
        train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
        train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

        weight_norm = torch.tensor(0.).to(device)
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        penalty_weight = (flags.penalty_weight
                          if step >= flags.penalty_anneal_iters else 1.0)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_acc = envs[2]['acc']
        if step % 100 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy()
            )

    # Now that the irm has been trained, we can use the confident inferences to pseudo label
    self_distill(mlp, train_envs, test_envs, flags.steps*6, flags.lr * 0.01,
                 num_confirmations=flags.num_confirmations, deepness=flags.deepness, filter_ratio=flags.filter_ratio,
                 restart=restart)


    # final_train_accs.append(train_acc.detach().cpu().numpy())
    # final_test_accs.append(test_acc.detach().cpu().numpy())
    # print('Final train acc (mean/std across restarts so far):')
    # print(np.mean(final_train_accs), np.std(final_train_accs))
    # print('Final test acc (mean/std across restarts so far):')
    # print(np.mean(final_test_accs), np.std(final_test_accs))
