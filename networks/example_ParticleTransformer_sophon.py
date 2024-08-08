import numpy as np
import awkward as ak
import torch
import tqdm
import time
from collections import defaultdict, Counter

import sklearn.metrics as m
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

from weaver.utils.logger import _logger
from weaver.utils.nn.tools import (
    _flatten_preds,
    _flatten_label,
    _concat
)
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger

'''
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py
'''

class ParticleTransformerSophonWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.export_embed = kwargs.pop('export_embed', False)
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        # return self.mod(features, v=lorentz_vectors, mask=mask) # not using the default foward implementation. Should add emport_embed flag

        x, padding_mask = self.mod._forward_encoder(features, v=lorentz_vectors, mask=mask)

        with torch.cuda.amp.autocast(enabled=self.mod.use_amp):
            x_cls = self.mod._forward_aggregator(x, padding_mask)
            if self.mod.fc is None:
                return x_cls
            # fc
            output = self.mod.fc(x_cls)
            if self.mod.for_inference:
                output = torch.softmax(output, dim=1)
            if self.export_embed:
                return torch.cat([output, x_cls], dim=1)
            else:
                return output


def get_model(data_config, **kwargs):

    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=None,
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=True,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerSophonWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()


def get_train_fn(data_config, **kwargs):
    return train_classification_sophon


def get_evaluate_fn(data_config, **kwargs):
    return evaluate_classification_sophon


# Customized training and evaluation functions for Sophon
# functions are adapted from https://github.com/hqucms/weaver-core/blob/main/weaver/utils/nn/tools.py

def train_classification_sophon(
        model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None,
        tb_helper=None, extra_args=None):
    model.train()

    data_config = train_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long().to(dev) # label is obtained from inputs
            entry_count += label.shape[0]
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                model_output = model(*inputs)
                logits, label, _ = _flatten_preds(model_output, label=label, mask=None)
                loss = loss_func(logits, label)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            _, preds = logits.max(1)
            loss = loss.item()

            num_examples = label.shape[0]
            label_counter.update(label.numpy(force=True))
            num_batches += 1
            count += num_examples
            correct = (preds == label).sum().item()
            total_loss += loss
            total_correct += correct

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgAcc: %.5f' % (total_loss / num_batches, total_correct / count))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))
    _logger.info('Max CUDA memory: %.1f MB' % (torch.cuda.max_memory_allocated(dev) / 1024.**2,))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
        ])

        # # customization: store hyperparameters
        # convert_to_str = lambda x: str(x) if not isinstance(x, (int, float, bool)) else x
        # if epoch == 0:
        #     tb_helper.writer.add_hparams({k: convert_to_str(v) for k, v in model.kwargs.items()}, {})

        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_classification_sophon(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                            eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                            tb_helper=None, extra_args=None):
    model.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                # X, y: torch.Tensor; Z: ak.Array
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].long().to(dev)
                entry_count += label.shape[0]
                model_output = model(*inputs)
                logits, label, _ = _flatten_preds(model_output, label=label, mask=None)
                scores.append(torch.softmax(logits.float(), dim=1).numpy(force=True))

                for k, v in y.items():
                    labels[k].append(_flatten_label(v, None).numpy(force=True))
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v)

                num_examples = label.shape[0]
                label_counter.update(label.numpy(force=True))

                _, preds = logits.max(1)
                loss = 0 if loss_func is None else loss_func(logits, label).item()

                num_batches += 1
                count += num_examples
                correct = (preds == label).sum().item()
                total_loss += loss * num_examples
                total_correct += correct

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count)})

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("Acc/%s (epoch)" % tb_mode, total_correct / count, epoch),
        ])

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}

    # customized evaluation: making ROC curves for tensorboard monitoring
    if tb_helper:
        scores_dict = {
            'Xbb': scores[:, 0],
            'Xcc': scores[:, 1],
            'QCD': np.sum(scores[:, 161:188], axis=1), # sum of the last 27 scores to form the QCD score
        }
        flag_dict = {
            'Xbb': labels['truth_label'] == 0,
            'Xcc': labels['truth_label'] == 1,
            'QCD': (labels['truth_label'] >= 161) & (labels['truth_label'] < 188),
        }
        comp_list = [('Xbb', 'QCD'), ('Xcc', 'QCD'), ('Xcc', 'Xbb')] # ROC curves for A vs B
        bkgrej = {}

        f, ax = plt.subplots(figsize=(5, 5))
        ax.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), linestyle='--', color='gray', label='Random guess')

        for name_sig, name_bkg in comp_list:
            discr = scores_dict[name_sig] / (scores_dict[name_sig] + scores_dict[name_bkg])
            discr_sig, discr_bkg = discr[flag_dict[name_sig]], discr[flag_dict[name_bkg]]
            fpr, tpr, _ = m.roc_curve(
                np.concatenate([np.ones_like(discr_sig), np.zeros_like(discr_bkg)]),
                np.concatenate([discr_sig, discr_bkg])
            )
            ax.plot(tpr, fpr, label='%s vs %s (AUC=%.4f)' % (name_sig, name_bkg, m.auc(fpr, tpr)))
            bkgrej[(name_sig, name_bkg)] = np.interp(0.3, tpr, 1. / np.maximum(fpr, 1e-10)) # bkgrej at eff_sig=30%
        ax.legend()
        ax.set_xlabel('True positive rate (signal eff.)', ha='right', x=1.0); ax.set_ylabel('False positive rate (BKG eff.)', ha='right', y=1.0)
        ax.set_xlim(0, 1); ax.set_ylim(1e-4, 1), ax.set_yscale('log')

        # write ROC curve figure
        tb_helper.writer.add_figure('ROC/%s/epoch%s' % (tb_mode, str(epoch).zfill(4)), f)

        # write bkgrej values
        for name_sig, name_bkg in comp_list:
            tb_helper.write_scalars([
                ('BkgRej_%s_vs_%s/%s (epoch)' % (name_sig, name_bkg, tb_mode), bkgrej[(name_sig, name_bkg)], epoch),
            ])


    if for_training:
        return total_correct / count
    else:
        # convert 2D labels/scores
        if len(scores) != entry_count:
            if len(labels_counts):
                labels_counts = np.concatenate(labels_counts)
                scores = ak.unflatten(scores, labels_counts)
                for k, v in labels.items():
                    labels[k] = ak.unflatten(v, labels_counts)
            else:
                assert (count % entry_count == 0)
                scores = scores.reshape((entry_count, int(count / entry_count), -1)).transpose((1, 2))
                for k, v in labels.items():
                    labels[k] = v.reshape((entry_count, -1))
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_correct / count, scores, labels, observers
