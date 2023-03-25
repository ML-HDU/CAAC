from fastai.vision import *


class loss_CAAC(nn.Module):

    def __init__(self, temperature=0.1, reduction='mean', record=True, supervised_flag=True,
                 contrastive_flag=True):
        super(loss_CAAC, self).__init__()
        self.temperature = temperature
        assert reduction in ['none', 'mean', 'sum']
        self.reduction = reduction
        self.record = record
        self.supervised_flag = supervised_flag
        self.supervised_loss = loss_recognition(one_hot=False)

        self.contrastive_flag = contrastive_flag
        self.contrastive_loss = loss_contrastive(temperature=self.temperature, reduction='mean')

    @property
    def last_losses(self):
        return self.losses

    def _supervised_loss_forward(self, outputs_view_1, outputs_view_2, gt_labels, gt_lengths):
        ce_loss_view_1 = self.supervised_loss(outputs_view_1, gt_labels, gt_lengths)
        ce_loss_view_2 = self.supervised_loss(outputs_view_2, gt_labels, gt_lengths)

        ce_loss = (ce_loss_view_1 + ce_loss_view_2) / 2

        return ce_loss

    def _contrastive_loss_forward(self, outputs_view_1, outputs_view_2, gt_labels, gt_lengths):

        def _flatten(sources, lengths):
            return torch.cat([t[:l] for t, l in zip(sources, lengths)])

        def _format(attn_vecs, gt_labels, gt_lengths):
            assert attn_vecs.shape[0] % gt_labels.shape[0] == 0
            iter_size = attn_vecs.shape[0] // gt_labels.shape[0]
            if iter_size > 1:
                gt_labels = gt_labels.repeat(3, 1, 1)
                gt_lengths = gt_lengths.repeat(3)
            flat_attn_vecs = _flatten(attn_vecs, gt_lengths)
            flat_gt_labels = _flatten(gt_labels, gt_lengths)

            return flat_attn_vecs, flat_gt_labels

        con_feature_view_1 = outputs_view_1['con_feature_view_1']
        con_feature_view_2 = outputs_view_2['con_feature_view_2']

        flat_con_feature_view_1, flat_gt_labels_view_1 = _format(con_feature_view_1, gt_labels, gt_lengths)
        flat_con_feature_view_2, flat_gt_labels_view_2 = _format(con_feature_view_2, gt_labels, gt_lengths)

        assert (
                    flat_gt_labels_view_1 == flat_gt_labels_view_2).all(), f'{flat_gt_labels_view_1} != {flat_gt_labels_view_2}'
        gt_labels_contrastive = flat_gt_labels_view_1

        con_features = torch.cat((flat_con_feature_view_2, flat_con_feature_view_1), dim=0)
        normalized_features = F.normalize(con_features, dim=1)

        f1, f2 = torch.split(normalized_features, [gt_lengths.sum(), gt_lengths.sum()], dim=0)
        normalized_con_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        con_loss_intra = self.contrastive_loss(normalized_con_features, gt_labels_contrastive)

        return con_loss_intra

    def forward(self, outputs, *args):
        self.losses = {}
        ce_loss = 0
        con_loss = 0

        is_training = False
        if 'outputs_view_1' in list(outputs.keys()) and 'outputs_view_2' in list(outputs.keys()):
            is_training = True
            outputs_view_1 = outputs['outputs_view_1']
            outputs_view_2 = outputs['outputs_view_2']

            if self.supervised_flag:
                ce_loss = self._supervised_loss_forward(outputs_view_1, outputs_view_2, *args)
                self.losses[f'ce_loss'] = ce_loss

            if self.contrastive_flag:
                con_loss = self._contrastive_loss_forward(outputs_view_1, outputs_view_2, *args)
                self.losses[f'contrastive_loss'] = con_loss

        else:
            ce_loss = self.supervised_loss(outputs, *args)
            self.losses[f'ce_loss'] = ce_loss

        if is_training:
            return outputs['loss_weight']['ce'] * ce_loss + outputs['loss_weight']['sup_con'] * con_loss
        else:
            return ce_loss


class loss_contrastive(nn.Module):
    """ Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf. """

    def __init__(self, temperature=0.01, contrast_mode='all', base_temperature=0.7, reduction='mean'):
        super(loss_contrastive, self).__init__()
        self.temperature = temperature
        print('temperature is {}'.format(self.temperature))
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features, labels=None, mask=None):
        """
        Compute loss for model.

        Args:
            features: hidden vector of shape [bsz, hidden_dim, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i, j} = 1 if sample j
                has the same class as sample i. Can be asymmetric.

        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, hidden_dim, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            # Contrastive Learning
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # Supervised Contrastive Learning
            labels = labels.contiguous().view(-1, 1)        #
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features.")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        if self.reduction == 'mean':
            loss = loss.view(anchor_count, batch_size).mean()
        elif self.reduction == 'sum':
            loss = loss.view(anchor_count, batch_size).sum()
        
        return loss


class loss_recognition(nn.Module):
    def __init__(self, one_hot=True):
        super(loss_recognition, self).__init__()
        self.ce = SoftCrossEntropyLoss() if one_hot else torch.nn.CrossEntropyLoss()

    @property
    def last_losses(self):
        return self.losses

    @staticmethod
    def _flatten(sources, lengths):
        return torch.cat([t[:l] for t, l in zip(sources, lengths)])

    def _ce_loss(self, output, gt_labels, gt_lengths, idx=None, record=True):
        pt_logits = output['logits']

        assert pt_logits.shape[0] % gt_labels.shape[0] == 0
        iter_size = pt_logits.shape[0] // gt_labels.shape[0]
        if iter_size > 1:
            gt_labels = gt_labels.repeat(3, 1, 1)
            gt_lengths = gt_lengths.repeat(3)
        flat_gt_labels = self._flatten(gt_labels, gt_lengths)
        flat_pt_logits = self._flatten(pt_logits, gt_lengths)

        loss = self.ce(flat_pt_logits, flat_gt_labels)

        return loss

    def forward(self, outputs, *args):
        self.losses = {}
        return self._ce_loss(outputs, *args, record=False)


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, softmax=True):
        if softmax:
            log_prob = F.log_softmax(input, dim=-1)
        else:
            log_prob = torch.log(input)
        loss = -(target * log_prob).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss