import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
import torch.distributed as dist
from sklearn.metrics import f1_score, confusion_matrix
import pickle
import pandas as pd

from .build import EVALUATOR_REGISTRY
from dassl.utils import lt_accuracies, lt_accuracies_splitted
from dassl.evaluation.evaluation_ek100_mir import calculate_DCG, calculate_k_counts, calculate_IDCG, calculate_nDCG, calculate_mAP

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        # self._correct = 0
        # self._total = 0
        # self._per_class_res = None
        # self._y_true = []
        # self._y_pred = []
        # self.feat_collection = defaultdict(list)
        # self.narration_ids_collection = []
        self._correct = defaultdict(int)
        self._total = defaultdict(int)
        self._per_class_res = None
        self._y_true = defaultdict(list)
        self._y_pred = defaultdict(list)
        self.feat_collection = defaultdict(dict)
        self.narration_ids_collection = defaultdict(list)
        self._preds_mat = defaultdict(list)

        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            # need to fix
            self._per_class_res = defaultdict(list)

        if cfg.TEST.EPIC_EVAL:
            # need to fix
            self._epic_results = defaultdict(list)

        if self.cfg.TEST.LT_EVAL:
            self.train_counts = kwargs['train_counts']
            print(f'Eval with {self.train_counts} train_counts')
            self.dist_splits = kwargs['dist_splits']

        if self.cfg.TEST.BASE_NOVEL_EVAL:
            self.n_test_classes = kwargs['n_test_classes']
            print(f'Eval with {self.n_test_classes} classes')
            self.dist_splits_novel_base = kwargs['dist_splits_novel_base']

        if self.cfg.TEST.CROSS_DATASET.EVAL:
            self.n_crosseval_classes = kwargs['n_crosseval_classes']
            print(f'Eval with {self.n_crosseval_classes} classes')
            self.dist_crosseval_splits_novel_base = kwargs['dist_crosseval_splits_novel_base']

    def reset(self):
        # self._correct = 0
        # self._total = 0
        # self._y_true = []
        # self._y_pred = []
        # self.feat_collection = defaultdict(list)
        # self.narration_ids_collection = []
        self._correct = defaultdict(int)
        self._total = defaultdict(int)
        self._y_true = defaultdict(list)
        self._y_pred = defaultdict(list)
        self._preds_mat = defaultdict(list)
        self.feat_collection = defaultdict(dict)
        self.narration_ids_collection = defaultdict(list)

        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt, label_type='orig'):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct[label_type] += int(matches.sum().item())
        self._total[label_type] += gt.shape[0]

        self._y_true[label_type].extend(gt.data.cpu().numpy().tolist())
        self._y_pred[label_type].extend(pred.data.cpu().numpy().tolist())

        self._preds_mat[label_type].append(mo.cpu())


        # NOTE: per class accuracy is not sync between GPUs
        # need to fix
        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def process_epic(self, output, narration_ids, label_type='orig' ):
        if self.cfg.TEST.EPIC_EVAL:
            label_type = self.cfg.DATASET.LABEL_TYPE if label_type == 'orig' else label_type
            for item_idx, narration_id in enumerate(narration_ids):
                item = {
                    f'{label_type}_output': output[item_idx],
                    'narration_id': narration_id
                }
                self._epic_results[label_type].append(item)
                # print('ADD epic results')

    def process_retrieval(self, narration_ids, features, label_type='orig'):
        if self.cfg.TEST.RETRIEVAL or self.cfg.TEST.CROSS_DATASET.RETRIEVAL:
            if 'image_feat' not in self.feat_collection[label_type]:
                self.feat_collection[label_type]['image_feat'] = []
            self.narration_ids_collection[label_type].append(narration_ids)
            # self.feat_collection['text_feat'].append(features['text_feat'])
            self.feat_collection[label_type]['image_feat'].append(features['image_feat'])
            if 'visual_ctx_feat' in features:
                if 'visual_ctx_feat' not in self.feat_collection[label_type]:
                    self.feat_collection[label_type]['visual_ctx_feat'] = []
                self.feat_collection[label_type]['visual_ctx_feat'].append(features['visual_ctx_feat'])

    def process_retrieval_text(self, text_features, label_type='orig'):
        if isinstance(text_features, dict):
            for k in text_features.keys():
                if k not in self.feat_collection[label_type]:
                    self.feat_collection[label_type][k] = []
                self.feat_collection[label_type][k].append(text_features[k])
        else:
            if 'text_feat' not in self.feat_collection[label_type]:
                self.feat_collection[label_type]['text_feat'] = []
            self.feat_collection[label_type]['text_feat'].append(text_features)


    def save_epic(self, local_rank, path, label_type='orig' ):
        if self.cfg.TEST.EPIC_EVAL:
            save_path = osp.join(path, f"{local_rank}_{self.cfg.DATASET.LABEL_TYPE}.pkl")
            print('SAVE epic results', save_path, flush=True)
            with open(save_path, 'wb') as f:
                pickle.dump(self._epic_results[label_type], f)

    def synchronize_between_processes(self, label_type='orig'):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self._correct[label_type], self._total[label_type]], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self._correct[label_type] = int(t[0])
        self._total[label_type] = int(t[1])

        _y_true = torch.tensor(self._y_true[label_type], device='cuda')
        _y_true = concat_all_gather(_y_true)
        self._y_true[label_type] = _y_true.cpu().numpy().tolist()

        _y_pred = torch.tensor(self._y_pred[label_type], device='cuda')
        _y_pred = concat_all_gather(_y_pred)
        self._y_pred[label_type] = _y_pred.cpu().numpy().tolist()


    def evaluate(self, split='val', prefix='', accuracy_old=None, label_type='orig', base_novel=True ):
        results = OrderedDict()
        acc = 100.0 * self._correct[label_type] / self._total[label_type]
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true[label_type],
            self._y_pred[label_type],
            average="macro",
            labels=np.unique(self._y_true[label_type])
        )

        # The first value will be returned by trainer.test()
        if label_type != 'orig':
            prefix = f'{prefix}{label_type}/'
        results[f"{prefix}accuracy"] = acc
        results[f"{prefix}error_rate"] = err
        results[f"{prefix}macro_f1"] = macro_f1
        if accuracy_old is not None:
            results['am_acc'] = (acc + accuracy_old) / 2
            if accuracy_old == 0:
                results['hm_acc'] = 0
            else:
                results['hm_acc'] = 2 * acc * accuracy_old / (acc + accuracy_old)

        print(
            "=> result\n"
            f"{prefix}* total: {self._total[label_type]:,}\n"
            f"{prefix}* correct: {self._correct[label_type]:,}\n"
            f"{prefix}* accuracy: {acc:.1f}%\n"
            f"{prefix}* error: {err:.1f}%\n"
            f"{prefix}* macro_f1: {macro_f1:.1f}%"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print(f"{prefix}=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"{prefix}* class: {label} ({classname})\t"
                    f"{prefix}total: {total:,}\t"
                    f"{prefix}correct: {correct:,}\t"
                    f"{prefix}acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"{prefix}* average: {mean_acc:.1f}%")

            results[f"{prefix}perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            # need to fix
            if self._lab2cname is not None:
                sfx = f'_{len(self._lab2cname)}'
                labels_cmat = list(self._lab2cname.keys())
            else:
                sfx = ''
                labels_cmat = None
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true", labels=labels_cmat
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, f"cmat_{split}.pt")
            torch.save(cmat, save_path)
            print(f"{prefix}Confusion matrix is saved to {save_path}")

        if base_novel and self.cfg.TEST.BASE_NOVEL_EVAL or self.cfg.TEST.CROSS_DATASET.BASE_NOVEL_EVAL:
            # output = avg_accs, test_correct_list, test_count_list, test_class_acc, train_counts
            preds = torch.tensor(self._y_pred[label_type])
            preds_full = torch.cat(self._preds_mat[label_type])
            labels = torch.tensor(self._y_true[label_type])
            # print('PREDS', preds[:100], flush=True)
            # print('LABELS', labels[:100], flush=True) #

            if 'cross' in prefix:
                if label_type != 'orig' and label_type not in ['noun', 'verb']:
                    return results
                output = lt_accuracies(preds, labels, self.n_crosseval_classes, self.dist_crosseval_splits_novel_base, label_type=label_type)
                output2 = lt_accuracies_splitted(preds_full, labels, self.dist_crosseval_splits_novel_base, label_type=label_type)
                # def lt_accuracies_splitted(preds, labels, dist_splits, label_type='orig'):
            else:
                if label_type != 'orig' and label_type not in ['noun', 'verb']:
                    return results
                # breakpoint()
                output = lt_accuracies(preds, labels, self.n_test_classes, self.dist_splits_novel_base, label_type=label_type)
                output2 = lt_accuracies_splitted(preds_full, labels, self.dist_splits_novel_base, label_type=label_type)

            # print(output, flush=True)
            print(f'{prefix}Avg C/A {output["avg_accs"][0]}\t Shared (EX) {output["avg_accs"][1]}\t Unique (EX) {output["avg_accs"][2]}\t Shared (SEM) {output["avg_accs"][3]}\t Unique (SEM) {output["avg_accs"][4]}\t Shared (SEM wo ex) {output["avg_accs"][5]} ', flush=True)
            print(f'{prefix}{output["avg_accs"][0]:.2f}, {output["avg_accs"][1]:.2f}, {output["avg_accs"][2]:.2f},{output["avg_accs"][3]:.2f},{output["avg_accs"][4]:.2f}, {output["avg_accs"][5]:.2f} ',flush=True)

            results[f'{prefix}avg_acc/accuracy'] = output['avg_accs'][0]
            results[f'{prefix}avg_acc/shared_ex'] = output['avg_accs'][1]
            results[f'{prefix}avg_acc/unique_ex'] = output['avg_accs'][2]
            results[f'{prefix}avg_acc/shared_sem'] = output['avg_accs'][3]
            results[f'{prefix}avg_acc/unique_sem'] = output['avg_accs'][4]
            results[f'{prefix}avg_acc/shared_sem_wo_ex'] = output['avg_accs'][5]

            print(f'{prefix}Full Acc {output["full_accs"][0]}\t Shared (EX) {output["full_accs"][1]}\t Unique (EX) {output["full_accs"][2]}\t Shared (SEM) {output["full_accs"][3]}\t Unique (SEM) {output["full_accs"][4]}\t Shared (SEM wo ex) {output["full_accs"][5]} ', flush=True)
            print(f'{prefix}{output["full_accs"][0]:.2f}, {output["full_accs"][1]:.2f},{output["full_accs"][2]:.2f},{output["full_accs"][3]:.2f},{output["full_accs"][4]:.2f},{output["full_accs"][5]:.2f} ', flush=True)
            results[f'{prefix}full_acc/accuracy'] = output['full_accs'][0]
            results[f'{prefix}full_acc/unique_ex'] = output['full_accs'][2]
            results[f'{prefix}full_acc/shared_sem'] = output['full_accs'][3]
            results[f'{prefix}full_acc/unique_sem'] = output['full_accs'][4]
            results[f'{prefix}full_acc/shared_sem_wo_ex'] = output['full_accs'][5]

            print(f'{prefix}Splitted  Shared (EX) {output2["full_accs"][0]}\t Unique (EX) {output2["full_accs"][1]}\t Shared (SEM) {output2["full_accs"][2]}\t Unique (SEM) {output2["full_accs"][3]}\t Shared (SEM wo ex) {output2["full_accs"][4]}')
            print(f'{prefix}{output2["full_accs"][0]:.2f}, {output2["full_accs"][1]:.2f},{output2["full_accs"][2]:.2f},{output2["full_accs"][3]:.2f},{output2["full_accs"][4]:.2f}',flush=True)



            # if not self.cfg.TEST.LT_EVAL:
            #     for k,v in output['test_class_acc'].items():
            #         results[f'{prefix}val_cl_acc/{k}'] = v

        if self.cfg.TEST.LT_EVAL:
            # output = avg_accs, test_correct_list, test_count_list, test_class_acc, train_counts
            preds = torch.tensor(self._y_pred)
            labels = torch.tensor(self._y_true)
            # print('PREDS', preds[:100], flush=True)
            # print('LABELS', labels[:100], flush=True)
            output = lt_accuracies(preds, labels, len(self.train_counts), self.dist_splits, label_type=label_type)
            # print(output, flush=True)
            print(f'{prefix}Avg C/A {output["avg_accs"][0]}\t Few {output["avg_accs"][1]}\t Tail {output["avg_accs"][2]}\t Head {output["avg_accs"][3]}', flush=True)
            results[f'{prefix}avg_acc/accuracy'] = output['avg_accs'][0]
            results[f'{prefix}avg_acc/few'] = output['avg_accs'][1]
            results[f'{prefix}avg_acc/tail'] = output['avg_accs'][2]
            results[f'{prefix}avg_acc/head'] = output['avg_accs'][3]

            print(f'{prefix}Full Acc {output["full_accs"][0]}\t Few {output["full_accs"][1]}\t Tail {output["full_accs"][2]}\t Head {output["full_accs"][3]}', flush=True)
            results[f'{prefix}full_acc/accuracy'] = output['avg_accs'][0]
            results[f'{prefix}full_acc/few'] = output['avg_accs'][1]
            results[f'{prefix}full_acc/tail'] = output['avg_accs'][2]
            results[f'{prefix}full_acc/head'] = output['avg_accs'][3]


            # per class accuracy, we skip it from now on

            # for k,v in output['test_class_acc'].items():
            #     results[f'{prefix}val_cl_acc/{k}'] = v

        return results

    # self.feat_collection['text_feat'].append(features['text_feat'])
    # self.feat_collection['image_feat'].append(features['image_feat'])
    # if 'visual_ctx_feat' in features:
    #     self.feat_collection['visual_ctx_feat'].append(features['visual_ctx_feat'])

    def evaluate_retrieval_epic(self, label_type='orig'):
        results = {}
        # data_normed = LA.norm(train_LT_features.numpy(), 2, axis=-1)
        # train_LT_features = train_LT_features.numpy() / data_normed.reshape(-1, 1)
        # already filtered text features
        # from numpy import linalg as LA
        # breakpoint()
        all_text_embed = torch.vstack(self.feat_collection[label_type]['text_feat'])
        all_video_embed = torch.vstack(self.feat_collection[label_type]['image_feat'])

        if 'visual_ctx_feat' in self.feat_collection[label_type]:
            print('retrieval with vis ctx')
            all_video_ctx_embed = torch.vstack(self.feat_collection[label_type]['visual_ctx_feat'])

            t2v_scale = self.cfg.TRAINER.DECOMP_COCOOP.SCALE_FACTOR_T2V
            v2t_scale = self.cfg.TRAINER.DECOMP_COCOOP.SCALE_FACTOR_V2T

            if self.cfg.TRAINER.DECOMP_COCOOP.MLP_NARRATIONS2:
                all_mapped_narrations_embed = torch.vstack(self.feat_collection[label_type]['mapped_narrations'])

                t2v_image_features = all_video_embed.squeeze().unsqueeze(0) + t2v_scale * all_mapped_narrations_embed
                t2v_shape = t2v_image_features.shape
                t2v_image_features = t2v_image_features.view(-1, t2v_shape[-1])
                t2v_image_features = t2v_image_features / t2v_image_features.norm(dim=-1, keepdim=True)
                t2v_image_features = t2v_image_features.view(*t2v_shape)

                # narrations: 1 x BS x 512 -> change to BS x 1 x 512
                narrations = all_text_embed.squeeze().unsqueeze(1)
                # t2v_image_features: BS (narr) x BS(img) x 512
                # t2v logits
                similarity_matrix_txt = ((narrations * t2v_image_features).sum(-1))
                similarity_matrix_txt = (similarity_matrix_txt.numpy() + 1) / 2
            else:
                all_video_embed_txt_sim = all_video_embed + t2v_scale * all_video_ctx_embed.squeeze()

                all_video_embed_txt_sim = all_video_embed_txt_sim / all_video_embed_txt_sim.norm(dim=-1, keepdim=True)
                all_text_embed_txt = all_text_embed / all_text_embed.norm(dim=-1, keepdim=True)
                similarity_matrix_txt = all_text_embed_txt @ all_video_embed_txt_sim.T
                similarity_matrix_txt = (similarity_matrix_txt.numpy() + 1) / 2

            # breakpoint()
            all_text_embed = all_text_embed.unsqueeze(0)
            all_text_embed = all_text_embed + v2t_scale * all_video_ctx_embed

            v2t_shape = all_text_embed.shape
            all_text_embed = all_text_embed.view(-1, v2t_shape[-1])
            all_text_embed = all_text_embed / all_text_embed.norm(dim=-1, keepdim=True)
            all_text_embed = all_text_embed.view(*v2t_shape)

            # all_text_embed = all_text_embed / all_text_embed.norm(dim=-1, keepdim=True)

            all_video_embed = all_video_embed.unsqueeze(1)
            similarity_matrix = ((all_video_embed * all_text_embed).sum(-1))

        else:
            similarity_matrix = all_video_embed @ all_text_embed.T
            similarity_matrix_txt = None

        similarity_matrix = similarity_matrix.numpy()

        similarity_matrix = (similarity_matrix + 1) / 2
        root = '/mnt/graphics_ssd/nimble/users/annakukleva/data/epic/annotations/retrieval_annotations/'
        # video_id = pd.read_csv(root + 'EPIC_100_retrieval_test.csv').values[:, 0]
        # text_id = pd.read_csv(root + 'EPIC_100_retrieval_test_sentence.csv').values[:, 0]
        # indexes = [video_id.tolist().index(elem) for elem in text_id]
        # similarity_matrix = similarity_matrix[:, indexes]
        print(similarity_matrix.shape)
        rel_matrix = pd.read_pickle(root + 'caption_relevancy_EPIC_100_retrieval_test.pkl')
        vis_map = calculate_mAP(similarity_matrix, rel_matrix)
        # breakpoint()
        if similarity_matrix_txt is None:
            txt_map = calculate_mAP(similarity_matrix.T, rel_matrix.T)
        else:
            txt_map = calculate_mAP(similarity_matrix_txt, rel_matrix.T)
        print('mAP: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_map, txt_map, (vis_map + txt_map) / 2))
        results.update({
            'mAP/v2t': vis_map,
            'mAP/t2v': txt_map,
            'mAP/avg': (vis_map + txt_map) / 2,
        })
        vis_k_counts = calculate_k_counts(rel_matrix)
        txt_k_counts = calculate_k_counts(rel_matrix.T)
        vis_IDCG = calculate_IDCG(rel_matrix, vis_k_counts)
        txt_IDCG = calculate_IDCG(rel_matrix.T, txt_k_counts)
        vis_nDCG = calculate_nDCG(similarity_matrix, rel_matrix, k_counts=vis_k_counts, IDCG=vis_IDCG)
        txt_nDCG = calculate_nDCG(similarity_matrix.T, rel_matrix.T, k_counts=txt_k_counts, IDCG=txt_IDCG)
        print('nDCG: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_nDCG, txt_nDCG, (vis_nDCG + txt_nDCG) / 2))
        results.update({
            'nDCG/v2t': vis_nDCG,
            'nDCG/t2v': txt_nDCG,
            'nDCG/avg': (vis_nDCG + txt_nDCG) / 2,
        })
        return results
        # return {**{k: v.avg for k, v in metrics.items()}}



#
# def validate_mir(val_loader, model, criterion, args):
#     batch_time = AverageMeter('Time', ':6.2f')
#     data_time = AverageMeter('Data', ':6.2f')
#     mem = AverageMeter('Mem (GB)', ':6.1f')
#     metric_names = ['loss', 'max_margin_loss']
#     iters_per_epoch = len(val_loader) // args.update_freq
#     metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
#     progress = ProgressMeter(
#         iters_per_epoch,
#         [batch_time, data_time, mem, *metrics.values()],
#         prefix="Test: "
#     )
#
#     # switch to eval mode
#     model.eval()
#
#     all_video_embed = []
#     all_text_embed = []
#     with torch.no_grad():
#         end = time.time()
#         for i, inputs in enumerate(val_loader):
#             # measure data loading time
#             data_time.update(time.time() - end)
#
#             inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]
#             relevancies = inputs.pop()
#
#             # compute output
#             outputs = model(
#                 *inputs,
#                 use_checkpoint=args.use_checkpoint,
#                 norm_embed=args.norm_embed
#             )
#             loss_dict = criterion(outputs, weight=relevancies)
#
#             for k in loss_dict:
#                 metrics[k].update(loss_dict[k].item(), args.batch_size)
#
#             image_features = outputs['image_embed']
#             text_features = outputs['text_embed']
#             all_video_embed.append(image_features.cpu().numpy())
#             all_text_embed.append(text_features.cpu().numpy())
#
#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             mem.update(torch.cuda.max_memory_allocated() // 1e9)
#
#             if i % args.print_freq == 0:
#                 if dist_utils.is_main_process() and args.wandb:
#                     wandb.log({**{k: v.item() for k, v in loss_dict.items()}})
#                 progress.display(i)
#     progress.synchronize()
#     all_text_embed = np.vstack(all_text_embed)
#     all_video_embed = np.vstack(all_video_embed)
#     similarity_matrix = np.matmul(all_video_embed, all_text_embed.T)
#     similarity_matrix = (similarity_matrix + 1) / 2
#     video_id = pd.read_csv(args.metadata.replace('train', 'test')).values[:, 0]
#     text_id = pd.read_csv(args.metadata.replace('train', 'test_sentence')).values[:, 0]
#     indexes = [video_id.tolist().index(elem) for elem in text_id]
#     similarity_matrix = similarity_matrix[:, indexes]
#     print(similarity_matrix.shape)
#     rel_matrix = pd.read_pickle(
#         args.relevancy_path
#     )
#     vis_map = calculate_mAP(similarity_matrix, rel_matrix)
#     txt_map = calculate_mAP(similarity_matrix.T, rel_matrix.T)
#     print('mAP: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_map, txt_map, (vis_map + txt_map) / 2))
#     vis_k_counts = calculate_k_counts(rel_matrix)
#     txt_k_counts = calculate_k_counts(rel_matrix.T)
#     vis_IDCG = calculate_IDCG(rel_matrix, vis_k_counts)
#     txt_IDCG = calculate_IDCG(rel_matrix.T, txt_k_counts)
#     vis_nDCG = calculate_nDCG(similarity_matrix, rel_matrix, k_counts=vis_k_counts, IDCG=vis_IDCG)
#     txt_nDCG = calculate_nDCG(similarity_matrix.T, rel_matrix.T, k_counts=txt_k_counts, IDCG=txt_IDCG)
#     print('nDCG: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_nDCG, txt_nDCG, (vis_nDCG + txt_nDCG) / 2))
#     return {**{k: v.avg for k, v in metrics.items()}}